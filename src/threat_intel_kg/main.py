import sys
import argparse
import logging
import os
from typing import Optional, List, Dict, Any
import time
from tqdm import tqdm
from dotenv import load_dotenv
import bs4
from bs4 import BeautifulSoup
import requests

from .config import TEXT_PROCESSING_CONFIG, GEMINI_CONFIG, CLAUDE_CONFIG
from .extractors import OpenAIGraphExtractor, GeminiGraphExtractor, ClaudeGraphExtractor, NERExtractor
from .storage import Neo4jStore

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Constants for Large Context Models Chunking ---
# Gemini Pro settings
# Estimate based on 1M context window, leaving buffer for prompt, output, and estimation errors
# Target ~800k tokens. Using 4 chars/token approximation.
GEMINI_PRO_TARGET_CHUNK_TOKENS = 800000
# Rough character count target
GEMINI_PRO_TARGET_CHUNK_CHARS = GEMINI_PRO_TARGET_CHUNK_TOKENS * 4

# Gemini Flash settings
# Estimate based on 128k context window, leaving buffer for prompt, output, and estimation errors
# Target ~100k tokens. Using 4 chars/token approximation.
GEMINI_FLASH_TARGET_CHUNK_TOKENS = 100000
# Rough character count target
GEMINI_FLASH_TARGET_CHUNK_CHARS = GEMINI_FLASH_TARGET_CHUNK_TOKENS * 4

# Minimum delay in seconds between Gemini API calls (60s / 5 RPM + 1s buffer)
GEMINI_MIN_REQUEST_INTERVAL = 13

# Claude Haiku settings
# Estimate based on 200k context window, leaving buffer for prompt, output, and estimation errors
# Target ~160k tokens. Using 4 chars/token approximation.
CLAUDE_HAIKU_TARGET_CHUNK_TOKENS = 160000
# Rough character count target
CLAUDE_HAIKU_TARGET_CHUNK_CHARS = CLAUDE_HAIKU_TARGET_CHUNK_TOKENS * 4

# Claude Sonnet settings
# Estimate based on 400k context window, leaving buffer for prompt, output, and estimation errors
# Target ~320k tokens. Using 4 chars/token approximation.
CLAUDE_SONNET_TARGET_CHUNK_TOKENS = 320000
# Rough character count target
CLAUDE_SONNET_TARGET_CHUNK_CHARS = CLAUDE_SONNET_TARGET_CHUNK_TOKENS * 4

# Minimum delay in seconds between Claude API calls (60s / 5 RPM + 1s buffer)
CLAUDE_MIN_REQUEST_INTERVAL = 13


def chunk_text_by_char_limit(text: str, char_limit: int, overlap: int = 200) -> List[str]:
    """Splits text into chunks based on a character limit."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + char_limit
        # Ensure we don't cut mid-word if possible, find last space before end
        # Simple approach: just slice. More complex logic could be added for word boundaries.
        chunk = text[start:end]
        if not chunk: # Should not happen with positive char_limit, but safety check
            break
        chunks.append(chunk)
        start += char_limit - overlap
        if start >= len(text):
             break # Prevent infinite loop if overlap is too large / chunk_limit too small
    return chunks


def process_url(
    url: str,
    extractor: Any,  # Can be either GraphExtractor or GeminiGraphExtractor
    store: Neo4jStore,
    # Keep original chunking defaults for non-Gemini models or as fallback
    default_chunk_size: int = TEXT_PROCESSING_CONFIG["chunk_size"],
    default_chunk_overlap: int = TEXT_PROCESSING_CONFIG["chunk_overlap"]
) -> Dict[str, Any]:
    """
    Process a URL to extract and store a knowledge graph.
    Adapts chunking strategy and adds delays for Gemini models.

    Args:
        url: The URL to process.
        extractor: The graph extractor instance to use.
        store: The Neo4j store instance to use.
        default_chunk_size: Default character chunk size (for non-Gemini).
        default_chunk_overlap: Default character chunk overlap (for non-Gemini).

    Returns:
        Statistics about the processing.
    """
    start_time = time.time()
    stats = {
        "url": url,
        "successful_chunks": 0,
        "failed_chunks": 0,
        "total_nodes": 0,
        "total_relationships": 0,
        "estimated_tokens": 0,
        "api_calls": 0,
    }

    # --- Document Loading - Support both URLs and local file paths ---
    try:
        logger.info(f"Loading document from {url}")
        
        # Check if this is a local file path
        if os.path.exists(url) and os.path.isfile(url):
            try:
                # Load local file
                with open(url, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                if not text_content.strip():
                    logger.warning(f"No text content found in file {url}")
                    return stats
                
                logger.info(f"Loaded and parsed content from local file {url}. Approx {len(text_content)} chars.")
                stats["estimated_tokens"] = len(text_content) // 4  # Rough estimate for the whole doc
                
            except Exception as e:
                logger.error(f"Failed to read local file {url}: {e}")
                return stats
        else:
            # Treat as URL
            response = requests.get(url, timeout=30)  # Add timeout
            response.raise_for_status()  # Check for HTTP errors

            soup = BeautifulSoup(response.content, 'html.parser')
            # Attempt to find main content area if possible (heuristic)
            main_content = soup.find('article') or soup.find('main') or soup.body
            if main_content:
                text_content = main_content.get_text(separator="\n", strip=True)
            else:
                text_content = soup.get_text(separator="\n", strip=True)  # Fallback

            if not text_content.strip():
                logger.warning(f"No text content found after parsing {url}")
                return stats

            logger.info(f"Loaded and parsed content from URL {url}. Approx {len(text_content)} chars.")
            stats["estimated_tokens"] = len(text_content) // 4  # Rough estimate for the whole doc

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to load document from {url}: {e}")
        return stats
    except Exception as e:
        logger.error(f"Failed to parse document content from {url}: {e}")
        return stats


    # --- Chunking Strategy ---
    documents_to_process = []
    is_gemini_extractor = isinstance(extractor, GeminiGraphExtractor)
    is_claude_extractor = isinstance(extractor, ClaudeGraphExtractor)
    is_ner_extractor = isinstance(extractor, NERExtractor)
    is_openai_extractor = isinstance(extractor, OpenAIGraphExtractor)

    if is_ner_extractor:
        # NER extractor can process the entire document at once
        logger.info("Using NER extractor. Processing entire document without chunking.")
        documents_to_process.append({
            "page_content": text_content,
            "metadata": {"source": url, "chunk_index": 0, "total_chunks": 1}
        })

    elif is_gemini_extractor:
        # Determine which Gemini model is being used to select the appropriate chunk size
        model_name = getattr(extractor, 'model_name', '')
        
        if "flash" in model_name.lower():
            target_chunk_chars = GEMINI_FLASH_TARGET_CHUNK_CHARS
            logger.info(f"Using Gemini Flash extractor. Applying medium chunking strategy (target chars: {target_chunk_chars}).")
        else:
            # Default to Gemini Pro settings
            target_chunk_chars = GEMINI_PRO_TARGET_CHUNK_CHARS
            logger.info(f"Using Gemini Pro extractor. Applying large chunking strategy (target chars: {target_chunk_chars}).")
            
        # Check if the whole document might fit (with buffer)
        if len(text_content) <= target_chunk_chars:
            logger.info("Entire document content is within the target chunk size. Processing as one chunk.")
            documents_to_process.append({
                "page_content": text_content,
                "metadata": {"source": url, "chunk_index": 0, "total_chunks": 1}
            })
        else:
            logger.info(f"Document content exceeds target chunk size. Splitting into chunks...")
            # Use simple character splitting for large chunks, overlap might be less critical
            # Overlap can be adjusted if needed. Using a smaller fixed overlap for large chunks.
            large_chunks = chunk_text_by_char_limit(text_content, target_chunk_chars, overlap=500)
            for i, chunk in enumerate(large_chunks):
                 documents_to_process.append({
                      "page_content": chunk,
                      "metadata": {"source": url, "chunk_index": i, "total_chunks": len(large_chunks)}
                 })
            logger.info(f"Split document into {len(documents_to_process)} chunks.")
    
    elif is_claude_extractor:
        # Determine which Claude model is being used to select the appropriate chunk size
        model_name = getattr(extractor, 'model_name', '')
        
        if "sonnet" in model_name.lower():
            target_chunk_chars = CLAUDE_SONNET_TARGET_CHUNK_CHARS
            logger.info(f"Using Claude Sonnet extractor. Applying large chunking strategy (target chars: {target_chunk_chars}).")
        else:
            # Default to Claude Haiku settings
            target_chunk_chars = CLAUDE_HAIKU_TARGET_CHUNK_CHARS
            logger.info(f"Using Claude Haiku extractor. Applying medium chunking strategy (target chars: {target_chunk_chars}).")
            
        # Check if the whole document might fit (with buffer)
        if len(text_content) <= target_chunk_chars:
            logger.info("Entire document content is within the target chunk size. Processing as one chunk.")
            documents_to_process.append({
                "page_content": text_content,
                "metadata": {"source": url, "chunk_index": 0, "total_chunks": 1}
            })
        else:
            logger.info(f"Document content exceeds target chunk size. Splitting into chunks...")
            # Use simple character splitting for large chunks, overlap might be less critical
            # Overlap can be adjusted if needed. Using a smaller fixed overlap for large chunks.
            large_chunks = chunk_text_by_char_limit(text_content, target_chunk_chars, overlap=500)
            for i, chunk in enumerate(large_chunks):
                 documents_to_process.append({
                      "page_content": chunk,
                      "metadata": {"source": url, "chunk_index": i, "total_chunks": len(large_chunks)}
                 })
            logger.info(f"Split document into {len(documents_to_process)} chunks.")

    else: # Default chunking for OpenAI or other extractors
        logger.info(f"Using non-Gemini/Claude extractor. Applying default chunking (size: {default_chunk_size}, overlap: {default_chunk_overlap}).")
        # Using the existing structure slightly adapted
        chunks = chunk_text_by_char_limit(text_content, default_chunk_size, default_chunk_overlap)
        for i, chunk in enumerate(chunks):
            documents_to_process.append({
                "page_content": chunk,
                "metadata": {"source": url, "chunk_index": i, "total_chunks": len(chunks)}
            })
        logger.info(f"Split document into {len(documents_to_process)} chunks.")

    if not documents_to_process:
         logger.warning("No document chunks created for processing.")
         return stats


    # --- Process Chunks with Rate Limiting for API-based models ---
    last_api_call_time = 0 # Track time of the last call for API RPM limiting

    for idx, document_chunk in tqdm(enumerate(documents_to_process), total=len(documents_to_process), desc="Processing Chunks"):
        chunk_meta = document_chunk["metadata"]
        logger.info(f"Processing chunk {chunk_meta['chunk_index'] + 1}/{chunk_meta['total_chunks']}...")

        # ** API Rate Limiting **
        if is_gemini_extractor:
            current_time = time.time()
            time_since_last_call = current_time - last_api_call_time
            if time_since_last_call < GEMINI_MIN_REQUEST_INTERVAL:
                wait_time = GEMINI_MIN_REQUEST_INTERVAL - time_since_last_call
                logger.warning(f"Gemini RPM limit (5 RPM): Waiting for {wait_time:.2f} seconds before next API call...")
                time.sleep(wait_time)

            # Record the time *before* the API call
            last_api_call_time = time.time()
        
        elif is_claude_extractor:
            current_time = time.time()
            time_since_last_call = current_time - last_api_call_time
            if time_since_last_call < CLAUDE_MIN_REQUEST_INTERVAL:
                wait_time = CLAUDE_MIN_REQUEST_INTERVAL - time_since_last_call
                logger.warning(f"Claude RPM limit (5 RPM): Waiting for {wait_time:.2f} seconds before next API call...")
                time.sleep(wait_time)

            # Record the time *before* the API call
            last_api_call_time = time.time()
            
        # NERExtractor doesn't require rate limiting as it's a local extractor


        # --- Extract knowledge graph ---
        try:
            # Enable more verbose logging for debug for NER extractor
            if isinstance(extractor, NERExtractor) and extractor.verbose:
                logging.getLogger("threat_intel_kg.extractors.ner_extractor").setLevel(logging.DEBUG)
                
            knowledge_graph = extractor.extract_from_document(document_chunk["page_content"])
            stats["api_calls"] += 1 # Increment API call count *after* successful call attempt
        except Exception as e:
             # Catch potential errors during the API call itself if not handled within extract_from_document
             logger.error(f"Extraction failed for chunk {chunk_meta['chunk_index'] + 1} due to extractor error: {e}", exc_info=True)
             knowledge_graph = None # Ensure KG is None if extraction fails catastrophically
             stats["failed_chunks"] += 1
             # Optionally add a delay even on failure for Gemini to avoid rapid retries?
             # if is_gemini_extractor: time.sleep(GEMINI_MIN_REQUEST_INTERVAL)
             continue # Skip storing for this chunk


        # --- Store knowledge graph ---
        if knowledge_graph and knowledge_graph.nodes:
            logger.info(f"Extracted {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.rels)} relationships from chunk {chunk_meta['chunk_index'] + 1}.")
            # Pass the original chunk dict including metadata to the store function
            success = store.store_knowledge_graph(knowledge_graph, document_chunk)

            if success:
                stats["successful_chunks"] += 1
                stats["total_nodes"] += len(knowledge_graph.nodes)
                stats["total_relationships"] += len(knowledge_graph.rels)
            else:
                stats["failed_chunks"] += 1
                logger.warning(f"Failed to store knowledge graph from chunk {chunk_meta['chunk_index'] + 1}")
        else:
            logger.warning(f"No knowledge graph extracted or graph was empty for chunk {chunk_meta['chunk_index'] + 1}")
            # If extract_from_document returns None, it's already logged within the extractor.
            # We count it as failed here.
            stats["failed_chunks"] += 1
            # Do we need a delay here for Gemini if extraction returns None?
            # If the API call was made but returned nothing valid, we should respect the delay.
            # The delay is placed *before* the next call, so it's implicitly handled.


    stats["processing_time"] = time.time() - start_time
    logger.info(f"Finished processing {url}. API Calls: {stats['api_calls']}, Successful Chunks: {stats['successful_chunks']}, Failed Chunks: {stats['failed_chunks']}.")
    if is_gemini_extractor:
        # Different warnings for Pro vs Flash models
        model_name = getattr(extractor, 'model_name', '').lower()
        if 'flash' in model_name:
            logger.warning("REMINDER: Gemini Flash models have rate limits that may affect usage!")
        else:
            logger.warning("REMINDER: Gemini-2.5-Pro has a very low 25 Requests Per Day (RPD) limit!")
    elif is_claude_extractor:
        # Different warnings for different Claude models
        model_name = getattr(extractor, 'model_name', '').lower()
        if 'sonnet' in model_name:
            logger.warning("REMINDER: Claude-3.5-Sonnet may have request volume limitations on your plan!")
        else:
            logger.warning("REMINDER: Claude-3.5-Haiku may have request volume limitations on your plan!")
    return stats


def process_urls(
    urls: List[str],
    allowed_nodes: Optional[List[str]] = None,
    allowed_relationships: Optional[List[str]] = None,
    verbose: bool = False,
    model_provider: str = "openai"
) -> List[Dict[str, Any]]:
    """
    Process multiple URLs to extract and store knowledge graphs.
    """
    extractor = None 
    store = None 
    
    # Initialize extractor (error handling as before)
    try:
        if model_provider.lower() == "gemini-2.5-pro":
            # Note: Gemini extractor doesn't use chunk_size/overlap from config directly anymore
            extractor = GeminiGraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose,
                model="gemini-2.5-pro-exp-03-25"  # Explicitly set model name
            )
            logger.info("Initialized Gemini-2.5-Pro graph extractor.")
            logger.warning("Gemini-2.5-Pro model selected: Using large chunking and rate limiting (5 RPM, 25 RPD).")
        elif model_provider.lower() == "gemini-2.0-flash":
            extractor = GeminiGraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose,
                model="gemini-2.0-flash"  # Explicitly set model name for Flash
            )
            logger.info("Initialized Gemini-2.0-Flash graph extractor.")
            logger.warning("Gemini-2.0-Flash model selected: Using appropriate chunking and rate limiting.")
        elif model_provider.lower() == "gemini-2.5-flash-preview-04-17":
            extractor = GeminiGraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose,
                model="gemini-2.5-flash-preview-04-17"  # New Gemini 2.5 Flash model
            )
            logger.info("Initialized Gemini-2.5-Flash-Preview graph extractor.")
            logger.warning("Gemini-2.5-Flash-Preview model selected: Using appropriate chunking and rate limiting.")
        elif model_provider.lower() == "claude-3-5-haiku":
            extractor = ClaudeGraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose,
                model="claude-3-5-haiku-latest"  # Explicitly set model name
            )
            logger.info("Initialized Claude-3.5-Haiku graph extractor.")
            logger.warning("Claude-3.5-Haiku model selected: Using appropriate chunking and rate limiting.")
        elif model_provider.lower() == "claude-3-5-sonnet-20240620":
            extractor = ClaudeGraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose,
                model="claude-3-5-sonnet-20240620"  # Explicitly set model name for Sonnet
            )
            logger.info("Initialized Claude-3.5-Sonnet graph extractor.")
            logger.warning("Claude-3.5-Sonnet model selected: Using large context chunking and rate limiting.")
        elif model_provider.lower() == "ner":
            # For NER extractor, use wildcard allowed nodes and relationships to capture all entities
            # from the STIXnet patterns which are different from the default model
            extractor = NERExtractor(
                allowed_nodes=['*'],  # Allow all node types from STIXnet
                allowed_relationships=['*'],  # Allow all relationship types from STIXnet
                verbose=verbose
            )
            logger.info("Initialized NER graph extractor (pattern-based, no API calls).")
        elif model_provider.lower() == "gpt-4-turbo":
            extractor = OpenAIGraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose,
                model="gpt-4-turbo"  # Explicitly set model name
            )
            logger.info("Initialized GPT-4-Turbo graph extractor")
        elif model_provider.lower() == "gpt-4o":
            extractor = OpenAIGraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose,
                model="gpt-4o"  # Use the GPT-4o model
            )
            logger.info("Initialized GPT-4o graph extractor")
        else:  # Default to gpt-3.5-turbo
            extractor = OpenAIGraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose,
                model="gpt-3.5-turbo-16k"  # Explicitly set model name
            )
            logger.info("Initialized GPT-3.5-Turbo graph extractor")
    except Exception as e:
        logger.error(f"Failed to initialize graph extractor: {e}", exc_info=True)
        sys.exit(1) # Exit if extractor fails

    # Initialize store
    try:
        store = Neo4jStore()
        logger.info("Initialized Neo4j store")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j store: {e}", exc_info=True)
        sys.exit(1) # Exit if store fails

    # Process each URL
    all_stats = []
    for url in urls:
        logger.info(f"Processing URL: {url}")
        # Pass extractor and store instances, let process_url handle chunking details
        url_stats = process_url(url, extractor, store)
        all_stats.append(url_stats)

        # Make sure we have all the expected fields
        if 'processing_time' in url_stats:
            time_info = f" in {url_stats['processing_time']:.2f} seconds"
        else:
            time_info = ""
            
        api_calls_info = f" API Calls: {url_stats.get('api_calls', 0)}." if model_provider != 'ner' else ""

        logger.info(
            f"Finished processing URL: {url} - Added {url_stats.get('total_nodes', 0)} nodes and "
            f"{url_stats.get('total_relationships', 0)} relationships{time_info}.{api_calls_info}"
        )

    return all_stats


def main():
    """
    Main function to execute the knowledge graph construction.
    """
    parser = argparse.ArgumentParser(
        description="Construct a knowledge graph from threat reports using AI models and Neo4j."
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        help="URLs of threat reports to process"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output verbose processing information"
    )
    parser.add_argument(
        "--model",
        choices=["openai", "gemini", "ner"],
        default="openai",
        help="Model provider to use (openai, gemini, or ner)"
    )
    # Potentially add arguments for allowed_nodes, allowed_relationships if needed
    args = parser.parse_args()

    # Set log level based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence overly verbose libraries if needed
    # logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Process URLs
    stats = process_urls(args.urls, verbose=args.verbose, model_provider=args.model)

    # Print overall statistics
    total_nodes = sum(s.get("total_nodes", 0) for s in stats)
    total_relationships = sum(s.get("total_relationships", 0) for s in stats)
    successful_chunks = sum(s.get("successful_chunks", 0) for s in stats)
    failed_chunks = sum(s.get("failed_chunks", 0) for s in stats)
    total_api_calls = sum(s.get("api_calls", 0) for s in stats)
    total_time = sum(s.get("processing_time", 0) for s in stats)

    logger.info("="*30 + " Processing Summary " + "="*30)
    # Use model name directly from CLI
    model_name = args.model
        
    logger.info(f"Processed {len(args.urls)} URLs using the {model_name} extractor.")
    logger.info(f"Total Chunks: {successful_chunks + failed_chunks} ({successful_chunks} successful, {failed_chunks} failed)")
    if args.model != 'ner':
        logger.info(f"Total API Calls: {total_api_calls}")
    logger.info(f"Total Nodes Added: {total_nodes}")
    logger.info(f"Total Relationships Added: {total_relationships}")
    logger.info(f"Total Processing Time: {total_time:.2f} seconds")
    logger.info("="*78)
    if args.model == 'gemini-2.5-pro':
        logger.warning("Reminder: Gemini-2.5-Pro has a 25 Requests Per Day (RPD) limit.")
    elif args.model == 'gemini-2.0-flash':
        logger.warning("Reminder: Gemini-2.0-Flash has rate limits that may affect usage.")
    elif args.model == 'gemini-2.5-flash-preview-04-17':
        logger.warning("Reminder: Gemini-2.5-Flash-Preview has rate limits that may affect usage.")
    elif args.model == 'claude-3-5-haiku':
        logger.warning("Reminder: Claude-3.5-Haiku may have request volume limitations on your plan.")
    elif args.model == 'claude-3-5-sonnet-20240620':
        logger.warning("Reminder: Claude-3.5-Sonnet may have request volume limitations on your plan.")


if __name__ == "__main__":
    main()