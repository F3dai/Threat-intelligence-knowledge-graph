import sys
import argparse
import logging
from typing import Optional, List, Dict, Any
import time
from tqdm import tqdm
from dotenv import load_dotenv
import bs4
from bs4 import BeautifulSoup
import requests

from .config import TEXT_PROCESSING_CONFIG, GEMINI_CONFIG 
from .extractors import GraphExtractor, GeminiGraphExtractor
from .storage import Neo4jStore

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Constants for Gemini Large Chunking ---
# Estimate based on 1M context window, leaving buffer for prompt, output, and estimation errors
# Target ~800k tokens. Using 4 chars/token approximation.
GEMINI_TARGET_CHUNK_TOKENS = 800000
# Rough character count target
GEMINI_TARGET_CHUNK_CHARS = GEMINI_TARGET_CHUNK_TOKENS * 4
# Minimum delay in seconds between Gemini API calls (60s / 5 RPM + 1s buffer)
GEMINI_MIN_REQUEST_INTERVAL = 13


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

    # --- Document Loading (Remains the same) ---
    try:
        logger.info(f"Loading document from {url}")
        response = requests.get(url, timeout=30) # Add timeout
        response.raise_for_status() # Check for HTTP errors

        soup = BeautifulSoup(response.content, 'html.parser')
        # Attempt to find main content area if possible (heuristic)
        main_content = soup.find('article') or soup.find('main') or soup.body
        if main_content:
             text_content = main_content.get_text(separator="\n", strip=True)
        else:
             text_content = soup.get_text(separator="\n", strip=True) # Fallback

        if not text_content.strip():
             logger.warning(f"No text content found after parsing {url}")
             return stats

        logger.info(f"Loaded and parsed content from {url}. Approx {len(text_content)} chars.")
        stats["estimated_tokens"] = len(text_content) // 4 # Rough estimate for the whole doc

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to load document from {url}: {e}")
        return stats
    except Exception as e:
        logger.error(f"Failed to parse document content from {url}: {e}")
        return stats


    # --- Chunking Strategy ---
    documents_to_process = []
    is_gemini_extractor = isinstance(extractor, GeminiGraphExtractor)

    if is_gemini_extractor:
        logger.info(f"Using Gemini extractor. Applying large chunking strategy (target chars: {GEMINI_TARGET_CHUNK_CHARS}).")
        # Check if the whole document might fit (with buffer)
        if len(text_content) <= GEMINI_TARGET_CHUNK_CHARS:
            logger.info("Entire document content is within the target chunk size. Processing as one chunk.")
            documents_to_process.append({
                "page_content": text_content,
                "metadata": {"source": url, "chunk_index": 0, "total_chunks": 1}
            })
        else:
            logger.info(f"Document content exceeds target chunk size. Splitting into large chunks...")
            # Use simple character splitting for large chunks, overlap might be less critical
            # Overlap can be adjusted if needed. Using a smaller fixed overlap for large chunks.
            large_chunks = chunk_text_by_char_limit(text_content, GEMINI_TARGET_CHUNK_CHARS, overlap=500)
            for i, chunk in enumerate(large_chunks):
                 documents_to_process.append({
                      "page_content": chunk,
                      "metadata": {"source": url, "chunk_index": i, "total_chunks": len(large_chunks)}
                 })
            logger.info(f"Split document into {len(documents_to_process)} large chunks.")

    else: # Default chunking for OpenAI or other extractors
        logger.info(f"Using non-Gemini extractor. Applying default chunking (size: {default_chunk_size}, overlap: {default_chunk_overlap}).")
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


    # --- Process Chunks with Rate Limiting for Gemini ---
    last_api_call_time = 0 # Track time of the last call for Gemini RPM limiting

    for idx, document_chunk in tqdm(enumerate(documents_to_process), total=len(documents_to_process), desc="Processing Chunks"):
        chunk_meta = document_chunk["metadata"]
        logger.info(f"Processing chunk {chunk_meta['chunk_index'] + 1}/{chunk_meta['total_chunks']}...")

        # ** Gemini Rate Limiting **
        if is_gemini_extractor:
            current_time = time.time()
            time_since_last_call = current_time - last_api_call_time
            if time_since_last_call < GEMINI_MIN_REQUEST_INTERVAL:
                wait_time = GEMINI_MIN_REQUEST_INTERVAL - time_since_last_call
                logger.warning(f"Gemini RPM limit (5 RPM): Waiting for {wait_time:.2f} seconds before next API call...")
                time.sleep(wait_time)

            # Record the time *before* the API call
            last_api_call_time = time.time()


        # --- Extract knowledge graph ---
        try:
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
         logger.warning("REMINDER: Gemini 2.5 Pro Experimental has a very low 25 Requests Per Day (RPD) limit!")
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
        if model_provider.lower() == "gemini":
            # Note: Gemini extractor doesn't use chunk_size/overlap from config directly anymore
            extractor = GeminiGraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose
                # Add other relevant Gemini parameters if needed (api_key, model name override etc)
            )
            logger.info("Initialized Gemini graph extractor.")
            logger.warning("Gemini model selected: Using large chunking and rate limiting (5 RPM, 25 RPD).")
        else:
            extractor = GraphExtractor(
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                verbose=verbose
                # Add other relevant OpenAI parameters if needed
            )
            logger.info("Initialized OpenAI graph extractor")
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

        logger.info(
            f"Finished processing URL: {url} - Added {url_stats['total_nodes']} nodes and "
            f"{url_stats['total_relationships']} relationships in {url_stats['processing_time']:.2f} seconds. "
            f"API Calls: {url_stats['api_calls']}."
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
        choices=["openai", "gemini"],
        default="openai",
        help="Model provider to use (openai or gemini)"
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
    logger.info(f"Processed {len(args.urls)} URLs.")
    logger.info(f"Total Chunks: {successful_chunks + failed_chunks} ({successful_chunks} successful, {failed_chunks} failed)")
    logger.info(f"Total API Calls: {total_api_calls}")
    logger.info(f"Total Nodes Added: {total_nodes}")
    logger.info(f"Total Relationships Added: {total_relationships}")
    logger.info(f"Total Processing Time: {total_time:.2f} seconds")
    logger.info("="*78)
    if args.model == 'gemini':
        logger.warning("Reminder: Gemini 2.5 Pro Experimental has a 25 Requests Per Day (RPD) limit.")


if __name__ == "__main__":
    main()