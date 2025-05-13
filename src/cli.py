#!/usr/bin/env python3
"""
Command-line interface for the Threat Intelligence Knowledge Graph tool.
"""

import argparse
import sys
import logging
from threading import Thread
from typing import List, Dict, Any

from threat_intel_kg.main import process_urls
from threat_intel_kg.config import DEFAULT_ALLOWED_NODES, DEFAULT_ALLOWED_RELATIONSHIPS


def print_summary(stats: List[Dict[str, Any]], model_provider: str = "openai") -> None:
    """
    Print a summary of the processing statistics.
    
    Args:
        stats: The processing statistics.
        model_provider: The model provider used.
    """
    total_nodes = sum(s.get("total_nodes", 0) for s in stats)
    total_relationships = sum(s.get("total_relationships", 0) for s in stats)
    successful_chunks = sum(s.get("successful_chunks", 0) for s in stats)
    failed_chunks = sum(s.get("failed_chunks", 0) for s in stats)
    total_time = sum(s.get("processing_time", 0) for s in stats)
    total_api_calls = sum(s.get("api_calls", 0) for s in stats)
    
    # Model name is already correctly formatted in the CLI
    model_name = model_provider
    
    print("\n=== Processing Summary ===")
    print(f"Processed {len(stats)} URLs using {model_name} extractor")
    print(f"Successful chunks: {successful_chunks}")
    print(f"Failed chunks: {failed_chunks}")
    if model_provider != "ner":
        print(f"Total API calls: {total_api_calls}")
    print(f"Total nodes added: {total_nodes}")
    print(f"Total relationships added: {total_relationships}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("==========================")
    
    if model_provider == "gemini-2.5-pro":
        print("NOTE: Gemini-2.5-Pro has rate limits of 5 RPM and 25 RPD")
    elif model_provider == "gemini-2.0-flash":
        print("NOTE: Gemini-2.0-Flash has rate limits that may affect usage")
    elif model_provider == "gemini-2.5-flash-preview-04-17":
        print("NOTE: Gemini-2.5-Flash-Preview has rate limits that may affect usage")
    elif model_provider == "claude-3-5-haiku":
        print("NOTE: Claude-3.5-Haiku may have request volume limitations on your plan")
    elif model_provider == "claude-3-5-sonnet-20240620":
        print("NOTE: Claude-3.5-Sonnet may have request volume limitations on your plan")


def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Extract threat intelligence from security reports and build a knowledge graph."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process URLs to extract threat intelligence")
    process_parser.add_argument(
        "urls", 
        nargs="+", 
        help="URLs of threat reports to process"
    )
    process_parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Output verbose processing information"
    )
    process_parser.add_argument(
        "--model",
        choices=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", 
                "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.5-flash-preview-04-17", 
                "claude-3-5-haiku", "claude-3-5-sonnet-20240620", "ner"],
        default="gpt-3.5-turbo",
        help="Model to use (OpenAI: gpt-3.5-turbo, gpt-4-turbo, gpt-4o; Gemini: gemini-2.5-pro, gemini-2.0-flash, gemini-2.5-flash-preview-04-17; Claude: claude-3-5-haiku, claude-3-5-sonnet-20240620; or ner)"
    )
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle version command
    if args.command == "version":
        print(f"Threat Intelligence Knowledge Graph v0.3.0")
        sys.exit(0)
    
    # Handle process command
    if args.command == "process":
        # Set up logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        # Always set DEBUG level for NER extractor when using it with verbose
        if args.verbose and args.model == "ner":
            logging.getLogger("threat_intel_kg.extractors.ner_extractor").setLevel(logging.DEBUG)
        
        # Process URLs
        stats = process_urls(
            args.urls,
            allowed_nodes=DEFAULT_ALLOWED_NODES,
            allowed_relationships=DEFAULT_ALLOWED_RELATIONSHIPS,
            verbose=args.verbose,
            model_provider=args.model
        )
        
        # Print summary
        print_summary(stats, args.model)


if __name__ == "__main__":
    main()