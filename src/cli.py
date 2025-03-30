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


def print_summary(stats: List[Dict[str, Any]]) -> None:
    """
    Print a summary of the processing statistics.
    
    Args:
        stats: The processing statistics.
    """
    total_nodes = sum(s["total_nodes"] for s in stats)
    total_relationships = sum(s["total_relationships"] for s in stats)
    successful_chunks = sum(s["successful_chunks"] for s in stats)
    failed_chunks = sum(s["failed_chunks"] for s in stats)
    total_time = sum(s["processing_time"] for s in stats)
    
    print("\n=== Processing Summary ===")
    print(f"Processed {len(stats)} URLs")
    print(f"Successful chunks: {successful_chunks}")
    print(f"Failed chunks: {failed_chunks}")
    print(f"Total nodes added: {total_nodes}")
    print(f"Total relationships added: {total_relationships}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("==========================")


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
        choices=["openai", "gemini"],
        default="openai",
        help="Model provider to use (openai or gemini)"
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
        from threat_intel_kg import __version__
        print(f"Threat Intelligence Knowledge Graph v{__version__}")
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
        
        # Process URLs
        stats = process_urls(
            args.urls,
            allowed_nodes=DEFAULT_ALLOWED_NODES,
            allowed_relationships=DEFAULT_ALLOWED_RELATIONSHIPS,
            verbose=args.verbose,
            model_provider=args.model
        )
        
        # Print summary
        print_summary(stats)


if __name__ == "__main__":
    main()