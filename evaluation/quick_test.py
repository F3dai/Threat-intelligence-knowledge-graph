#!/usr/bin/env python3
"""
Quick test script for the model evaluation framework.

This script runs a limited evaluation with 1-2 models and 1 article
to demonstrate the evaluation and visualization framework more quickly.

Usage:
    python3 evaluation/quick_test.py 
    
    # Or specify a model and article
    python3 evaluation/quick_test.py --models gpt-4o --article https://example.com/report
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import from the evaluation module
from evaluation.model_evaluation import (
    run_evaluation, create_output_directory, 
    MODELS as ALL_MODELS, TEST_ARTICLES as DEFAULT_ARTICLES
)
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default limited test settings
DEFAULT_TEST_MODELS = [
    "gpt-3.5-turbo",  # Fast OpenAI model
    "ner"             # Rule-based (no API cost)
]

DEFAULT_TEST_ARTICLE = [
    "https://www.microsoft.com/en-us/security/blog/2023/08/24/fog-of-war-how-the-ukraine-conflict-transformed-the-cyber-threat-landscape/"
]

def main():
    """Run a quick test of the model evaluation framework."""
    parser = argparse.ArgumentParser(description="Run a quick test of the model evaluation framework")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, 
                        help=f"Models to evaluate (defaults to {DEFAULT_TEST_MODELS})")
    parser.add_argument("--article", help="URL of article to test (defaults to Microsoft security blog article)")
    parser.add_argument("--parallel", action="store_true", help="Run evaluations in parallel")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Use provided values or defaults
    models_to_evaluate = args.models if args.models else DEFAULT_TEST_MODELS
    articles_to_test = [args.article] if args.article else DEFAULT_TEST_ARTICLE
    
    logger.info("Starting quick test of model evaluation framework")
    
    # Create output directory
    output_dir = create_output_directory()
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    # Create visualization directory
    charts_dir = os.path.join(output_dir, "charts")
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    
    logger.info(f"Testing {len(models_to_evaluate)} models on {len(articles_to_test)} article(s)")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Run evaluation
    evaluation_results = run_evaluation(
        models=models_to_evaluate,
        articles=articles_to_test,
        output_file=results_file,
        parallel=args.parallel,
        verbose=args.verbose
    )
    
    # Generate our own visualizations - simplified version
    logger.info("Generating visualizations...")
    
    # Extract data from results
    models = []
    processing_times = []
    nodes = []
    relationships = []
    
    for result in evaluation_results["results"]:
        if "error" not in result:
            models.append(result["model"])
            processing_times.append(result["total_processing_time"])
            nodes.append(result["total_nodes"])
            relationships.append(result["total_relationships"])
    
    # Create data frame for plotting
    df = pd.DataFrame({
        'Model': models,
        'Processing Time (s)': processing_times,
        'Nodes': nodes,
        'Relationships': relationships
    })
    
    # Create simple bar chart for processing time
    fig_time = px.bar(df, x='Model', y='Processing Time (s)', title='Processing Time by Model')
    fig_time.write_html(os.path.join(charts_dir, 'processing_time.html'))
    
    # Create simple bar chart for entity extraction
    fig_entity = px.bar(df, x='Model', y=['Nodes', 'Relationships'], 
                       title='Entity Extraction by Model',
                       barmode='group')
    fig_entity.write_html(os.path.join(charts_dir, 'entity_extraction.html'))
    
    # Create a simple HTML report that doesn't require fetch
    with open(os.path.join(charts_dir, 'simple_report.html'), 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .chart {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Evaluation Report</h1>
            <p>Generated on {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</p>
            
            <h2>Summary</h2>
            <p>Tested {len(models)} models on {len(articles_to_test)} articles</p>
            <p>Best model by speed: {evaluation_results["summary"]["fastest_model"]}</p>
            <p>Best model by detail: {evaluation_results["summary"]["most_detailed_model"]}</p>
            
            <h2>Results</h2>
            <p>See individual chart files for detailed visualizations.</p>
            <ul>
                <li><a href="processing_time.html" target="_blank">Processing Time Chart</a></li>
                <li><a href="entity_extraction.html" target="_blank">Entity Extraction Chart</a></li>
            </ul>
        </body>
        </html>
        """)
    
    logger.info(f"Test completed. Results available at: {output_dir}")
    logger.info(f"Interactive report: {os.path.join(charts_dir, 'simple_report.html')}")
    
    return output_dir

if __name__ == "__main__":
    main()