#!/usr/bin/env python3
"""
Model evaluation script for the Threat Intelligence Knowledge Graph.

This script evaluates different models based on:
1. Speed (time to process)
2. Extraction quality (nodes/relationships extracted)
3. Accuracy (compared to a reference dataset)

Usage:
    python3 evaluation/model_evaluation.py
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from threat_intel_kg.extractors import (
    GraphExtractor, GeminiGraphExtractor, ClaudeGraphExtractor, NERExtractor
)
from threat_intel_kg.models import KnowledgeGraph
from threat_intel_kg.storage import Neo4jStore
from threat_intel_kg.main import process_url

# For visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# List of models to evaluate
MODELS = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4o",
    "gemini-1.5-pro-latest",   # Correct model name for Gemini 1.5 Pro
    "gemini-1.5-flash-latest", # Correct model name for Gemini 1.5 Flash
    "gemini-2.0-flash",        # Correct model name for Gemini 2.0 Flash
    "gemini-2.5-pro-preview-03-25",    # Gemini 2.5 Pro
    "gemini-2.5-flash-preview-04-17",  # Gemini 2.5 Flash
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-20240620",
    "ner"  # Baseline rule-based extractor
]

# List of test articles (try to use articles from different sources)
TEST_ARTICLES = [
    "https://cloud.google.com/blog/topics/threat-intelligence/sandworm-disrupts-power-ukraine-operational-technology/"
]

# Estimated cost per 1000 tokens for each model (in USD)
MODEL_COSTS = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.01, "output": 0.03},
    "gemini-1.5-pro-latest": {"input": 0.0035, "output": 0.0035},
    "gemini-1.5-flash-latest": {"input": 0.0015, "output": 0.0015},
    "gemini-2.0-flash": {"input": 0.0015, "output": 0.0015},
    "gemini-2.5-pro-preview-03-25": {"input": 0.0035, "output": 0.0035},
    "gemini-2.5-flash-preview-04-17": {"input": 0.0015, "output": 0.0015},
    "claude-3-5-haiku-latest": {"input": 0.00325, "output": 0.00975},
    "claude-3-5-sonnet-20240620": {"input": 0.015, "output": 0.075},
    "ner": {"input": 0, "output": 0}  # Free, rule-based
}

# Token count estimations (characters per token)
TOKEN_RATIO = {
    "openai": 4.0,  # OpenAI models
    "gemini": 4.0,  # Google models
    "claude": 4.0,  # Claude models
    "ner": 0        # Not applicable
}

def initialize_extractor(model_name: str, verbose: bool = False) -> Any:
    """Initialize the appropriate extractor based on model name."""
    if model_name == "ner":
        return NERExtractor(allowed_nodes=['*'], allowed_relationships=['*'], verbose=verbose)
    elif model_name.startswith("gpt"):
        return GraphExtractor(model=model_name, verbose=verbose)
    elif model_name.startswith("gemini"):
        return GeminiGraphExtractor(model=model_name, verbose=verbose)
    elif model_name.startswith("claude"):
        return ClaudeGraphExtractor(model=model_name, verbose=verbose)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def estimate_cost(model: str, input_chars: int, output_nodes: int, output_rels: int) -> float:
    """Estimate the cost of processing with a given model."""
    if model == "ner":
        return 0.0  # Rule-based extraction is free
    
    model_family = "openai" if model.startswith("gpt") else "gemini" if model.startswith("gemini") else "claude"
    
    # Estimate input tokens
    input_tokens = input_chars / TOKEN_RATIO[model_family]
    
    # Estimate output tokens - rough estimate based on typical node/relationship JSON structure
    avg_tokens_per_node = 50  # Average tokens per node in JSON
    avg_tokens_per_rel = 80   # Average tokens per relationship in JSON
    output_tokens = (output_nodes * avg_tokens_per_node) + (output_rels * avg_tokens_per_rel)
    
    # Calculate cost
    input_cost = (input_tokens / 1000) * MODEL_COSTS[model]["input"]
    output_cost = (output_tokens / 1000) * MODEL_COSTS[model]["output"]
    
    return input_cost + output_cost

def evaluate_model(model_name: str, articles: List[str], verbose: bool = False) -> Dict[str, Any]:
    """Evaluate a single model on multiple articles."""
    results = {
        "model": model_name,
        "articles": [],
        "total_processing_time": 0,
        "total_nodes": 0,
        "total_relationships": 0,
        "total_chars_processed": 0,
        "estimated_cost": 0.0,
        "successful_chunks": 0,
        "failed_chunks": 0
    }
    
    try:
        extractor = initialize_extractor(model_name, verbose=verbose)
        store = Neo4jStore()
        
        for url in articles:
            try:
                start_time = time.time()
                logger.info(f"Processing {url} with model {model_name}...")
                
                # Process the URL with robust error handling
                try:
                    article_stats = process_url(url, extractor, store)
                    processing_time = time.time() - start_time
                    successful = True
                except Exception as process_error:
                    # Log but don't re-raise the exception
                    logger.error(f"Error processing URL {url} with model {model_name}: {process_error}", exc_info=True)
                    processing_time = time.time() - start_time
                    article_stats = {
                        "total_nodes": 0,
                        "total_relationships": 0,
                        "estimated_tokens": 0,
                        "successful_chunks": 0,
                        "failed_chunks": 1,
                        "error": str(process_error)
                    }
                    successful = False
                
                # Extract article data
                article_result = {
                    "url": url,
                    "processing_time": processing_time,
                    "nodes": article_stats.get("total_nodes", 0),
                    "relationships": article_stats.get("total_relationships", 0),
                    "chars_processed": article_stats.get("estimated_tokens", 0) * 4,  # Convert tokens to chars
                    "successful_chunks": article_stats.get("successful_chunks", 0),
                    "failed_chunks": article_stats.get("failed_chunks", 0)
                }
                
                # Add error if we caught one during processing
                if "error" in article_stats:
                    article_result["error"] = article_stats["error"]
                
                # Calculate cost estimate
                cost = estimate_cost(
                    model_name, 
                    article_result["chars_processed"],
                    article_result["nodes"],
                    article_result["relationships"]
                )
                article_result["estimated_cost"] = cost
                
                # Add to overall stats
                results["articles"].append(article_result)
                results["total_processing_time"] += processing_time
                results["total_nodes"] += article_result["nodes"]
                results["total_relationships"] += article_result["relationships"]
                results["total_chars_processed"] += article_result["chars_processed"]
                results["estimated_cost"] += cost
                results["successful_chunks"] += article_result["successful_chunks"]
                results["failed_chunks"] += article_result["failed_chunks"]
                
                if successful:
                    logger.info(f"Model {model_name} processed {url} in {processing_time:.2f}s")
                    logger.info(f"  Nodes: {article_result['nodes']}, Relationships: {article_result['relationships']}")
                    logger.info(f"  Estimated cost: ${cost:.4f}")
                else:
                    logger.warning(f"Model {model_name} failed to process {url} after {processing_time:.2f}s")
                
            except Exception as article_error:
                # This should rarely happen since we're already catching errors in process_url
                logger.error(f"Unexpected error evaluating article {url} with model {model_name}: {article_error}", exc_info=True)
                
                # Add failed article to results
                results["articles"].append({
                    "url": url,
                    "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
                    "nodes": 0,
                    "relationships": 0,
                    "chars_processed": 0,
                    "successful_chunks": 0, 
                    "failed_chunks": 1,
                    "estimated_cost": 0.0,
                    "error": str(article_error)
                })
                results["failed_chunks"] += 1
            
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {e}", exc_info=True)
        results["error"] = str(e)
    
    return results

def run_evaluation(models: List[str], articles: List[str], output_file: str, parallel: bool = False, verbose: bool = False):
    """Run the evaluation on multiple models."""
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "models_evaluated": models,
        "articles_tested": articles,
        "results": []
    }
    
    # Track successful and failed models
    successful_models = []
    failed_models = []
    partially_successful_models = []
    
    if parallel:
        # Run evaluations in parallel
        with ThreadPoolExecutor(max_workers=min(len(models), 3)) as executor:
            futures = [executor.submit(evaluate_model, model, articles, verbose) for model in models]
            for future in futures:
                try:
                    result = future.result()
                    all_results["results"].append(result)
                    
                    # Track success/failure status
                    if "error" in result:
                        failed_models.append(result["model"])
                    elif any("error" in article for article in result.get("articles", [])):
                        partially_successful_models.append(result["model"])
                    else:
                        successful_models.append(result["model"])
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}", exc_info=True)
    else:
        # Run evaluations sequentially
        for model in models:
            logger.info(f"Evaluating model: {model}")
            result = evaluate_model(model, articles, verbose)
            all_results["results"].append(result)
            
            # Track success/failure status
            if "error" in result:
                failed_models.append(result["model"])
            elif any("error" in article for article in result.get("articles", [])):
                partially_successful_models.append(result["model"])
            else:
                successful_models.append(result["model"])
    
    # Add summary stats
    summary = {
        "fastest_model": "",
        "most_efficient_model": "",
        "most_detailed_model": "",
        "successful_models": successful_models,
        "partially_successful_models": partially_successful_models,
        "failed_models": failed_models
    }
    
    fastest_time = float('inf')
    lowest_cost_per_entity = float('inf')
    most_entities = 0
    
    # Only consider successful or partially successful models for rankings
    valid_results = [r for r in all_results["results"] if "error" not in r]
    
    for result in valid_results:
        # Skip completely failed models for the rankings
        if result["model"] in failed_models:
            continue
            
        # Speed comparison
        if result["total_processing_time"] < fastest_time:
            fastest_time = result["total_processing_time"]
            summary["fastest_model"] = result["model"]
        
        # Cost efficiency comparison
        total_entities = result["total_nodes"] + result["total_relationships"]
        if total_entities > 0:
            cost_per_entity = result["estimated_cost"] / total_entities
            if cost_per_entity < lowest_cost_per_entity:
                lowest_cost_per_entity = cost_per_entity
                summary["most_efficient_model"] = result["model"]
        
        # Detail comparison
        if total_entities > most_entities:
            most_entities = total_entities
            summary["most_detailed_model"] = result["model"]
    
    all_results["summary"] = summary
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    logger.info("=" * 40)
    logger.info("Evaluation Summary:")
    logger.info(f"- Fastest model: {summary['fastest_model']}")
    logger.info(f"- Most cost-efficient model: {summary['most_efficient_model']}")
    logger.info(f"- Most detailed model: {summary['most_detailed_model']}")
    logger.info(f"- Successful models: {', '.join(successful_models) if successful_models else 'None'}")
    
    if partially_successful_models:
        logger.info(f"- Partially successful models: {', '.join(partially_successful_models)}")
    if failed_models:
        logger.info(f"- Failed models: {', '.join(failed_models)}")
        
    logger.info(f"Detailed results saved to: {output_file}")
    
    return all_results

def create_interactive_visualizations(results: Dict[str, Any], output_dir: str):
    """Create interactive visualizations of evaluation results."""
    # Extract data from results
    models = []
    processing_times = []
    nodes = []
    relationships = []
    costs = []
    cost_per_entity = []
    
    for result in results["results"]:
        if "error" not in result:
            models.append(result["model"])
            processing_times.append(result["total_processing_time"])
            nodes.append(result["total_nodes"])
            relationships.append(result["total_relationships"])
            costs.append(result["estimated_cost"])
            
            total_entities = result["total_nodes"] + result["total_relationships"]
            if total_entities > 0:
                cost_per_entity.append(result["estimated_cost"] / total_entities)
            else:
                cost_per_entity.append(0)
    
    # Create data frames for easier plotting
    # Add error status information
    errors = []
    for result in results["results"]:
        if "error" in result:
            errors.append("Failed")
        else:
            has_article_errors = any("error" in article for article in result.get("articles", []))
            errors.append("Partial Success" if has_article_errors else "Success")
    
    df = pd.DataFrame({
        'Model': models,
        'Processing Time (s)': processing_times,
        'Nodes': nodes,
        'Relationships': relationships,
        'Total Entities': [n + r for n, r in zip(nodes, relationships)],
        'Cost ($)': costs,
        'Cost per Entity ($)': cost_per_entity,
        'Status': errors
    })
    
    # 1. Processing Time Chart
    fig_time = px.bar(
        df, 
        x='Model', 
        y='Processing Time (s)',
        title='Processing Time by Model',
        color='Status',
        color_discrete_map={'Success': '#32a852', 'Partial Success': '#e6b800', 'Failed': '#e63900'},
        text='Processing Time (s)',
        hover_data=['Nodes', 'Relationships', 'Status']
    )
    fig_time.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
    fig_time.update_layout(xaxis_tickangle=-45)
    fig_time.write_html(os.path.join(output_dir, 'processing_time.html'))
    
    # 2. Entity Extraction Chart
    fig_entity = px.bar(
        df, 
        x='Model', 
        y=['Nodes', 'Relationships'],
        title='Entity Extraction by Model',
        barmode='group',
        labels={'value': 'Count', 'variable': 'Entity Type'},
        color='Status',
        color_discrete_map={'Success': '#32a852', 'Partial Success': '#e6b800', 'Failed': '#e63900'},
        pattern_shape='variable',
        hover_data=['Processing Time (s)', 'Status']
    )
    fig_entity.update_layout(xaxis_tickangle=-45)
    fig_entity.write_html(os.path.join(output_dir, 'entity_extraction.html'))
    
    # 3. Cost Efficiency Chart
    fig_cost_eff = px.bar(
        df, 
        x='Model', 
        y='Cost per Entity ($)',
        title='Cost Efficiency by Model (Cost per Entity)',
        color='Cost per Entity ($)',
        color_continuous_scale='Greens_r',  # Reversed so darker = more efficient
        text='Cost per Entity ($)'
    )
    fig_cost_eff.update_traces(texttemplate='$%{text:.5f}', textposition='outside')
    fig_cost_eff.update_layout(xaxis_tickangle=-45)
    fig_cost_eff.write_html(os.path.join(output_dir, 'cost_efficiency.html'))
    
    # 4. Cost vs Entities Scatter Plot
    fig_scatter = px.scatter(
        df,
        x='Total Entities',
        y='Cost ($)',
        text='Model',
        title='Cost vs. Entities Extracted',
        size='Total Entities',
        color='Cost per Entity ($)',
        color_continuous_scale='Greens_r',
        hover_data=['Nodes', 'Relationships', 'Processing Time (s)']
    )
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.update_layout(xaxis_title='Total Entities (Nodes + Relationships)', yaxis_title='Estimated Cost ($)')
    fig_scatter.write_html(os.path.join(output_dir, 'cost_vs_entities.html'))
    
    # Create comprehensive dashboard with all charts
    create_interactive_report(df, results, output_dir, fig_table, fig_time, fig_entity, fig_cost_eff, fig_scatter)

def create_interactive_report(df, results, output_dir, fig_table, fig_time, fig_entity, fig_cost_eff, fig_scatter):
    """Create a comprehensive HTML report with interactive elements."""
    # Create a summary dataframe for the table
    summary_df = df.copy()
    summary_df['Processing Time (s)'] = summary_df['Processing Time (s)'].map(lambda x: f"{x:.2f}")
    summary_df['Cost ($)'] = summary_df['Cost ($)'].map(lambda x: f"${x:.4f}")
    summary_df['Cost per Entity ($)'] = summary_df['Cost per Entity ($)'].map(lambda x: f"${x:.5f}")
    
    # If fig_table wasn't provided, create it
    if fig_table is None:
        # Create the table figure
        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=list(summary_df.columns),
                fill_color='#f2f2f2',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[summary_df[col] for col in summary_df.columns],
                fill_color=[['#ffffff', '#f9f9f9'] * len(summary_df)],
                align='left',
                font=dict(size=11)
            )
        )])
        fig_table.update_layout(
            title="Model Evaluation Summary Table",
            margin=dict(l=0, r=0, t=30, b=0),
        )
    
    # Highlight the best performers
    best_time_model = results["summary"]["fastest_model"]
    best_efficiency_model = results["summary"]["most_efficient_model"]
    best_detail_model = results["summary"]["most_detailed_model"]
    
    # Create the charts as self-contained HTML files first
    fig_table.write_html(os.path.join(output_dir, 'summary_table.html'))
    fig_time.write_html(os.path.join(output_dir, 'processing_time.html'))
    fig_entity.write_html(os.path.join(output_dir, 'entity_extraction.html'))
    fig_cost_eff.write_html(os.path.join(output_dir, 'cost_efficiency.html'))
    fig_scatter.write_html(os.path.join(output_dir, 'cost_vs_entities.html'))
    
    # Read each chart's HTML to embed directly
    def read_html_file(file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                # Extract just the div with the plotly chart
                if '<div id="plotly-html-element"' in content:
                    start = content.find('<div id="plotly-html-element"')
                    end = content.find('</script>', content.find('<script type="text/javascript">')) + 9
                    if start > -1 and end > -1:
                        return content[start:end]
                return '<div>Chart could not be loaded</div>'
        except Exception as e:
            logger.error(f"Error reading chart file: {e}")
            return '<div>Chart could not be loaded</div>'
            
    # Read chart content
    summary_table_html = read_html_file(os.path.join(output_dir, 'summary_table.html'))
    processing_time_html = read_html_file(os.path.join(output_dir, 'processing_time.html'))
    entity_extraction_html = read_html_file(os.path.join(output_dir, 'entity_extraction.html'))
    cost_efficiency_html = read_html_file(os.path.join(output_dir, 'cost_efficiency.html'))
    cost_vs_entities_html = read_html_file(os.path.join(output_dir, 'cost_vs_entities.html'))
    
    # Build the full HTML report with embedded interactive charts
    with open(os.path.join(output_dir, 'evaluation_report.html'), 'w') as f:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Threat Intelligence Model Evaluation Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                }}
                .summary-box {{
                    background-color: #e9f7ef;
                    border-left: 5px solid #27ae60;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 0 5px 5px 0;
                }}
                .chart-container {{
                    margin-bottom: 30px;
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .timestamp {{
                    font-style: italic;
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
                .chart {{
                    height: 500px;
                    width: 100%;
                }}
                .best {{
                    font-weight: bold;
                    color: #27ae60;
                }}
                footer {{
                    text-align: center;
                    margin-top: 30px;
                    font-size: 12px;
                    color: #7f8c8d;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Threat Intelligence Model Evaluation Report</h1>
                    <p class="timestamp">Generated on {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</p>
                </div>
                
                <div class="summary-box">
                    <h2>Key Findings</h2>
                    <p><span class="best">Fastest model:</span> {best_time_model}</p>
                    <p><span class="best">Most cost-efficient model:</span> {best_efficiency_model}</p>
                    <p><span class="best">Most detailed model:</span> {best_detail_model}</p>
                    
                    <h3>Model Status</h3>
                    <p><span style="color: #32a852;">✓ Successful models:</span> {', '.join(results["summary"]["successful_models"]) if results["summary"]["successful_models"] else "None"}</p>
                    
                    {f'<p><span style="color: #e6b800;">⚠ Partially successful models:</span> {", ".join(results["summary"]["partially_successful_models"])}</p>' if results["summary"]["partially_successful_models"] else ''}
                    
                    {f'<p><span style="color: #e63900;">✗ Failed models:</span> {", ".join(results["summary"]["failed_models"])}</p>' if results["summary"]["failed_models"] else ''}
                </div>
                
                <div id="overview" class="chart-container">
                    <h2>Evaluation Summary Table</h2>
                    <div id="summary-table" class="chart">
                        {summary_table_html}
                    </div>
                </div>
                
                <div id="processing-time" class="chart-container">
                    <h2>Processing Time Comparison</h2>
                    <div id="time-chart" class="chart">
                        {processing_time_html}
                    </div>
                </div>
                
                <div id="entity-extraction" class="chart-container">
                    <h2>Entity Extraction Comparison</h2>
                    <div id="entity-chart" class="chart">
                        {entity_extraction_html}
                    </div>
                </div>
                
                <div id="cost-efficiency" class="chart-container">
                    <h2>Cost Efficiency Comparison</h2>
                    <div id="cost-chart" class="chart">
                        {cost_efficiency_html}
                    </div>
                </div>
                
                <div id="cost-vs-entities" class="chart-container">
                    <h2>Cost vs. Entities Extracted</h2>
                    <div id="scatter-chart" class="chart">
                        {cost_vs_entities_html}
                    </div>
                </div>
                
                <footer>
                    <p>This report was automatically generated by the Threat Intelligence Knowledge Graph evaluation script.</p>
                </footer>
            </div>
        </body>
        </html>
        """
        f.write(html_content)
    
    # We've already saved all charts earlier, so no need to save again

def create_output_directory():
    """Create a timestamped output directory for evaluation results."""
    # Create base evaluation directory if it doesn't exist
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(eval_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(results_dir, f"eval_{timestamp}")
    os.makedirs(output_dir)
    
    return output_dir

def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate different models for threat intelligence extraction")
    parser.add_argument("--models", nargs="+", choices=MODELS, help="Models to evaluate (defaults to all)")
    parser.add_argument("--articles", nargs="+", help="URLs of articles to use for evaluation")
    parser.add_argument("--output-dir", help="Directory for results (default: timestamped directory)")
    parser.add_argument("--parallel", action="store_true", help="Run evaluations in parallel")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    output_dir = args.output_dir if args.output_dir else create_output_directory()
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    # Create visualization directory inside the output directory
    charts_dir = os.path.join(output_dir, "charts")
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    
    models_to_evaluate = args.models or MODELS
    articles_to_test = args.articles or TEST_ARTICLES
    
    logger.info(f"Starting evaluation of {len(models_to_evaluate)} models on {len(articles_to_test)} articles")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Run the evaluation
    evaluation_results = run_evaluation(
        models=models_to_evaluate,
        articles=articles_to_test,
        output_file=results_file,
        parallel=args.parallel,
        verbose=args.verbose
    )
    
    # Generate visualizations automatically
    logger.info("Generating visualizations and interactive report...")
    create_interactive_visualizations(evaluation_results, charts_dir)
    
    logger.info(f"Interactive evaluation report created in {charts_dir}")
    logger.info(f"Full report available at: {os.path.join(charts_dir, 'evaluation_report.html')}")
    
    return output_dir
    
if __name__ == "__main__":
    main()