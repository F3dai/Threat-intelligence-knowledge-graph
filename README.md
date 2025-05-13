# Threat Intelligence Knowledge Graph

This project automates the creation of a knowledge graph from security reports using Large Language Models (LLMs). The extracted entities and relationships are structured based on defined schemas and stored as nodes and edges in a Neo4j database, allowing for a detailed, queryable graph representation of threat intelligence.

## Overview

The Threat Intelligence Knowledge Graph is a Python-based tool that extracts contextual structured intelligence from unstructured cyber security reports (fetched from URLs) and generates a knowledge graph using Neo4j. It supports multiple LLM providers (OpenAI, Google Gemini, Anthropic Claude) and aims to structure data based on concepts like the MITRE ATT&CK taxonomy and STIX 2.1 objects. This tool automates the process of identifying and structuring entities and their relationships within security contexts.

Example graph:

![Sandworm](/images/graph.svg)

Generated from the following article: https://cloud.google.com/blog/topics/threat-intelligence/sandworm-disrupts-power-ukraine-operational-technology

## Project architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│                     │    │   Model Extractors  │    │                     │
│  Threat Intel       │    │ ┌─────────────────┐ │    │  Knowledge Graph    │
│  Reports            │───>│ │OpenAI           │ │───>│  Storage            │
│                     │    │ │Claude           │ │    │                     │
│  - URLs             │    │ │Gemini           │ │    │  - Neo4j Database   │
│  - Local Files      │    │ │NER/STIXnet      │ │    │  - Query Interface  │
└─────────────────────┘    │ └─────────────────┘ │    └─────────────────────┘
                           └─────────────────────┘
```

The system consists of three main components:
1. **Input processing** - Handles document loading and text chunking from various sources
2. **Entity extraction** - Uses LLMs or pattern-based extractors to identify entities and relationships
3. **Storage and visualisation** - Persists the knowledge graph in Neo4j for visualization and analysis

Note, [STIXnet](https://github.com/Mhackiori/STIXnet) is a modified submodule.

## Prerequisites

- Python 3.8+
- An active Neo4j instance (local or cloud-based, AuraDB free tier works well)
- API Key for your chosen LLM provider
- Python libraries via `requirements.txt`

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/F3dai/Threat-intelligence-knowledge-graph.git
cd Threat-intelligence-knowledge-graph
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

# If using Named Entity Recognition, download NLTK resources
python STIXnet/download_nltk.py
```

## Configuration

1.  Configure a `.env` file in the root directory. Template provided.
2.  (Optional) Adjust default model names, chunking parameters, allowed nodes/relationships in `src/threat_intel_kg/config.py`.

## Usage

The tool is run from the command line using `src/cli.py`.

```bash
usage: cli.py [-h] {process,version} ...

Extract threat intelligence from security reports and build a knowledge graph.

positional arguments:
  {process,version}  Command to run
    process          Process URLs to extract threat intelligence
    version          Show version information

options:
  -h, --help         show this help message and exit
```

When using `process`:
```
usage: cli.py process [-h] [--verbose]
                      [--model {gpt-3.5-turbo,gpt-4-turbo,gpt-4o,gemini-2.5-pro,gemini-2.0-flash,gemini-2.5-flash-preview-04-17,claude-3-5-haiku,claude-3-5-sonnet-20240620,ner}]
                      urls [urls ...]
cli.py process: error: the following arguments are required: urls
```

Example usage:
```
python src/cli.py process --model gemini-2.5-flash-preview-04-17 https://cloud.google.com/blog/topics/threat-intelligence/sandworm-disrupts-power-ukraine-operational-technology
```

## Supported models

The system supports a variety of models for entity extraction:

| Provider | Models | Context Window | Features |
|----------|--------|---------------|----------|
| **OpenAI** | gpt-3.5-turbo<br>gpt-4-turbo<br>gpt-4o | 16K<br>128K<br>128K | High accuracy<br>Good relationship extraction<br>Fast response times |
| **Google** | gemini-2.5-pro<br>gemini-2.0-flash<br>gemini-2.5-flash | 1M<br>128K<br>128K | Very large context window<br>Handles long reports<br>Rate-limited (5 RPM, 25 RPD) |
| **Anthropic** | claude-3-5-haiku<br>claude-3-5-sonnet | 200K<br>400K | Strong at nuanced extraction<br>Good with technical details |
| **STIXnet** | ner | Unlimited | No API key needed<br>Specialized IoC detection<br>Limited to predefined patterns |

## Evaluation results

We've conducted comprehensive evaluations of different models to measure their performance in threat intelligence extraction:

![Model Evaluation Summary](/images/summary.png)

This summary table shows that:
- GPT-3.5-Turbo extracts the most entities but at a relatively high cost
- Gemini models offer good cost efficiency with varying levels of entity extraction
- NER extractor is free but extracts fewer relationships and is not able to enrich threat intelligence with existing knowledge.
- Claude models occupy a middle ground in terms of cost and extraction performance

![Cost vs. Entities Extracted](/images/cost.png)

**Note**: The number of entities extracted does not necessarily indicate better quality. We are working on enhancing our evaluation framework to include qualitative measures such as relevance, accuracy, and completeness of the extracted knowledge graph. The current metrics focus on quantitative aspects only.

## Model evaluation framework

The project includes a comprehensive evaluation framework for comparing different models:

```bash
# Run model evaluation with default settings
python evaluation/model_evaluation.py

# Evaluate specific models
python evaluation/model_evaluation.py --models gpt-4o claude-3-5-sonnet-20240620

# Run evaluations in parallel. Caution for hitting rate limits with this option.
python evaluation/model_evaluation.py --parallel
```

The evaluation generates an interactive HTML report with:
- Processing time comparison
- Entity extraction metrics
- Cost efficiency analysis
- Detailed performance statistics

Results are stored in timestamped directories under `evaluation/results/` with interactive visualisations in the `charts/` subdirectory.

## Recent improvements

- Enforced STIX taxonomy
- Imeplented additional models, including STIXnet, a local Named Entity Recognition model
- Created evaluation framework to compare models
- Added MIT Licesne

## Currently working on:

- Evaluation framework:
  - Improving metrics
  - High quality anotation for benchmarking
- Prompt engineering now that we have an evaluation framework
- Implementing additional NER models
- Possibly a web interface

## Disclaimer

This project is in an early stage (alpha) - you should review and test if using in any production environment. It interacts with external APIs (potentially incurring costs), processes data from untrusted external URLs, and modifies a database. Use with caution and at your own risk!