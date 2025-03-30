# Threat Intelligence Knowledge Graph

This project automates the creation of a knowledge graph from security reports using Large Language Models (LLMs). The extracted entities and relationships are structured based on defined schemas and stored as nodes and edges in a Neo4j database, allowing for a detailed, queryable graph representation of threat intelligence.

## Overview

The Threat Intelligence Knowledge Graph is a Python-based tool that extracts contextual structured intelligence from unstructured cyber security reports (fetched from URLs) and generates a knowledge graph using Neo4j. It supports multiple LLM providers (currently OpenAI and Google Gemini) and aims to structure data loosely based on concepts like the MITRE ATT&CK taxonomy (though specific mapping depends on model output and configuration). This tool automates the process of identifying and structuring entities and their relationships within security contexts.

## Prerequisites

- Python 3.8+
- An active Neo4j instance (local or cloud-based, AuraDB free tier works well)
- API Key for your chosen LLM provider:
    - OpenAI API Key (if using `--model openai`)
    - Google API Key enabled for the Gemini API (if using `--model gemini`)
- Key Python libraries (install via `requirements.txt`): `openai`, `google-generativeai`, `neo4j`, `requests`, `beautifulsoup4`, `python-dotenv`, `tqdm`, `pydantic`

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/F3dai/Threat-intelligence-knowledge-graph.git
cd Threat-intelligence-knowledge-graph
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

1.  Create a `.env` file in the root directory.
2.  Add the following variables, replacing the placeholder values:

    ```dotenv
    # Neo4j Connection Details
    NEO4J_URI=bolt://localhost:7687 # Or your Neo4j Aura URI (neo4j+s://...)
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your-neo4j-password

    # API Keys (Only provide the key for the service you intend to use)
    OPENAI_API_KEY=sk-your-openai-api-key
    GOOGLE_API_KEY=your-google-api-key
    ```

3.  (Optional) Adjust default model names, chunking parameters, allowed nodes/relationships in `src/threat_intel_kg/config.py`.

## Usage

The tool is run from the command line using `src/cli.py`.

```bash
# Process a single URL using the default model (OpenAI)
python src/cli.py process https://example.com/security-report

# Process multiple URLs
python src/cli.py process https://example.com/report1 https://example.com/report2

# Use Google's Gemini model instead of OpenAI
python src/cli.py process --model gemini https://example.com/security-report

# Enable verbose logging for detailed output
python src/cli.py process --verbose https://example.com/security-report
```

## Important Notes on Model Usage

### OpenAI (Default)
- Uses standard chunking based on parameters in `config.py`.
- Generally faster due to higher default rate limits.
- Cost depends on the specific OpenAI model used and token consumption.

### Google Gemini (`--model gemini`)
- **New 2.5 Pro Model:** The application now supports the new Gemini 2.5 Pro model which offers a significantly higher token input size.
- **Reduced Chunking:** Due to the larger context window, many documents can now be processed in a single chunk, improving efficiency.
- **Rate Limits:** The script automatically handles rate limits by enforcing appropriate delays between API calls.
- **API Key:** Ensure your Google API key is enabled for the Gemini API in your Google Cloud project or AI Studio settings.

## Project Structure

```
.
├── src/
│   ├── threat_intel_kg/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration (models, chunking, API keys fallback)
│   │   ├── main.py            # Main processing workflow, chunking, rate limiting
│   │   ├── models/            # Data models & Prompts
│   │   │   ├── __init__.py
│   │   │   ├── data_models.py # Pydantic models for KnowledgeGraph
│   │   │   └── prompts.py     # LLM prompt templates
│   │   ├── extractors/        # LLM interaction logic
│   │   │   ├── __init__.py
│   │   │   ├── openai_extractor.py  # OpenAI-based extractor
│   │   │   └── gemini_extractor.py  # Google Gemini-based extractor
│   │   ├── storage/           # Database interaction
│   │   │   ├── __init__.py
│   │   │   └── neo4j_store.py # Neo4j storage logic
│   │   └── utils/             # Helper utilities
│   │       ├── __init__.py
│   │       └── helpers.py     # Utility functions
│   └── cli.py                 # Command-line interface definition
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Currently Implementing / Future Work

This is an **alpha stage** project with ongoing development. Potential future features include:

- Fine-tuning models or refining prompts for better accuracy and consistency.
- Expanding scope beyond specific security reports (e.g., geopolitical context, vulnerability databases) which requires a new taxonomy/schema.
- Adding more extractors (e.g., local LLMs, other cloud providers).
- Developing a web interface for easier interaction and graph visualization.
- Implementing continuous monitoring and analysis of OSINT feeds.
- More robust error handling and retries.

## Disclaimer

This project is in an early stage (alpha) and should **not** be used in production environments without thorough review and testing. It interacts with external APIs (potentially incurring costs), processes data from untrusted external URLs, and modifies a database. Use with caution and at your own risk.