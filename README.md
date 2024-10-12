# Threat intelligence knowledge graph

Constructing Knowledge Graphs from cyber security reports using OpenAI and neo4j

## Overview

The Threat intelligence knowledge graph is a Python-based tool that extracts contextual structured intelligence from unstructured cyber security reports and generates a knowledge graph using neo4j, based on MITRE taxonomy. This tool automates the process of identifying and structuring entities and their relationships within security contexts.

## Prerequisites

- Python
- Existing neo4j instance (local or cloud-based)
- OpenAI API Key

## Usage

```
python -m venv venv
./venv/scripts/activate
pip install -r .\requirements.txt
```

Replace `security_report_url` with a URL and `# neo4j configuration` with your neo4j database information.

Add your openai API to the environment variables, or a .env file.

## Currently implementing

- More precise prompting / threat intel extraction techniques
- Web interface for more useful and accessible views of intel
- Opening the application to more case studies, such as geopolitical threat analysis.

## Disclaimer

This is a very early stage project and should not be used in production. The script can extract information from unknown sources and executes commands on a database. It's probably vulnerable.