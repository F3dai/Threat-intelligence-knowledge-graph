# Threat intelligence knowledge graph

This project automates the creation of a knowledge graph from security reports. The extracted entities and relationships are structured and stored as nodes and edges in a Neo4j database, allowing for a detailed, queryable graph representation of intel.

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

Update `src/threat intel knowledge graph/config.py` with necessary info.

Add a `.env` file, or set your environment variables.

## Currently implementing

This is an **alpha stage** project with incomplete features.

- New techniques such as fine-tuning and refining prompts.
- Widening scope to any article related to security, such as geopolitical reports (terrorism, crime, politics).
- Web interface for more useful and accessible views of intel.

Once the project is in beta, the application will continuously run and analyse open source intelligence.

## Disclaimer

This is a very early stage project and should not be used in production. The script can extract information from unknown sources and executes commands on a database. It's probably vulnerable.