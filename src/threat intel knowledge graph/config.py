"""
Configuration file for knowledge graph application
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# neo4j configuration
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")

# Security report URL
REPORT_URL = "https://cloud.google.com/blog/topics/threat-intelligence/fortimanager-zero-day-exploitation-cve-2024-47575"