"""
Configuration settings for the Threat Intelligence Knowledge Graph application.
"""

import os
import logging
from typing import Dict, Any, Union
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Neo4j configuration
NEO4J_CONFIG: Dict[str, str] = {
    "url": os.getenv("NEO4J_URL", "bolt://127.0.0.1:7687"),
    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "password"),
}

# OpenAI configuration
OPENAI_CONFIG: Dict[str, str] = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-16k"),
    "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0")),
}

# Google Gemini configuration
GEMINI_CONFIG: Dict[str, Any] = {
    "api_key": os.getenv("GOOGLE_API_KEY", ""),
    "model": os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25"),
    "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0")),
}

# Default values for extracting knowledge graph
DEFAULT_ALLOWED_NODES = [
    "Threat", "Vulnerability", "Asset", "Actor", 
    "Attack", "Mitigation", "Indicator"
]

DEFAULT_ALLOWED_RELATIONSHIPS = [
    "TARGETS", "EXPLOITS", "PROTECTS", "MITIGATES", 
    "ASSOCIATES_WITH", "IDENTIFIES"
]

# Text processing configuration
TEXT_PROCESSING_CONFIG: Dict[str, Any] = {
    "chunk_size": int(os.getenv("CHUNK_SIZE", "2048")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "24")),
}

# Validate critical configuration
if not OPENAI_CONFIG["api_key"] or OPENAI_CONFIG["api_key"] == "your_openai_api_key":
    logger.warning(
        "OpenAI API key not properly set. Please set the OPENAI_API_KEY environment variable."
    )
    
if not GEMINI_CONFIG["api_key"] or GEMINI_CONFIG["api_key"] == "your_google_api_key":
    logger.warning(
        "Google API key not properly set. Please set the GOOGLE_API_KEY environment variable if using Gemini models."
    )