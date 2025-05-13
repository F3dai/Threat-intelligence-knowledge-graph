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
    "url": os.getenv("NEO4J_URL", "bolt://localhost:7687"),
    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "password"),
}

# OpenAI configuration
OPENAI_CONFIG: Dict[str, Any] = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-16k"),
    "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0")),
    # Dictionary of available models and their contexts
    "models": {
        "gpt-3.5-turbo": 16000,
        "gpt-3.5-turbo-16k": 16000,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000
    }
}

# Google Gemini configuration
GEMINI_CONFIG: Dict[str, Any] = {
    "api_key": os.getenv("GOOGLE_API_KEY", ""),
    "model": os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25"),
    "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0")),
    # Dictionary of available models and their contexts
    "models": {
        "gemini-2.5-pro": 1000000, 
        "gemini-2.5-pro-exp-03-25": 1000000,
        "gemini-2.0-flash": 128000,
        "gemini-2.5-flash-preview-04-17": 128000  # New Gemini 2.5 Flash model
    }
}

# Claude configuration
CLAUDE_CONFIG: Dict[str, Any] = {
    "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "model": os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-latest"),
    "temperature": float(os.getenv("CLAUDE_TEMPERATURE", "0")),
    # Dictionary of available models and their contexts
    "models": {
        "claude-3-5-haiku-latest": 200000,
        "claude-3-5-sonnet-20240620": 400000  # New Claude 3.5 Sonnet model with larger context
    }
}

# STIX 2.1 Aligned Default Node Labels
DEFAULT_ALLOWED_NODES = [
    "threat-actor",
    "intrusion-set",
    "campaign",
    "identity",
    "malware",
    "tool",
    "attack-pattern",
    "course-of-action",
    "vulnerability",
    "indicator",
    "observed-data",
    "location",
    "infrastructure"
]

# STIX 2.1 Aligned Default Relationship Types
DEFAULT_ALLOWED_RELATIONSHIPS = [
    "uses",
    "targets",
    "attributed-to",
    "mitigates",
    "indicates",
    "located-at",
    "compromises",
    "delivers",
    "related-to"
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
    
if not CLAUDE_CONFIG["api_key"] or CLAUDE_CONFIG["api_key"] == "your_anthropic_api_key":
    logger.warning(
        "Anthropic API key not properly set. Please set the ANTHROPIC_API_KEY environment variable if using Claude models."
    )