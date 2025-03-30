"""
Extraction modules for parsing security reports into structured data.
"""

from .openai_extractor import GraphExtractor
from .gemini_extractor import GeminiGraphExtractor

__all__ = ["GraphExtractor", "GeminiGraphExtractor"]