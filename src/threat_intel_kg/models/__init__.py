"""
Data models and prompts for the Threat Intelligence Knowledge Graph package.
"""

from .data_models import Property, Node, Relationship, KnowledgeGraph
from .prompts import get_prompt_template, create_prompt_template

__all__ = [
    "Property", "Node", "Relationship", "KnowledgeGraph",
    "get_prompt_template", "create_prompt_template"
]