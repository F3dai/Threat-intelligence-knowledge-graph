"""
Defines the data structures (using Pydantic) for nodes, relationships, and the overall knowledge graph.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class Property(BaseModel):
    """A single property consisting of key and value"""
    key: str = Field(..., description="Key of the property")
    value: str = Field(..., description="Value of the property")

class Node(BaseModel):
    id: str
    type: str
    properties: Optional[List[Property]] = Field(None, description="List of node properties")

class Relationship(BaseModel):
    source: Node
    target: Node
    type: str
    properties: Optional[List[Property]] = Field(None, description="List of relationship properties")

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(..., description="List of relationships in the knowledge graph")
