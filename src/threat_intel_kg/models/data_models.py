"""
Defines the data structures for nodes, relationships, and the overall knowledge graph.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


class Property(BaseModel):
    """A single property consisting of key and value."""
    key: str = Field(..., description="Key of the property")
    value: str = Field(..., description="Value of the property")


class Node(BaseModel):
    """A node in the knowledge graph."""
    id: str = Field(..., description="Unique identifier for the node")
    type: str = Field(..., description="Type/label of the node")
    properties: Optional[List[Property]] = Field(None, description="List of node properties")
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate that the node ID is not empty."""
        if not v.strip():
            raise ValueError("Node ID cannot be empty")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that the node type is not empty."""
        if not v.strip():
            raise ValueError("Node type cannot be empty")
        return v


class Relationship(BaseModel):
    """A relationship in the knowledge graph."""
    source: Node = Field(..., description="Source node of the relationship")
    target: Node = Field(..., description="Target node of the relationship")
    type: str = Field(..., description="Type of the relationship")
    properties: Optional[List[Property]] = Field(None, description="List of relationship properties")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that the relationship type is not empty."""
        if not v.strip():
            raise ValueError("Relationship type cannot be empty")
        return v


class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(..., description="List of relationships in the knowledge graph")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the knowledge graph to a dictionary."""
        return {
            "nodes": [node.model_dump() for node in self.nodes],
            "relationships": [rel.model_dump() for rel in self.rels]
        }