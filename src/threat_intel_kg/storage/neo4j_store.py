"""
Neo4j storage module for persisting knowledge graphs.
"""

import logging
from typing import List, Optional, Dict, Any
import uuid
from neo4j import GraphDatabase

from ..models import Node, Relationship, KnowledgeGraph
from ..utils import props_to_dict
from ..config import NEO4J_CONFIG

logger = logging.getLogger(__name__)


class Neo4jStore:
    """
    Store knowledge graphs in Neo4j.
    """
    
    def __init__(
        self, 
        url: Optional[str] = None, 
        username: Optional[str] = None, 
        password: Optional[str] = None
    ):
        """
        Initialize the Neo4jStore.
        
        Args:
            url: Neo4j URL.
            username: Neo4j username.
            password: Neo4j password.
        """
        self.url = url or NEO4J_CONFIG["url"]
        self.username = username or NEO4J_CONFIG["username"]
        self.password = password or NEO4J_CONFIG["password"]
        self.driver = None
        
        # Connect to Neo4j
        self.connect()
    
    def connect(self) -> None:
        """
        Connect to the Neo4j database.
        
        Raises:
            Exception: If connection fails.
        """
        try:
            self.driver = GraphDatabase.driver(
                self.url,
                auth=(self.username, self.password)
            )
            # Verify connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                if test_value != 1:
                    raise Exception("Failed to validate Neo4j connection")
            logger.info(f"Connected to Neo4j at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}. URL: {self.url}")
            raise
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def store_knowledge_graph(
        self, 
        knowledge_graph: KnowledgeGraph, 
        source_document: Optional[Any] = None
    ) -> bool:
        """
        Store a knowledge graph in Neo4j.
        
        Args:
            knowledge_graph: The knowledge graph to store.
            source_document: Optional source document to associate with the graph.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.driver:
            logger.error("No connection to Neo4j")
            return False
        
        try:
            # Extract source metadata if available
            source_url = None
            if source_document and hasattr(source_document, 'metadata'):
                source_url = source_document.metadata.get('source', None)
            elif isinstance(source_document, dict) and 'metadata' in source_document:
                source_url = source_document['metadata'].get('source', None)
            
            # Generate a unique batch ID for this import
            batch_id = str(uuid.uuid4())
            
            with self.driver.session() as session:
                # Create/merge all nodes first
                for node in knowledge_graph.nodes:
                    properties = props_to_dict(node.properties) if node.properties else {}
                    # Add name property for better Cypher statement generation
                    properties["name"] = node.id.title()
                    
                    # Add source and batch information
                    if source_url:
                        properties["sourceUrl"] = source_url
                    properties["batchId"] = batch_id
                    
                    # Create node with Cypher
                    session.run(
                        f"""
                        MERGE (n:{node.type.capitalize()} {{id: $id}})
                        SET n += $properties
                        """,
                        {
                            "id": node.id.title(),
                            "properties": properties
                        }
                    )
                    
                # Create relationships
                for rel in knowledge_graph.rels:
                    properties = props_to_dict(rel.properties) if rel.properties else {}
                    
                    # Add source and batch information
                    if source_url:
                        properties["sourceUrl"] = source_url
                    properties["batchId"] = batch_id
                    
                    # Create relationship with Cypher
                    session.run(
                        f"""
                        MATCH (source:{rel.source.type.capitalize()} {{id: $source_id}})
                        MATCH (target:{rel.target.type.capitalize()} {{id: $target_id}})
                        MERGE (source)-[r:{rel.type}]->(target)
                        SET r += $properties
                        """,
                        {
                            "source_id": rel.source.id.title(),
                            "target_id": rel.target.id.title(),
                            "properties": properties
                        }
                    )
            
            logger.info(
                f"Successfully added {len(knowledge_graph.nodes)} nodes and "
                f"{len(knowledge_graph.rels)} relationships to Neo4j with batch ID {batch_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Error while adding knowledge graph to Neo4j: {e}")
            return False
    
    def query(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Run a Cypher query against the Neo4j database.
        
        Args:
            cypher_query: The Cypher query to run.
            params: Optional parameters for the query.
            
        Returns:
            Results of the query.
            
        Raises:
            Exception: If the query fails.
        """
        if not self.driver:
            logger.error("No connection to Neo4j")
            raise ConnectionError("No connection to Neo4j")
            
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            raise