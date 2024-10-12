# knowledge_graph_constructor.py

"""
Constructing Knowledge Graphs from Text using OpenAI Functions and Neo4j

This script extracts information from cyber security reports and constructs a knowledge graph
using Neo4j. It leverages LangChain, OpenAI's GPT-3.5-turbo-16k model, and Neo4jGraph for graph operations.
"""

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from pydantic import Field, BaseModel
from langchain.chains.openai_functions import create_structured_output_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import TokenTextSplitter
from tqdm import tqdm
from langchain_community.document_loaders import WebBaseLoader
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# neo4j configuration
url = "bolt://127.0.0.1:7687"
username ="neo4j"
password = "password"

# Initialise neo4j server
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

# Security report URL
security_report_url = "https://cloud.google.com/blog/topics/threat-intelligence/sandworm-disrupts-power-ukraine-operational-technology/"

# Define data models
class Property(BaseModel):
  """A single property consisting of key and value"""
  key: str = Field(..., description="key")
  value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )

# Utility functions
def format_property_key(s: str) -> str:
    """
    Convert a string to camelCase format.
    
    Args:
        s (str): The string to format.
    
    Returns:
        str: The formatted string.
    """

    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)

def props_to_dict(props) -> dict:
    """
    Convert a list of Property objects to a dictionary.
    
    Args:
        props (Optional[List[Property]]): List of Property objects.
    
    Returns:
        dict: Dictionary of properties.
    """

    properties = {}
    if not props:
      return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties

def map_to_base_node(node: Node) -> BaseNode:
    """
    Map a KnowledgeGraph Node to a BaseNode.
    
    Args:
        node (Node): The node to map.
    
    Returns:
        BaseNode: The mapped BaseNode.
    """

    properties = props_to_dict(node.properties) if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )

def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """
    Map a KnowledgeGraph Relationship to a BaseRelationship.
    
    Args:
        rel (Relationship): The relationship to map.
    
    Returns:
        BaseRelationship: The mapped BaseRelationship.
    """

    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return BaseRelationship(
        source=source, target=target, type=rel.type, properties=properties
    )


# Initialise OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None
) -> Any:
    """
    Create a structured output chain for extracting knowledge graph data.
    
    Args:
        allowed_nodes (Optional[List[str]]): List of allowed node labels.
        allowed_rels (Optional[List[str]]): List of allowed relationship types.
    
    Returns:
        Any: The extraction chain.
    """

    # Prepare allowed nodes and relationships as a formatted string
    allowed_nodes_str = "- **Allowed Node Labels:** " + ", ".join(allowed_nodes) if allowed_nodes else ""
    allowed_rels_str = "- **Allowed Relationship Types:** " + ", ".join(allowed_rels) if allowed_rels else ""

    # Construct the full prompt text
    prompt_text = f"""# Neo4j Knowledge Graph Instructions for GPT-4
## 1. Overview
You are an advanced algorithm specialised in extracting and structuring information to build a Neo4j graph from cyber security reports.
- **Nodes** represent cyber security entities and concepts relevant to the report.
- The goal is to create a clear and comprehensive Neo4j graph that accurately reflects the relationships and entities within the cyber security context.
## 2. Labeling Nodes
- **Consistency**: Use standardised labels for node types to maintain uniformity across the graph.
  - Examples of node labels include **"Threat"**, **"Vulnerability"**, **"Asset"**, **"Actor"**, **"Attack"**, **"Mitigation"**, and **"Indicator"**.
- **Node IDs**: Utilise descriptive and human-readable identifiers for node IDs based on the entities' names or unique attributes found in the text. Do not use integers or autogenerated IDs.
{allowed_nodes_str}
{allowed_rels_str}
## 3. Handling Numerical Data and Dates
- **Attributes**: Incorporate numerical data and dates as properties of the relevant nodes.
  - For example, attach the property `severityScore` to a **"Vulnerability"** node or `discoveryDate` to a **"Threat"** node.
- **No Separate Nodes**: Do not create separate nodes for numerical values or dates. Always attach them as properties of existing nodes.
- **Property Format**: Use key-value pairs for all properties.
- **Quotation Marks**: Avoid using escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, such as `impactLevel` or `firstSeen`.
## 4. Coreference Resolution
- **Entity Consistency**: Ensure that each entity is represented by a single, consistent identifier throughout the graph.
  - For instance, if "Advanced Persistent Threat 29" is referred to as "APT29" or "the group" in different parts of the report, always use "Advanced Persistent Threat 29" as the node ID.
- **Clarity**: Maintain clarity by using the most complete and descriptive identifier for each entity to avoid ambiguity.
## 5. Neo4j Compatibility
- **Format**: Structure the output in a JSON format compatible with Neo4j's import tools, including separate sections for nodes and relationships.
- **Identifiers**: Ensure that each node has a unique `id` and that relationships correctly reference these IDs in the `startNode` and `endNode` fields.
## 6. Strict Compliance
Adhere strictly to these guidelines. Any deviation may lead to errors in the graph structure or import process.
"""

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "Use the given format to extract information from the following cyber security threat report: {input}"),
        ("human", "Tip: Ensure that the output adheres to the Neo4j-compatible JSON format specified in the instructions."),
    ])

    # Create and return the extraction chain
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=True)


def extract_and_store_graph(
    document: Document,
    nodes: Optional[List[str]] = None,
    rels: Optional[List[str]] = None
) -> None:
    """
    Extract knowledge graph data from a document and store it in Neo4j.
    
    Args:
        document (Document): The document to process.
        nodes (Optional[List[str]]): Allowed node labels.
        rels (Optional[List[str]]): Allowed relationship types.
    """

    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(nodes, rels)

    # # Debug
    print("##################### Expected input keys:", extract_chain.input_keys) # debug
    # # Invoke chain with complete input data
    data = extract_chain.invoke(document.page_content)['function']
    
    # Construct a graph document
    graph_document = GraphDocument(
        nodes=[map_to_base_node(node) for node in data.nodes],
        relationships=[map_to_base_relationship(rel) for rel in data.rels],
        source=document
    )
    # Store information into a graph
    graph.add_graph_documents([graph_document])


def main():
    """
    Main function to execute the knowledge graph construction.
    """

    loader = WebBaseLoader("https://cloud.google.com/blog/topics/threat-intelligence/apt44-unearthing-sandworm")

    docs = loader.load()

    # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)

    # Only take the first the raw_documents
    documents = text_splitter.split_documents(docs[:3])

    allowed_nodes = ["Threat", "Vulnerability", "Asset", "Actor", "Attack", "Mitigation", "Indicator"]
    allowed_rels = ["TARGETS", "EXPLOITS", "PROTECTS", "MITIGATES", "ASSOCIATES_WITH", "IDENTIFIES"]

    for _, document in tqdm(enumerate(documents), total=len(documents)):
        extract_and_store_graph(document=document, nodes=allowed_nodes, rels=allowed_rels)

if __name__ == "__main__":
    main()
