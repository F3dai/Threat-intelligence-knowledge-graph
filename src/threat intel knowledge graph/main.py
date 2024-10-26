# src/threat_knowledge_graph/main.py

"""
Main script to construct a Knowledge Graph from threat reports using OpenAI and Neo4j.

This script performs the following steps:
1. Loads configuration settings.
2. Connects to the Neo4j database.
3. Loads and splits the cybersecurity report.
4. Extracts entities and relationships using OpenAI's GPT model.
5. Maps the extracted data to defined data models.
6. Stores the structured data into Neo4j.
"""

import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.openai_functions import create_structured_output_chain
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# Import custom modules
from config import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY, REPORT_URL
from utils import format_property_key, props_to_dict
from models.data_models import Node, Relationship, KnowledgeGraph
from models.prompts import get_prompt_template
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node as BaseNode, Relationship as BaseRelationship
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

def map_to_base_node(node: Node) -> BaseNode:
    """
    Map a custom Node to a Neo4j-compatible BaseNode.

    Args:
        node (Node): The custom Node object.

    Returns:
        BaseNode: The Neo4j-compatible Node object.
    """
    properties = props_to_dict(node.properties) if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return BaseNode(
        id=node.id.title(),
        type=node.type.capitalize(),
        properties=properties
    )

def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """
    Map a custom Relationship to a Neo4j-compatible BaseRelationship.

    Args:
        rel (Relationship): The custom Relationship object.

    Returns:
        BaseRelationship: The Neo4j-compatible Relationship object.
    """
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return BaseRelationship(
        source=source,
        target=target,
        type=rel.type,
        properties=properties
    )

def get_extraction_chain(allowed_nodes: list, allowed_rels: list, llm: ChatOpenAI) -> any:
    """
    Create a structured output chain for extracting knowledge graph data.

    Args:
        allowed_nodes (list): List of allowed node labels.
        allowed_rels (list): List of allowed relationship types.
        llm (ChatOpenAI): Initialised OpenAI language model.

    Returns:
        Any: The extraction chain.
    """
    prompt_text = get_prompt_template(allowed_nodes, allowed_rels)
    
    # Create the prompt template
    prompt = create_prompt_template(prompt_text)
    
    # Create and return the extraction chain
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=True)

def create_prompt_template(prompt_text: str):
    """
    Create a prompt template for the extraction chain.

    Args:
        prompt_text (str): The formatted prompt text.

    Returns:
        ChatPromptTemplate: The created prompt template.
    """
    from langchain.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "Use the given format to extract information from the following cyber security threat report: {input}"),
        ("human", "Tip: Ensure that the output adheres to the Neo4j-compatible JSON format specified in the instructions."),
    ])

def extract_and_store_graph(document: Document, extract_chain: any, graph: Neo4jGraph) -> None:
    """
    Extract knowledge graph data from a document and store it in Neo4j.

    Args:
        document (Document): The document to process.
        extract_chain (any): The extraction chain.
        graph (Neo4jGraph): The Neo4j graph instance.
    """

    # Invoke chain with complete input data
    try:
        data = extract_chain.invoke(document.page_content)['function']
    except Exception as e:
        print(f"Error during extraction: {e}")
        return
    
    # Map extracted data to data models
    try:
        if isinstance(data, KnowledgeGraph):
            knowledge_graph = data
        elif isinstance(data, dict):
            # Attempt to create KnowledgeGraph from dict
            knowledge_graph = KnowledgeGraph(nodes=data.get('nodes', []), rels=data.get('rels', []))
        else:
            print(f"Unexpected data type: {type(data)}")
            return
    except KeyError as e:
        print(f"Missing key in extracted data: {e}")
        return

    # Construct a graph document
    graph_document = GraphDocument(
        nodes=[map_to_base_node(node) for node in knowledge_graph.nodes],
        relationships=[map_to_base_relationship(rel) for rel in knowledge_graph.rels],
        source=document
    )
    
    # Store information into Neo4j
    try:
        graph.add_graph_documents([graph_document])
        print("Successfully added graph document to Neo4j.")
    except Exception as e:
        print(f"Error while adding graph document to Neo4j: {e}")

def main():
    """
    Main function to execute the knowledge graph construction.
    """
    # Initialise Neo4j graph
    try:
        graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        print("Connected to Neo4j successfully.")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)
    
    # Initialise OpenAI LLM
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0, openai_api_key=OPENAI_API_KEY)
        print("Initialised OpenAI LLM successfully.")
    except Exception as e:
        print(f"Failed to initialise OpenAI LLM: {e}")
        sys.exit(1)
    
    # Load document from the web
    try:
        loader = WebBaseLoader(REPORT_URL)
        docs = loader.load()
        print(f"Loaded {len(docs)} documents from {REPORT_URL}.")
    except Exception as e:
        print(f"Failed to load documents: {e}")
        sys.exit(1)
    
    # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)
    
    # Split documents into manageable chunks
    try:
        documents = text_splitter.split_documents(docs)
        print(f"Split documents into {len(documents)} chunks.")
    except Exception as e:
        print(f"Failed to split documents: {e}")
        sys.exit(1)
    
    # Define allowed nodes and relationships
    allowed_nodes = ["Threat", "Vulnerability", "Asset", "Actor", "Attack", "Mitigation", "Indicator"]
    allowed_rels = ["TARGETS", "EXPLOITS", "PROTECTS", "MITIGATES", "ASSOCIATES_WITH", "IDENTIFIES"]
    
    # Initialise the extraction chain
    extract_chain = get_extraction_chain(allowed_nodes, allowed_rels, llm)
    
    # Process each document chunk and store in Neo4j
    for idx, document in tqdm(enumerate(documents), total=len(documents), desc="Processing Chunks"):
        print(f"\nProcessing chunk {idx + 1}/{len(documents)}...")
        extract_and_store_graph(document=document, extract_chain=extract_chain, graph=graph)
    
    print("\nKnowledge graph construction complete.")

if __name__ == "__main__":
    main()
