"""
Graph extraction module for parsing security reports into structured knowledge graphs using OpenAI.
"""

import logging
import json
from typing import List, Dict, Any, Optional
import openai

# Ensure these imports point to the correct location in your project structure
from ..models import KnowledgeGraph, get_prompt_template
from ..config import DEFAULT_ALLOWED_NODES, DEFAULT_ALLOWED_RELATIONSHIPS, OPENAI_CONFIG
from ..utils.helpers import repair_json

logger = logging.getLogger(__name__)


class OpenAIGraphExtractor:
    """
    Extract knowledge graph data from security reports using OpenAI's models.
    """
    
    def __init__(
        self,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the OpenAIGraphExtractor.
        
        Args:
            allowed_nodes: List of allowed node types
            allowed_relationships: List of allowed relationship types
            model: OpenAI model name to use
            temperature: Temperature setting for generation
            api_key: OpenAI API key
            verbose: Whether to enable verbose logging
        """
        self.allowed_nodes = allowed_nodes or DEFAULT_ALLOWED_NODES
        self.allowed_relationships = allowed_relationships or DEFAULT_ALLOWED_RELATIONSHIPS
        self.verbose = verbose
        self.model_name = model or OPENAI_CONFIG["model"]
        self.temperature = temperature if temperature is not None else OPENAI_CONFIG["temperature"]

        # Initialize the OpenAI client
        try:
            api_key_to_use = api_key or OPENAI_CONFIG.get("api_key")
            if not api_key_to_use:
                 logger.warning("OpenAI API key not provided via argument or config. Attempting to use environment variable.")
            openai.api_key = api_key_to_use
            self.client = openai.OpenAI(api_key=api_key_to_use)
            logger.info(f"Initialized OpenAI client with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise

        # Initialize prompt template
        try:
            # Get the base instructions with allowed nodes/relationships filled in
            self.base_prompt_template = get_prompt_template(self.allowed_nodes, self.allowed_relationships)
            if self.verbose:
                 logger.debug(f"OpenAI base prompt template initialized:\n{self.base_prompt_template[:500]}...")
        except Exception as e:
             logger.error(f"Failed to create base prompt template: {e}", exc_info=True)
             raise

    def extract(self, text: str) -> Optional[KnowledgeGraph]:
        """
        Extract knowledge graph data from text.
        
        Args:
            text: The text to extract knowledge graph from
            
        Returns:
            KnowledgeGraph object or None if extraction failed
        """
        if not self.base_prompt_template:
             logger.error("Base prompt template not initialized. Cannot extract.")
             return None

        try:
            # Construct the final prompt
            formatted_prompt = f"""{self.base_prompt_template}

## 8. Input Text
Process the following text:
{text}
"""
            # Define the knowledge graph schema
            schema = {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "properties": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "key": {"type": "string"},
                                            "value": {"type": "string"}
                                        },
                                        "required": ["key", "value"]
                                    }
                                }
                            },
                            "required": ["id", "type"]
                        }
                    },
                    "rels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "type": {"type": "string"}
                                    },
                                    "required": ["id", "type"]
                                },
                                "target": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "type": {"type": "string"}
                                    },
                                    "required": ["id", "type"]
                                },
                                "type": {"type": "string"},
                                "properties": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "key": {"type": "string"},
                                            "value": {"type": "string"}
                                        },
                                        "required": ["key", "value"]
                                    }
                                }
                            },
                            "required": ["source", "target", "type"]
                        }
                    }
                },
                 "required": ["nodes", "rels"]
            }

            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a knowledge graph extraction system. Follow the instructions precisely."},
                    {"role": "user", "content": formatted_prompt}
                ],
                functions=[{
                    "name": "extract_knowledge_graph",
                    "description": "Extract a knowledge graph from text according to the specified schema.",
                    "parameters": schema
                }],
                function_call={"name": "extract_knowledge_graph"}
            )

            if self.verbose:
                logger.debug(f"OpenAI API raw response choice: {response.choices[0]}")

            # Extract function call arguments
            message = response.choices[0].message
            if not message.function_call:
                 logger.warning("No function call was returned by the OpenAI API.")
                 if message.content:
                      logger.warning(f"API returned content instead: {message.content}")
                 return None
            
            # Parse JSON arguments with error handling
            try:
                data = json.loads(message.function_call.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON arguments from function call: {e}")
                
                # Log portion of the arguments if very large
                if len(message.function_call.arguments) > 1000:
                    logger.error(f"Raw arguments beginning: {message.function_call.arguments[:500]}")
                    logger.error(f"Raw arguments ending: {message.function_call.arguments[-500:]}")
                else:
                    logger.error(f"Raw arguments string: {message.function_call.arguments}")
                
                # Try to repair the JSON using the centralized utility
                fixed_json = repair_json(message.function_call.arguments)
                if fixed_json:
                    logger.info("Successfully repaired malformed JSON. Proceeding with fixed data.")
                    data = fixed_json
                    
                    # Additional validation for nested nodes/rels arrays
                    if "nodes" in data and isinstance(data["nodes"], list) and data["nodes"] and isinstance(data["nodes"][0], list):
                        logger.warning("Detected nested nodes array - flattening structure")
                        data["nodes"] = data["nodes"][0]
                        
                    if "rels" in data and isinstance(data["rels"], list) and data["rels"] and isinstance(data["rels"][0], list):
                        logger.warning("Detected nested rels array - flattening structure")
                        data["rels"] = data["rels"][0]
                else:
                    logger.error("Could not repair JSON data")
                    return None

            # Create and return KnowledgeGraph
            try:
                return KnowledgeGraph(
                    nodes=data.get("nodes", []),
                    rels=data.get("rels", [])
                )
            except Exception as e:
                logger.error(f"Failed to create KnowledgeGraph from parsed data: {e}", exc_info=True)
                logger.error(f"Parsed data: {data}")
                return None

        # Specific error handling for OpenAI API errors
        except openai.APIError as e:
             logger.error(f"OpenAI API Error during extraction: {e}", exc_info=True)
             return None
        except openai.RateLimitError as e:
             logger.error(f"OpenAI Rate Limit Error during extraction: {e}", exc_info=True)
             return None
        except Exception as e:
             logger.error(f"Unexpected error during extraction: {e}", exc_info=True)
             return None

    def extract_from_document(self, document) -> Optional[KnowledgeGraph]:
        """
        Extract knowledge graph data from a document.

        Args:
            document: The document to extract from. Can be either a string,
                     an object with a 'page_content' attribute, or
                     a dict with a 'page_content' key.

        Returns:
            The extracted knowledge graph or None if extraction failed.
        """
        text_content = None
        if isinstance(document, str):
            text_content = document
        elif isinstance(document, dict) and 'page_content' in document:
            text_content = document['page_content']
        elif hasattr(document, 'page_content'):
             try:
                 text_content = document.page_content
                 if not isinstance(text_content, str):
                      logger.warning(f"Document attribute 'page_content' is not a string (type: {type(text_content)}). Attempting conversion.")
                      text_content = str(text_content)
             except Exception as e:
                 logger.error(f"Error accessing 'page_content' attribute: {e}", exc_info=True)
                 return None
        
        if text_content is not None:
             if not text_content.strip(): 
                  logger.warning("Document content is empty or whitespace.")
                  return None
             return self.extract(text_content)
        else:
            if isinstance(document, dict):
                 logger.warning(f"Unsupported dictionary structure: Missing 'page_content' key. Keys found: {list(document.keys())}")
            else:
                 logger.warning(f"Unsupported document type: {type(document)}")
            return None