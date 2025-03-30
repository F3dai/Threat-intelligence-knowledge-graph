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

logger = logging.getLogger(__name__)


class GraphExtractor:
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
        Initialize the GraphExtractor (OpenAI).
        """
        self.allowed_nodes = allowed_nodes or DEFAULT_ALLOWED_NODES
        self.allowed_relationships = allowed_relationships or DEFAULT_ALLOWED_RELATIONSHIPS
        self.verbose = verbose
        self.model_name = model or OPENAI_CONFIG["model"]
        self.temperature = temperature if temperature is not None else OPENAI_CONFIG["temperature"]

        # Initialize the OpenAI client (error handling as before)
        try:
            api_key_to_use = api_key or OPENAI_CONFIG.get("api_key")
            if not api_key_to_use:
                 logger.warning("OpenAI API key not provided via argument or config. Attempting to use environment variable.")
            openai.api_key = api_key_to_use
            self.client = openai.OpenAI(api_key=api_key_to_use)
            logger.info(f"Initialized OpenAI client with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

        # --- Corrected Prompt Initialization ---
        # Store the base template (instructions WITHOUT the input part)
        try:
            # get_prompt_template returns the base instructions with allowed nodes/rels filled in,
            # and {{ }} converted to { } correctly for the example JSON.
            self.base_prompt_template = get_prompt_template(self.allowed_nodes, self.allowed_relationships)
            if self.verbose:
                 logger.debug(f"OpenAI base prompt template initialized:\n{self.base_prompt_template[:500]}...")

        except Exception as e:
             logger.error(f"Failed to create base prompt template: {e}")
             raise
        # --- End of Corrected Prompt Initialization ---


    def extract(self, text: str) -> Optional[KnowledgeGraph]:
        """
        Extract knowledge graph data from text.
        """
        if not self.base_prompt_template:
             logger.error("Base prompt template not initialized. Cannot extract.")
             return None

        try:
            # --- Corrected Prompt Construction ---
            # Construct the final prompt using an f-string or concatenation.
            # This avoids calling .format() on the base_prompt_template again.
            formatted_prompt = f"""{self.base_prompt_template}

## 8. Input Text
Process the following text:
{text}
"""
            # --- End of Corrected Prompt Construction ---


            # Define the knowledge graph schema (remains the same)
            schema = {
                "type": "object",
                "properties": { # ... schema details ... 
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


            # Call the OpenAI API (remains the same)
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a knowledge graph extraction system. Follow the instructions precisely."},
                    {"role": "user", "content": formatted_prompt} # Pass the combined prompt here
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

            # Extract function call arguments (remains the same)
            message = response.choices[0].message
            if not message.function_call:
                 logger.warning("No function call was returned by the OpenAI API.")
                 if message.content:
                      logger.warning(f"API returned content instead: {message.content}")
                 return None
            
            # Parse JSON arguments (remains the same)
            try:
                data = json.loads(message.function_call.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON arguments from function call: {e}")
                logger.error(f"Raw arguments string: {message.function_call.arguments}")
                return None

            # Create KnowledgeGraph (remains the same)
            try:
                return KnowledgeGraph(
                    nodes=data.get("nodes", []),
                    rels=data.get("rels", [])
                )
            except Exception as e:
                logger.error(f"Failed to create KnowledgeGraph from parsed data: {e}")
                logger.error(f"Parsed data: {data}")
                return None

        # Error handling (remains the same)
        except openai.APIError as e:
             logger.error(f"OpenAI API Error during extraction: {e}")
             return None
        except openai.RateLimitError as e:
             logger.error(f"OpenAI Rate Limit Error during extraction: {e}")
             return None
        except Exception as e:
             logger.error(f"Unexpected error during extraction: {e}", exc_info=True)
             return None

    # extract_from_document method remains the same as the last correct version
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
                 logger.error(f"Error accessing 'page_content' attribute: {e}")
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