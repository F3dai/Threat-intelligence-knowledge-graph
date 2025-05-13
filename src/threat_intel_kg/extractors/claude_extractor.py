"""
Graph extraction module for parsing security reports into structured knowledge graphs using Claude.
"""

import logging
import json
from typing import List, Dict, Any, Optional
import time

from anthropic import Anthropic

from ..models import KnowledgeGraph, get_prompt_template
from ..config import DEFAULT_ALLOWED_NODES, DEFAULT_ALLOWED_RELATIONSHIPS, CLAUDE_CONFIG
from ..utils.helpers import repair_json

logger = logging.getLogger(__name__)


class ClaudeGraphExtractor:
    """
    Extract knowledge graph data from security reports using Anthropic's Claude models.
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
        self.allowed_nodes = allowed_nodes or DEFAULT_ALLOWED_NODES
        self.allowed_relationships = allowed_relationships or DEFAULT_ALLOWED_RELATIONSHIPS
        self.verbose = verbose
        self.model_name = model or CLAUDE_CONFIG["model"]
        self.temperature = temperature if temperature is not None else CLAUDE_CONFIG["temperature"]

        try:
            self.system_prompt = get_prompt_template(self.allowed_nodes, self.allowed_relationships)
        except Exception as e:
            logger.error(f"Failed to generate system prompt: {e}")
            raise

        try:
            api_key_to_use = api_key or CLAUDE_CONFIG.get("api_key")
            if not api_key_to_use:
                logger.warning("Claude API key not provided via argument or config. Attempting environment variable.")

            self.client = Anthropic(api_key=api_key_to_use)
            logger.info(f"Initialized Claude model {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Claude model: {e}", exc_info=True)
            raise

    def extract(self, text: str) -> Optional[KnowledgeGraph]:
        if not hasattr(self, 'client') or not self.client:
            logger.error("Claude client not initialized. Cannot extract.")
            return None

        try:
            # Check if we're using Claude 3.5 Sonnet to provide a clearer prompt
            model_name = getattr(self, 'model_name', '').lower()
            if 'sonnet' in model_name:
                final_prompt = f"""## Input Text for Knowledge Graph Extraction
Process the following text to extract a knowledge graph:

{text}

IMPORTANT: Your response must be a valid JSON object with proper syntax. 
Do NOT use nested JSON strings or any other formatting that would make the JSON invalid. 
Return ONLY valid JSON with 'nodes' and 'rels' arrays.
"""
            else:
                final_prompt = f"""## 8. Input Text
Process the following text:
{text}"""

            # Prepare schema as a JSON schema for the Claude response
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

            # For Claude 3.5 Sonnet, we need to be extra careful about the tool configuration
            if 'sonnet' in getattr(self, 'model_name', '').lower():
                response = self.client.messages.create(
                    model=self.model_name,
                    temperature=min(self.temperature, 0.2),  # Lower temperature for more structured output
                    system=self.system_prompt,
                    max_tokens=4000,
                    messages=[
                        {"role": "user", "content": final_prompt}
                    ],
                    tools=[{
                        "name": "extract_knowledge_graph",
                        "description": "Extract a structured knowledge graph from text. Return a valid JSON object with 'nodes' and 'rels' arrays.",
                        "input_schema": schema
                    }],
                    tool_choice={"type": "tool", "name": "extract_knowledge_graph"}
                )
            else:
                # Standard configuration for other Claude models
                response = self.client.messages.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    system=self.system_prompt,
                    max_tokens=4000,
                    messages=[
                        {"role": "user", "content": final_prompt}
                    ],
                    tools=[{
                        "name": "extract_knowledge_graph",
                        "description": "Extract a knowledge graph from text according to the specified schema.",
                        "input_schema": schema
                    }],
                    tool_choice={"type": "tool", "name": "extract_knowledge_graph"}
                )

            if self.verbose:
                try:
                    # Log response text safely
                    log_text = str(response)[:1000] + ('...' if len(str(response)) > 1000 else '')
                    logger.debug(f"Raw Claude response (truncated): {log_text}")
                except Exception as log_e:
                    logger.warning(f"Could not log Claude response: {log_e}")

            # Extract tool call content from the response
            if not response.content or len(response.content) == 0:
                logger.warning("No content was returned by the Claude API.")
                logger.warning(f"Raw response: {response}")
                return None

            # Parse the tool call output
            try:
                # Access the tool outputs from the response
                for content_block in response.content:
                    if content_block.type == 'tool_use':
                        # Direct access to the input without .tool_use
                        json_data = content_block.input
                        
                        if not json_data:
                            logger.warning("Empty JSON data from tool use.")
                            continue
                            
                        # Add debug logging to see the structure
                        if self.verbose:
                            logger.debug(f"Raw JSON data structure: {type(json_data)}")
                            if isinstance(json_data, dict):
                                logger.debug(f"JSON keys: {list(json_data.keys())}")
                        
                        # Improved error handling for JSON parsing
                        try:
                            # Handle case where nodes or rels might be strings instead of lists
                            nodes = json_data.get("nodes", [])
                            rels = json_data.get("rels", [])
                            
                            # Specially handle Claude-3.5-Sonnet responses
                            # In Claude-3.5-Sonnet, we've observed that sometimes the response comes with malformed JSON
                            model_name = getattr(self, 'model_name', '').lower()
                            if 'sonnet' in model_name:
                                # Try a more aggressive approach for fixing Claude-3.5-Sonnet responses
                                if isinstance(nodes, str):
                                    logger.info("Claude Sonnet: Nodes field is a string, attempting parsing")
                                    fixed_json = repair_json(nodes)
                                    if fixed_json and "nodes" in fixed_json:
                                        nodes = fixed_json["nodes"]
                                    elif isinstance(fixed_json, list):
                                        nodes = fixed_json
                                    else:
                                        nodes = []
                                
                                if isinstance(rels, str):
                                    logger.info("Claude Sonnet: Rels field is a string, attempting parsing")
                                    fixed_json = repair_json(rels)
                                    if fixed_json and "rels" in fixed_json:
                                        rels = fixed_json["rels"]
                                    elif isinstance(fixed_json, list):
                                        rels = fixed_json
                                    else:
                                        rels = []
                            else:
                                # Standard handling for other models
                                # If nodes or rels are strings (JSON strings), try to parse them
                                if isinstance(nodes, str):
                                    logger.debug("Nodes field is a string, attempting to parse as JSON")
                                    try:
                                        import json
                                        nodes = json.loads(nodes)
                                    except json.JSONDecodeError:
                                        logger.warning("Failed to parse nodes string as JSON")
                                        # Try a simpler approach - create an empty list instead of failing
                                        nodes = []
                                
                                if isinstance(rels, str):
                                    logger.debug("Rels field is a string, attempting to parse as JSON")
                                    try:
                                        import json
                                        rels = json.loads(rels)
                                    except json.JSONDecodeError:
                                        logger.warning("Failed to parse rels string as JSON")
                                        # Try a simpler approach - create an empty list instead of failing
                                        rels = []
                                    
                            # Check if we have valid data after all parsing attempts
                            if not nodes and not rels:
                                logger.warning("Extracted JSON has no 'nodes' or 'rels' after processing.")
                                logger.warning(f"JSON structure keys found: {list(json_data.keys()) if isinstance(json_data, dict) else 'not a dict'}")
                                continue
                            
                            # Create the knowledge graph with the parsed data
                            return KnowledgeGraph(nodes=nodes, rels=rels)
                        
                        except Exception as parsing_e:
                            logger.warning(f"Error processing JSON data: {parsing_e}")
                            # Try a fallback approach if we can detect something to salvage
                            if isinstance(json_data, dict) and ('nodes' in str(json_data) or 'rels' in str(json_data)):
                                logger.info("Attempting to create minimal knowledge graph from problematic data")
                                # Return an empty knowledge graph rather than None so processing continues
                                return KnowledgeGraph(nodes=[], rels=[])
                
                logger.warning("No valid tool_use content found in the response.")
                return None
            except Exception as e:
                logger.error(f"Error processing Claude response: {e}", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"Error during Claude extraction: {e}", exc_info=True)
            return None

    def extract_from_document(self, document_content) -> Optional[KnowledgeGraph]:
        """
        Extract knowledge graph data from a document.

        Args:
            document_content: The document to extract from. Can be either a string,
                            an object with a 'page_content' attribute, or
                            a dict with a 'page_content' key.

        Returns:
            The extracted knowledge graph or None if extraction failed.
        """
        text_content = None
        if isinstance(document_content, str):
            text_content = document_content
        elif isinstance(document_content, dict) and 'page_content' in document_content:
            text_content = document_content['page_content']
        elif hasattr(document_content, 'page_content'):
            try:
                text_content = document_content.page_content
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
            if isinstance(document_content, dict):
                logger.warning(f"Unsupported dictionary structure: Missing 'page_content' key. Keys found: {list(document_content.keys())}")
            else:
                logger.warning(f"Unsupported document type: {type(document_content)}")
            return None