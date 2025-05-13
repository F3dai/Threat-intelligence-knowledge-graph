"""
Graph extraction module for parsing security reports into structured knowledge graphs using Google's Gemini models.
"""

import logging
import json
from typing import List, Dict, Any, Optional
import time

import google.generativeai as genai
from pydantic import ValidationError

from ..models import KnowledgeGraph, get_prompt_template
from ..config import DEFAULT_ALLOWED_NODES, DEFAULT_ALLOWED_RELATIONSHIPS, GEMINI_CONFIG
from ..utils.helpers import repair_json

logger = logging.getLogger(__name__)


class GeminiGraphExtractor:
    """
    Extract knowledge graph data from security reports using Google's Gemini models.
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
        self.model_name = model or GEMINI_CONFIG["model"]
        self.temperature = temperature if temperature is not None else GEMINI_CONFIG["temperature"]

        try:
            self.system_prompt = get_prompt_template(self.allowed_nodes, self.allowed_relationships)
        except Exception as e:
            logger.error(f"Failed to generate system prompt: {e}")
            raise

        try:
            api_key_to_use = api_key or GEMINI_CONFIG.get("api_key")
            if not api_key_to_use:
                logger.warning("Gemini API key not provided via argument or config. Attempting environment variable.")

            genai.configure(api_key=api_key_to_use)
            
            # Get the model name in the correct format
            # If it doesn't start with 'models/', add it
            full_model_name = self.model_name
            if not full_model_name.startswith("models/"):
                # Check if this is a model name that's in the list
                available_models = [m.name for m in genai.list_models() 
                                   if 'generateContent' in m.supported_generation_methods]
                
                # Try to find an exact match
                exact_match = [m for m in available_models if m.endswith(f"/{self.model_name}")]
                if exact_match:
                    full_model_name = exact_match[0]
                else:
                    # Try to find a fuzzy match
                    fuzzy_match = [m for m in available_models if self.model_name in m]
                    if fuzzy_match:
                        full_model_name = fuzzy_match[0]
                    else:
                        # If no match, use as is and let the API handle the error
                        full_model_name = f"models/{self.model_name}"
                        
                logger.info(f"Resolved model name '{self.model_name}' to '{full_model_name}'")
                self.model_name = full_model_name

            # Setup more specific config for different model types
            if "flash" in self.model_name.lower():
                # For Flash model, use a lower temperature for more structured output
                generation_config = {
                    "temperature": min(self.temperature, 0.2),  # Lower temperature for structured output
                    "response_mime_type": "application/json",   # Hint for JSON response
                    "top_p": 0.95,
                    "top_k": 40
                }
            else:
                # Default config for other models
                generation_config = {"temperature": self.temperature}
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                system_instruction=self.system_prompt
            )
            logger.info(f"Initialized Gemini model {self.model_name} with system instruction.")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
            raise

    def extract(self, text: str) -> Optional[KnowledgeGraph]:
        if not hasattr(self, 'model') or not self.model:
            logger.error("Gemini model not initialized. Cannot extract.")
            return None

        try:
            prompt = f"""Extract a knowledge graph from the following cyber security threat report:

{text}

IMPORTANT: Return a valid JSON object with the EXACT format below:
{{
  "nodes": [
    {{
      "id": "unique_id_string",
      "type": "NodeType",
      "properties": [
        {{"key": "propertyName", "value": "propertyValue"}}
      ]
    }}
  ],
  "rels": [
    {{
      "source": {{"id": "source_node_id", "type": "SourceNodeType"}},
      "target": {{"id": "target_node_id", "type": "TargetNodeType"}},
      "type": "RELATIONSHIP_TYPE",
      "properties": [
        {{"key": "propertyName", "value": "propertyValue"}}
      ]
    }}
  ]
}}

Do not include any explanations, markdown formatting, or anything else outside the JSON. The JSON object must contain the top-level 'nodes' and 'rels' arrays."""

            response = self.model.generate_content(
                [{"role": "user", "parts": [prompt]}]
            )

            if self.verbose:
                try:
                    # Log response text safely
                    response_text_for_log = getattr(response, 'text', '[No text attribute found]')
                    log_text = response_text_for_log[:1000] + ('...' if len(response_text_for_log) > 1000 else '')
                    logger.debug(f"Raw Gemini response text (truncated): {log_text}")
                except Exception as log_e:
                    logger.warning(f"Could not log Gemini response text: {log_e}")

            # For gemini-2.0-flash which doesn't handle JSON generation well, try to provide clearer instructions
            if "flash" in self.model_name.lower() and not response.text.strip().startswith('{'):
                # If the response doesn't start with JSON, try a second attempt with clearer instructions
                logger.info("Initial response from Gemini Flash doesn't appear to be JSON. Trying a second attempt with more explicit JSON instructions.")
                
                retry_prompt = f"""You need to extract information from the text and return ONLY a valid JSON document with the following structure:
{{
  "nodes": [
    {{
      "id": "unique_id_string",
      "type": "NodeType",
      "properties": [
        {{"key": "propertyName", "value": "propertyValue"}}
      ]
    }}
  ],
  "rels": [
    {{
      "source": {{"id": "source_node_id", "type": "SourceNodeType"}},
      "target": {{"id": "target_node_id", "type": "TargetNodeType"}},
      "type": "RELATIONSHIP_TYPE",
      "properties": [
        {{"key": "propertyName", "value": "propertyValue"}}
      ]
    }}
  ]
}}

Extract the cybersecurity entities and relationships from this text:
{text}

IMPORTANT: Return ONLY the JSON object without any explanations, markdown formatting, or extra text.
"""
                retry_response = self.model.generate_content(
                    [{"role": "user", "parts": [retry_prompt]}]
                )
                
                # Try to extract JSON from the retry response
                json_data = self._extract_json_from_response(retry_response.text)
            else:
                # Use the original response for other models
                json_data = self._extract_json_from_response(response.text)

            if not json_data:
                # Logging is handled within _extract_json_from_response if it fails
                return None

            try:
                nodes = json_data.get("nodes", [])
                rels = json_data.get("rels", [])
                
                # Handle special case where nodes or rels might be strings
                if isinstance(nodes, str):
                    logger.info("Nodes field is a string rather than a list, attempting to parse")
                    try:
                        parsed_nodes = json.loads(nodes)
                        if isinstance(parsed_nodes, list):
                            nodes = parsed_nodes
                        else:
                            logger.warning("Parsed nodes is not a list; defaulting to empty list")
                            nodes = []
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse nodes string; defaulting to empty list")
                        nodes = []
                
                if isinstance(rels, str):
                    logger.info("Rels field is a string rather than a list, attempting to parse")
                    try:
                        parsed_rels = json.loads(rels)
                        if isinstance(parsed_rels, list):
                            rels = parsed_rels
                        else:
                            logger.warning("Parsed rels is not a list; defaulting to empty list")
                            rels = []
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse rels string; defaulting to empty list")
                        rels = []
                
                # If still no valid nodes or relationships, try deep parsing of the response
                if (not nodes or not rels) and isinstance(json_data, dict):
                    # Look for nested data structures
                    logger.info("Looking for nested data structures in response")
                    for key, value in json_data.items():
                        if isinstance(value, dict):
                            if 'nodes' in value and not nodes:
                                nodes = value.get('nodes', [])
                                logger.info(f"Found nodes in nested field '{key}'")
                            if 'rels' in value and not rels:
                                rels = value.get('rels', [])
                                logger.info(f"Found rels in nested field '{key}'")
                
                if not nodes and not rels:
                    logger.warning("Extracted JSON has no 'nodes' or 'rels'.")
                    if not json_data: # Should not happen if json_data is truthy, but defensive check
                        logger.warning("JSON data was empty despite passing initial check.")
                    else:
                        logger.warning(f"JSON structure keys found: {list(json_data.keys())}")
                    return None
                
                return KnowledgeGraph(nodes=nodes, rels=rels)
            except ValidationError as e:
                logger.warning(f"Pydantic validation error converting response to KnowledgeGraph: {e}")
                # Attempt manual construction only if structure seems correct despite validation error
                if isinstance(json_data.get("nodes"), list) and isinstance(json_data.get("rels"), list):
                    logger.info("Attempting basic KnowledgeGraph construction despite validation error.")
                    return KnowledgeGraph(
                        nodes=json_data.get("nodes", []),
                        rels=json_data.get("rels", [])
                    )
                logger.warning("Cannot attempt manual construction due to unexpected data types for nodes/rels.")
                return None

        except Exception as e:
            logger.error(f"Error during Gemini extraction: {e}", exc_info=True)
            return None

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        if not response_text:
            logger.warning("Received empty response text from model.")
            return {}
            
        is_flash_model = hasattr(self, 'model_name') and "flash" in getattr(self, 'model_name', '').lower()

        # Attempt 1: Parse ```json block
        if "```json" in response_text:
            try:
                json_block = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
                logger.debug("Attempting to parse content within ```json block.")
                return json.loads(json_block)
            except (IndexError, json.JSONDecodeError) as e:
                logger.debug(f"Failed to parse ```json block: {e}. Trying next method.")
                # Fall through

        # Attempt 2: Parse ``` block (if not ```json)
        elif "```" in response_text:
             try:
                  json_block = response_text.split("```", 1)[1].split("```", 1)[0].strip()
                  if json_block.startswith('{') and json_block.endswith('}'):
                       logger.debug("Attempting to parse content within ``` block.")
                       return json.loads(json_block)
                  else:
                       logger.debug("Content within ``` block doesn't look like JSON.")
             except (IndexError, json.JSONDecodeError) as e:
                  logger.debug(f"Failed to parse ``` block: {e}. Trying next method.")
                  # Fall through

        # Attempt 3: Parse entire response directly
        try:
            logger.debug("Attempting to parse entire response text as JSON.")
            return json.loads(response_text)
        except json.JSONDecodeError as e:
             logger.debug(f"Failed to parse entire response as JSON: {e}. Trying substring.")
             # Fall through

        # Attempt 4: Parse substring between {}
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                logger.debug("Attempting to parse substring between {} as JSON.")
                return json.loads(json_str)
            else:
                 logger.debug("Could not find valid JSON object markers '{' and '}' in response for substring parsing.")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extracted JSON substring: {e}")
            # Fall through to next method for Flash models
        
        # For Flash models, additional repair attempts
        if is_flash_model:
            # Attempt 5: Extra JSON repair for Flash model responses
            try:
                logger.debug("Attempting special JSON repair for Flash model response...")
                # Try to fix common issues like unquoted property names
                if response_text.find('{') >= 0 and response_text.rfind('}') > 0:
                    json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
                    
                    # Create a simple minimal structure if nothing else works
                    if "nodes" in json_str and "rels" in json_str:
                        # Find the indices right after the opening brace and before the closing brace
                        open_brace_idx = json_str.find('{')
                        close_brace_idx = json_str.rfind('}')
                        
                        # Create a minimalist JSON with empty arrays
                        fixed_json = '{"nodes":[],"rels":[]}'
                        logger.warning("Created fallback minimal JSON structure with empty nodes and rels")
                        parsed_json = json.loads(fixed_json)
                        
                        # Try to extract useful nodes and relationships manually from the malformed JSON
                        if "nodes" in json_str and "rels" in json_str:
                            logger.info("Attempting to manually extract nodes and relationships from malformed JSON")
                            try:
                                # Try to repair common JSON errors for nodes
                                nodes_start = json_str.find('"nodes"') 
                                if nodes_start != -1:
                                    nodes_data = json_str[nodes_start:].split('"rels"')[0]
                                    
                                    # If we find node objects with proper format, try to parse them
                                    import re
                                    node_objects = re.findall(r'\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"[^"]+"\s*(?:,\s*"properties"\s*:\s*\[\s*(?:\{\s*"key"\s*:\s*"[^"]+"\s*,\s*"value"\s*:\s*"[^"]+"\s*\}\s*,?\s*)*\s*\])?\s*\}', json_str)
                                    
                                    if node_objects:
                                        logger.info(f"Found {len(node_objects)} properly formatted node objects")
                                        # Try to parse each node object
                                        valid_nodes = []
                                        for node_obj in node_objects:
                                            try:
                                                fixed_node = node_obj.replace("'", '"')  # Replace single quotes with double quotes
                                                node = json.loads(fixed_node)
                                                valid_nodes.append(node)
                                            except:
                                                continue  # Skip if parsing fails
                                                
                                        if valid_nodes:
                                            logger.info(f"Successfully parsed {len(valid_nodes)} nodes")
                                            parsed_json["nodes"] = valid_nodes
                            except Exception as e:
                                logger.warning(f"Manual node extraction failed: {e}")
                        
                        return parsed_json
            except Exception as e:
                logger.debug(f"Special JSON repair failed: {e}")
                # Fall through to final warning

        # Final attempt - try our centralized JSON repair function
        try:
            logger.info("Attempting repair of malformed JSON as last resort")
            repaired_json = repair_json(response_text)
            if repaired_json:
                logger.info("Successfully repaired malformed JSON")
                return repaired_json
        except Exception as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}", exc_info=True)
                
        # If all attempts failed
        logger.warning("Could not extract valid JSON from response after all attempts.")
        if self.verbose:
             logger.debug(f"Response text when JSON extraction failed: {response_text[:500]}...")
        return {}


    def extract_from_document(self, document_content: str) -> Optional[KnowledgeGraph]:
        if not isinstance(document_content, str):
            logger.error(f"extract_from_document received non-string input: {type(document_content)}")
            return None
        if not document_content.strip():
            logger.warning("Document content is empty or whitespace.")
            return None
        return self.extract(document_content)