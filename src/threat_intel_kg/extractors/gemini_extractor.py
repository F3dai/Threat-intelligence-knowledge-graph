import logging
import json
from typing import List, Dict, Any, Optional
import time

import google.generativeai as genai
from pydantic import ValidationError

from ..models import KnowledgeGraph, get_prompt_template
from ..config import DEFAULT_ALLOWED_NODES, DEFAULT_ALLOWED_RELATIONSHIPS, GEMINI_CONFIG

logger = logging.getLogger(__name__)


class GeminiGraphExtractor:
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

            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"temperature": self.temperature},
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
            prompt = f"""Use the given format (specified in the system instructions) to extract information from the following cyber security threat report:

{text}

Tip: Ensure that the output adheres to the Neo4j-compatible JSON format specified in the instructions. Return only valid JSON with nodes and rels properties."""

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

            json_data = self._extract_json_from_response(response.text)

            if not json_data:
                # Logging is handled within _extract_json_from_response if it fails
                return None

            try:
                nodes = json_data.get("nodes", [])
                rels = json_data.get("rels", [])
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
            # Fall through to final warning

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