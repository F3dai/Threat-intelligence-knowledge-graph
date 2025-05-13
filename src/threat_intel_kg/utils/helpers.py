"""
Helper functions used throughout the application.
"""

import logging
import json
import re
import datetime
import time
from typing import Dict, List, Optional, Any, Union

from ..models.data_models import Property

logger = logging.getLogger(__name__)


def format_property_key(s: str) -> str:
    """
    Convert a string to camelCase format.
    
    Args:
        s: The string to format.
    
    Returns:
        The formatted string.
    """
    if not s:
        logger.warning("Empty string provided to format_property_key")
        return ""
        
    words = s.split()
    if not words:
        return s
    
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def props_to_dict(props: Optional[List[Property]]) -> Dict[str, str]:
    """
    Convert a list of Property objects to a dictionary.
    
    Args:
        props: List of Property objects.
    
    Returns:
        Dictionary of properties.
    """
    properties = {}
    if not props:
        return properties
    
    for p in props:
        properties[format_property_key(p.key)] = p.value
    
    return properties


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary.
    
    Args:
        data: The dictionary to get the value from.
        key: The key to get the value for.
        default: The default value to return if the key is not found.
        
    Returns:
        The value for the key, or the default value if the key is not found.
    """
    try:
        return data.get(key, default)
    except (AttributeError, KeyError, TypeError):
        logger.warning(f"Failed to get key '{key}' from data structure")
        return default


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a readable string.
    
    Args:
        seconds: The duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def create_timestamp() -> str:
    """
    Create a formatted timestamp string.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def repair_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to repair malformed JSON.
    
    Args:
        json_str: The potentially malformed JSON string
        
    Returns:
        Fixed JSON data as a dictionary if repair is successful, None otherwise
    """
    if not isinstance(json_str, str):
        return None
        
    # First, check if we can already parse it cleanly
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If not, proceed with repairs
        pass
        
    # Try to find and fix unclosed strings that end with a quotation mark missing
    if '"id":"' in json_str and (json_str.endswith('"id') or json_str.endswith('"type')):
        # Add missing quotation mark and repair structure
        json_str = json_str + '""}'
        
    # Try to find and fix missing closing brackets
    def count_brackets(s):
        return s.count('{'), s.count('}'), s.count('['), s.count(']')
        
    open_braces, close_braces, open_brackets, close_brackets = count_brackets(json_str)
    if open_braces > close_braces:
        # Add missing closing braces
        json_str = json_str + "}" * (open_braces - close_braces)
        
    if open_brackets > close_brackets:
        # Add missing closing brackets
        json_str = json_str + "]" * (open_brackets - close_brackets)

    # Check if the JSON is potentially truncated mid-string
    if json_str.count('"') % 2 == 1:
        # Odd number of quotation marks means there's an unclosed string
        json_str = json_str + '"'
        
    # Try to fix truncated JSON by adding minimal valid structure
    if json_str.endswith('"id') or json_str.endswith('"type'):
        # Basic completion for common node reference patterns
        json_str = json_str + '""}'
        
    if json_str.endswith('{"source'):
        # Basic completion for truncated relationship
        json_str = json_str + ':{"id":"unknown","type":"unknown"},"target":{"id":"unknown","type":"unknown"},"type":"unknown"}]}'
            
    # Check if we have properly balanced brackets in the "nodes" array
    if '"nodes":' in json_str:
        nodes_start = json_str.find('"nodes":') + 8
        brackets_balance = 0
        in_nodes_array = False
        
        # Try to ensure proper array structure
        for i in range(nodes_start, len(json_str)):
            if json_str[i] == '[' and not in_nodes_array:
                in_nodes_array = True
                brackets_balance += 1
            elif json_str[i] == '[' and in_nodes_array:
                brackets_balance += 1
            elif json_str[i] == ']' and in_nodes_array:
                brackets_balance -= 1
                if brackets_balance == 0:
                    break
        
        # If we didn't find a closing bracket for the nodes array
        if in_nodes_array and brackets_balance > 0:
            position = json_str.find('"rels"')
            if position > 0:
                # Insert closing brackets before the rels key
                prefix = json_str[:position-1]
                suffix = json_str[position-1:]
                json_str = prefix + "]" * brackets_balance + suffix
            else:
                # Just append closing brackets at the end
                json_str = json_str + "]" * brackets_balance
            
    # After repairs, try to parse the JSON
    try:
        parsed_json = json.loads(json_str)
        
        # Fix case where nested arrays might have been created
        if "nodes" in parsed_json and isinstance(parsed_json["nodes"], list):
            if len(parsed_json["nodes"]) > 0 and isinstance(parsed_json["nodes"][0], list):
                parsed_json["nodes"] = parsed_json["nodes"][0]
                
        if "rels" in parsed_json and isinstance(parsed_json["rels"], list):
            if len(parsed_json["rels"]) > 0 and isinstance(parsed_json["rels"][0], list):
                parsed_json["rels"] = parsed_json["rels"][0]
        
        return parsed_json
        
    except json.JSONDecodeError:
        # If we still can't parse it, try to extract just the valid parts
        if '"nodes":' in json_str and '"rels":' in json_str:
            # Try to salvage just the nodes if they exist
            try:
                nodes_str = json_str.split('"nodes":')[1].split('"rels":')[0].strip()
                # Remove the trailing comma if present
                if nodes_str.endswith(','):
                    nodes_str = nodes_str[:-1]
                
                # Make sure the nodes string starts and ends with square brackets
                if not nodes_str.startswith('['):
                    nodes_str = '[' + nodes_str
                if not nodes_str.endswith(']'):
                    nodes_str = nodes_str + ']'
                    
                try:
                    nodes = json.loads(nodes_str)
                    if isinstance(nodes, list) and nodes and isinstance(nodes[0], list):
                        # If we have a nested list, flatten it
                        nodes = nodes[0]
                    return {"nodes": nodes, "rels": []}
                except:
                    # If we can't parse the nodes array, try one more approach
                    # Extract individual node objects using regex
                    node_objects = re.findall(r'\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"[^"]+"\s*(?:,\s*"properties"\s*:\s*\[\s*(?:\{\s*"key"\s*:\s*"[^"]+"\s*,\s*"value"\s*:\s*"[^"]+"\s*\}\s*,?\s*)*\s*\])?\s*\}', json_str)
                    
                    if node_objects:
                        valid_nodes = []
                        for node_obj in node_objects:
                            try:
                                node = json.loads(node_obj)
                                valid_nodes.append(node)
                            except:
                                pass
                                
                        if valid_nodes:
                            return {"nodes": valid_nodes, "rels": []}
            except Exception as e:
                logger.debug(f"Failed to extract nodes from malformed JSON: {e}")
        
        # If all else fails, return minimal valid structure
        return {"nodes": [], "rels": []}