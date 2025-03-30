"""
Helper functions used throughout the application.
"""

import logging
from typing import Dict, List, Optional, Any

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