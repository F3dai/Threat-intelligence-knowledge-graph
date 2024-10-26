"""
Contains utility functions that are used across multiple modules or scripts
"""

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
