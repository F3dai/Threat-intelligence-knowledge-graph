"""
Module for extracting STIX relationships from threat intelligence reports.
"""

import os
import csv
import logging
import re
from typing import Dict, List, Tuple, Set, Optional
import nltk
from nltk.tokenize import sent_tokenize

# Import centralized STIX mappings
from ..models.stix_mappings import STIX_TYPE_MAPPING, STIX_TO_OUR_TYPE

# Configure logging
logger = logging.getLogger(__name__)

class STIXRelationExtractor:
    """Extract relationships between entities based on STIX relationship definitions."""
    
    def __init__(self):
        """Initialize the STIX relationship extractor."""
        self.relations = {}
        self.relation_verbs = set()
        self._load_relations()
        
        # Ensure NLTK resources are available
        nltk_resources = [('punkt', 'tokenizers/punkt'), ('wordnet', 'corpora/wordnet')]
        for resource_name, resource_path in nltk_resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info(f"Downloading NLTK {resource_name}...")
                nltk.download(resource_name, quiet=True)
            
    def _load_relations(self):
        """Load STIX relationship definitions from the CSV file."""
        try:
            # Get the path to the Relations.csv file
            stixnet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "STIXnet")
            relations_path = os.path.join(stixnet_path, "Relation-Extraction", "Relations.csv")
            
            # Check if the file exists
            if not os.path.exists(relations_path):
                logger.warning(f"STIX relations file not found at {relations_path}")
                # Attempt to find it relative to current file
                current_dir = os.path.dirname(os.path.abspath(__file__))
                alt_paths = [
                    os.path.join(current_dir, "..", "..", "..", "STIXnet", "Relation-Extraction", "Relations.csv"),
                    os.path.join(current_dir, "..", "..", "STIXnet", "Relation-Extraction", "Relations.csv"),
                    os.path.join(current_dir, "..", "STIXnet", "Relation-Extraction", "Relations.csv")
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(os.path.abspath(alt_path)):
                        relations_path = os.path.abspath(alt_path)
                        logger.info(f"Found STIX relations file at {relations_path}")
                        break
            
            # If we found the file, read it
            if os.path.exists(relations_path):
                with open(relations_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    
                    for row in reader:
                        if len(row) >= 4:
                            src, rel_type, dst, reverse = row
                            
                            # Create a key for source->destination relationships
                            key = (src, dst)
                            if key not in self.relations:
                                self.relations[key] = []
                                
                            # Add this relationship type
                            self.relations[key].append(rel_type)
                            self.relation_verbs.add(rel_type)
                    
                logger.info(f"Loaded {len(self.relations)} STIX relationship definitions")
                if logger.isEnabledFor(logging.DEBUG) and self.relation_verbs:
                    # Only log verb list in debug mode and only if we have verbs
                    logger.debug(f"First 10 relationship verbs: {', '.join(sorted(list(self.relation_verbs))[:10])}...")
            else:
                logger.warning("Could not find STIX relations file, using fallback relationships")
                self._setup_basic_relationships()
                
        except Exception as e:
            logger.error(f"Error loading STIX relationship definitions: {e}")
            # Use a basic set of relationships as fallback
            self._setup_basic_relationships()
    
    def _setup_basic_relationships(self):
        """Set up basic relationships as fallback if loading from file fails."""
        # Some basic relationships
        relations = [
            ("threat-actor", "uses", "malware", "used-by"),
            ("threat-actor", "uses", "tool", "used-by"),
            ("threat-actor", "targets", "identity", "targeted-by"),
            ("threat-actor", "targets", "vulnerability", "targeted-by"),
            ("malware", "targets", "vulnerability", "targeted-by"),
            ("malware", "communicates-with", "ipv4-addr", "communicated-with"),
            ("malware", "communicates-with", "domain-name", "communicated-with"),
            ("malware", "communicates-with", "url", "communicated-with"),
            ("domain-name", "resolves-to", "ipv4-addr", "resolved-from"),
            ("url", "contains", "domain-name", "contained-in")
        ]
        
        for src, rel_type, dst, reverse in relations:
            key = (src, dst)
            if key not in self.relations:
                self.relations[key] = []
            self.relations[key].append(rel_type)
            self.relation_verbs.add(rel_type)
            
        logger.info("Using basic relationship definitions as fallback")
    
    def extract_relationships(self, entities: List[Dict], text: str) -> List[Dict]:
        """
        Extract relationships between entities based on STIX definitions.
        
        Args:
            entities: List of entity dictionaries with type and value
            text: The full text from which entities were extracted
            
        Returns:
            List of relationship dictionaries
        """
        if not entities:
            return []
            
        # Split text into sentences for better relationship context
        try:
            # Try to use NLTK's sentence tokenizer
            try:
                sentences = sent_tokenize(text)
            except (LookupError, Exception) as e:
                # If NLTK tokenization fails, try downloads one more time
                try:
                    nltk.download('punkt')
                    sentences = sent_tokenize(text)
                except Exception as inner_e:
                    # If still failing, use regex fallback
                    logger.warning("NLTK sentence tokenization failed. Using regex fallback.")
                    sentences = re.split(r'[.!?]+\s+', text)
        except Exception as e:
            logger.warning(f"Error in sentence splitting: {e}. Using basic approach.")
            # Ultimate fallback - just put everything in one sentence
            sentences = [text]
        
        # Group entities by sentence for contextual analysis
        entities_by_sentence = {}
        for entity in entities:
            entity_type = entity.get('type')
            entity_value = entity.get('value', '')
            
            # Skip entities without proper type or value
            if not entity_type or not entity_value:
                continue
                
            # Convert to STIX type for relationship lookup
            stix_type = STIX_TYPE_MAPPING.get(entity_type)
            if not stix_type:
                continue
                
            # Find entity in sentences
            for i, sentence in enumerate(sentences):
                if entity_value in sentence:
                    if i not in entities_by_sentence:
                        entities_by_sentence[i] = []
                    entities_by_sentence[i].append({
                        'original_type': entity_type,
                        'stix_type': stix_type,
                        'value': entity_value,
                        'entity': entity
                    })
        
        # Extract relationships within each sentence
        relationships = []
        
        for sentence_idx, sentence_entities in entities_by_sentence.items():
            if len(sentence_entities) < 2:
                continue
            
            sentence = sentences[sentence_idx]
            sentence_lower = sentence.lower()
            
            # Check each pair of entities in this sentence
            for i, entity1 in enumerate(sentence_entities):
                for j, entity2 in enumerate(sentence_entities):
                    if i == j:
                        continue
                    
                    src_type = entity1['stix_type']
                    dst_type = entity2['stix_type']
                    
                    # Look up possible relationships
                    possible_rel_types = self.relations.get((src_type, dst_type), [])
                    
                    # If no predefined relationship, try some common ones
                    if not possible_rel_types:
                        # Special case for domain -> IP
                        if src_type == 'domain-name' and dst_type == 'ipv4-addr':
                            possible_rel_types = ['resolves-to']
                        # Special case for URL -> domain with domain name extraction
                        elif src_type == 'url' and dst_type == 'domain-name':
                            # Extract domain from URL source to see if it matches the target domain
                            domain_match = False
                            src_value = entity1['value'].lower()
                            dst_value = entity2['value'].lower()
                            
                            # Extract domain from URL using regex for various protocols
                            domain_pattern = re.search(r'(?:https?|ftp|sftp)://([^:/]+)', src_value)
                            if domain_pattern and domain_pattern.group(1).lower() == dst_value:
                                domain_match = True
                                
                            # Only add relationship if the domain actually matches
                            if domain_match:
                                possible_rel_types = ['contains']
                            else:
                                possible_rel_types = []
                        # Special case for malware -> infrastructure
                        elif src_type == 'file' and dst_type in ['ipv4-addr', 'domain-name', 'url']:
                            possible_rel_types = ['communicates-with']
                    
                    # Skip if no relationship types found
                    if not possible_rel_types:
                        continue
                    
                    # Look for relationship indicators in sentence
                    selected_rel_type = None
                    
                    # Try to find context clues for specific relationship
                    for rel_type in possible_rel_types:
                        # Use verb-based detection
                        if rel_type in sentence_lower:
                            selected_rel_type = rel_type
                            break
                            
                        # Use other heuristics
                        # "communicates with" / "beacons to" / "connects to"
                        if rel_type == 'communicates-with' and any(x in sentence_lower for x in 
                            ['communicate', 'connection', 'connecting', 'connect', 'beacon', 'c2', 'command and control']):
                            selected_rel_type = 'communicates-with'
                            break
                            
                        # "resolves to"
                        if rel_type == 'resolves-to' and any(x in sentence_lower for x in 
                            ['resolve', 'resolving', 'resolved', 'points to', 'pointing to']):
                            selected_rel_type = 'resolves-to'
                            break
                            
                        # "uses"
                        if rel_type == 'uses' and any(x in sentence_lower for x in 
                            ['use', 'using', 'used', 'utilize', 'deploy', 'leverage']):
                            selected_rel_type = 'uses'
                            break
                            
                        # "targets"
                        if rel_type == 'targets' and any(x in sentence_lower for x in 
                            ['target', 'targeting', 'targeted', 'against', 'victim']):
                            selected_rel_type = 'targets'
                            break
                    
                    # If no specific relationship found, use the first one
                    if not selected_rel_type and possible_rel_types:
                        selected_rel_type = possible_rel_types[0]
                    
                    # Create relationship if found
                    if selected_rel_type:
                        relationship = {
                            'source': entity1['entity'],
                            'target': entity2['entity'],
                            'type': selected_rel_type.upper(),  # Convert to uppercase for our schema
                            'context': sentence
                        }
                        relationships.append(relationship)
        
        # Filter any duplicate relationships
        unique_relationships = []
        seen_relationships = set()
        
        for rel in relationships:
            # Create a key for this relationship
            src_id = rel['source'].get('id', '')
            rel_type = rel['type']
            dst_id = rel['target'].get('id', '')
            
            key = (src_id, rel_type, dst_id)
            
            if key not in seen_relationships:
                seen_relationships.add(key)
                unique_relationships.append(rel)
        
        logger.info(f"Extracted {len(unique_relationships)} relationships from text")
        return unique_relationships