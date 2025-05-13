"""
Graph extraction module for parsing security reports into structured knowledge graphs using STIXnet NER.
This module leverages all STIXnet components including:
- IOC-Finder: For extracting cybersecurity observables
- Knowledge Base: For enhancing entity recognition with nationalities
- rcATT: For MITRE ATT&CK tactics and techniques detection
"""

import logging
import json
import re
import uuid
import pandas as pd
import joblib
import nltk
from typing import List, Dict, Any, Optional, Tuple, Set
import sys
import os

# Add STIXnet paths to Python path if needed
stixnet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "STIXnet")
ioc_finder_path = os.path.join(stixnet_path, "Entity-Extraction", "IOC-Finder")
knowledge_base_path = os.path.join(stixnet_path, "Entity-Extraction", "Knowledge-Base")
rcatt_path = os.path.join(stixnet_path, "Entity-Extraction", "rcATT")
sys.path.append(stixnet_path)
sys.path.append(ioc_finder_path)
sys.path.append(knowledge_base_path)
sys.path.append(rcatt_path)

# Import STIXnet IOC finder
from ioc_finder.ioc_finder import find_iocs, prepare_text

# Import STIX relation extractor
from .stix_relation_extractor import STIXRelationExtractor

# Ensure these imports point to the correct location in your project structure
from ..models.data_models import KnowledgeGraph, Node, Relationship, Property
from ..config import DEFAULT_ALLOWED_NODES, DEFAULT_ALLOWED_RELATIONSHIPS
from ..models.stix_mappings import (
    IOC_TO_ENTITY_TYPE_MAPPING as ENTITY_TYPE_MAPPING,
    ENTITY_PROPERTY_MAPPINGS,
    RELATIONSHIP_TYPES,
    CODE_TACTICS,
    NAME_TACTICS,
    TACTIC_TO_CODE
)

logger = logging.getLogger(__name__)

class NERExtractor:
    """
    Extract knowledge graph data from security reports using STIXnet NER.
    Integrates all STIXnet components for comprehensive extraction.
    """

    def __init__(
        self,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        verbose: bool = False
    ):
        """
        Initialize the NERExtractor.
        """
        self.allowed_nodes = allowed_nodes or DEFAULT_ALLOWED_NODES
        self.allowed_relationships = allowed_relationships or DEFAULT_ALLOWED_RELATIONSHIPS
        self.verbose = verbose
        
        # Set logging level for more detailed output if verbose is True
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize the STIX relation extractor
        self.relation_extractor = STIXRelationExtractor()
        
        # Load knowledge base for location extraction
        self._load_knowledge_base()
        
        # Load rcATT models for ATT&CK tactics and techniques detection
        self._load_rcatt_models()
        
        logger.info("Initialized STIXnet NER extractor with all components")
        
        # Check if we have wildcard in allowed nodes/relationships
        self.allow_all_nodes = '*' in self.allowed_nodes
        self.allow_all_relationships = '*' in self.allowed_relationships
        
        # Ensure NLTK resources are downloaded
        self._ensure_nltk_resources()
        
        # Log configuration details in debug mode
        if self.verbose:
            logger.debug(f"Allowed nodes: {self.allowed_nodes}")
            logger.debug(f"Allowed relationships: {self.allowed_relationships}")
            logger.debug(f"Allow all nodes: {self.allow_all_nodes}")
            logger.debug(f"Allow all relationships: {self.allow_all_relationships}")
    
    def _load_knowledge_base(self):
        """Load STIXnet Knowledge Base for enhanced entity extraction."""
        try:
            kb_path = os.path.join(knowledge_base_path, "nationalities.csv")
            self.nationalities_df = pd.read_csv(kb_path)
            logger.info(f"Loaded knowledge base with {len(self.nationalities_df)} nationalities")
        except Exception as e:
            logger.warning(f"Failed to load knowledge base: {e}")
            self.nationalities_df = pd.DataFrame(columns=["Nationality", "Nation"])
    
    def _load_rcatt_models(self):
        """Load rcATT models for ATT&CK detection."""
        try:
            # Path to rcATT models
            rcatt_models_path = os.path.join(rcatt_path, "Models", "configuration.joblib")
            
            # Check if the model file exists
            if os.path.exists(rcatt_models_path):
                self.rcatt_config = joblib.load(rcatt_models_path)
                
                # Load the rcATT models
                tactics_model_path = os.path.join(rcatt_path, "Models", "tactics.joblib")
                techniques_model_path = os.path.join(rcatt_path, "Models", "techniques.joblib")
                
                if os.path.exists(tactics_model_path) and os.path.exists(techniques_model_path):
                    self.tactics_model = joblib.load(tactics_model_path)
                    self.techniques_model = joblib.load(techniques_model_path)
                    self.rcatt_loaded = True
                    logger.info("Loaded rcATT models for ATT&CK detection")
                else:
                    self.rcatt_loaded = False
                    logger.warning("Could not find rcATT model files")
            else:
                self.rcatt_loaded = False
                logger.warning(f"rcATT configuration not found at {rcatt_models_path}")
        except Exception as e:
            self.rcatt_loaded = False
            logger.warning(f"Failed to load rcATT models: {e}")

    def _ensure_nltk_resources(self):
        """Ensure necessary NLTK resources are available."""
        nltk_resources = [('punkt', 'tokenizers/punkt'), ('stopwords', 'corpora/stopwords')]
        for resource_name, resource_path in nltk_resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info(f"Downloading NLTK {resource_name}...")
                nltk.download(resource_name, quiet=True)

    def is_allowed_node_type(self, node_type: str) -> bool:
        """Check if a node type is allowed."""
        if self.allow_all_nodes:
            return True
        return node_type in self.allowed_nodes
    
    def is_allowed_relationship_type(self, rel_type: str) -> bool:
        """Check if a relationship type is allowed."""
        if self.allow_all_relationships:
            return True
        return rel_type in self.allowed_relationships

    def create_node_id(self, entity_type: str, entity_value: str) -> str:
        """Create a node ID from entity type and value."""
        # For most entities, use type and value as identifier
        clean_value = re.sub(r'[^a-zA-Z0-9_-]', '_', entity_value)
        return f"{entity_type.lower()}_{clean_value}"
        
    def _extract_locations(self, text: str) -> List[Dict]:
        """
        Extract locations from text using the STIXnet Knowledge Base.
        Uses nationality information to identify countries and regions.
        
        Args:
            text: The text to extract locations from
        
        Returns:
            A list of location entities with their properties
        """
        locations = []
        
        if self.nationalities_df.empty:
            return locations
            
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # For each nationality in the knowledge base, check if it appears in the text
        for _, row in self.nationalities_df.iterrows():
            try:
                # Handle potential non-string values
                if not isinstance(row['Nationality'], str) or not isinstance(row['Nation'], str):
                    continue
                    
                nationality = row['Nationality'].lower()
                nation = row['Nation']
                
                # Skip very short nationalities to avoid false positives
                if len(nationality) < 4:
                    continue
                    
                # Check if the nationality appears in the text
                if nationality in text_lower:
                    # Create location entity
                    location_id = self.create_node_id("Location", nation)
                    
                    # Add to locations list if not already present
                    if not any(loc['id'] == location_id for loc in locations):
                        locations.append({
                            'id': location_id,
                            'type': 'Location',
                            'name': nation,
                            'entity_type': 'nation'
                        })
            except (TypeError, AttributeError) as e:
                # Skip problematic rows without breaking the whole extraction
                logger.warning(f"Error processing nationality row: {e}")
                continue
        
        return locations
        
    def _extract_attack_entities(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract MITRE ATT&CK tactics and techniques from text using rcATT.
        
        Args:
            text: The text to extract ATT&CK information from
            
        Returns:
            A tuple of (tactics, techniques) lists
        """
        tactics = []
        techniques = []
        
        # Return empty lists if rcATT is not loaded
        if not hasattr(self, 'rcatt_loaded') or not self.rcatt_loaded:
            return tactics, techniques
            
        try:
            # Preprocess the text for rcATT
            processed_text = self._preprocess_text_for_rcatt(text)
            
            if not processed_text:
                return tactics, techniques
                
            # Create a DataFrame with the processed text
            text_df = pd.DataFrame([processed_text], columns=['processed'])
            
            # Predict tactics and get confidence scores
            pred_tactics = self.tactics_model.predict(text_df)
            predprob_tactics = self.tactics_model.decision_function(text_df)
            
            # Predict techniques and get confidence scores
            pred_techniques = self.techniques_model.predict(text_df)
            predprob_techniques = self.techniques_model.decision_function(text_df)
            
            # Apply rcATT post-processing if configured
            if hasattr(self, 'rcatt_config') and isinstance(self.rcatt_config, list) and len(self.rcatt_config) > 0:
                post_processing_type = self.rcatt_config[0]
                
                if post_processing_type == "CP":
                    # Apply confidence propagation
                    pred_techniques, predprob_techniques = self._confidence_propagation(
                        predprob_tactics, pred_techniques, predprob_techniques
                    )
                elif post_processing_type == "HN" and len(self.rcatt_config) > 1:
                    # Apply hanging node
                    c, d = self.rcatt_config[1]
                    pred_techniques = self._hanging_node(
                        pred_tactics, predprob_tactics, 
                        pred_techniques, predprob_techniques,
                        c, d
                    )
            
            # Extract tactics that were detected
            for i, tactic_detected in enumerate(pred_tactics[0]):
                if tactic_detected == 1:
                    tactic_name = NAME_TACTICS[i]
                    tactic_code = CODE_TACTICS[i]
                    confidence = float(predprob_tactics[0][i])
                    
                    # Create tactic entity
                    tactic_id = self.create_node_id("ATT&CK_Tactic", tactic_code)
                    tactics.append({
                        'id': tactic_id,
                        'type': 'ATT&CK_Tactic',
                        'name': tactic_name,
                        'code': tactic_code,
                        'confidence': confidence
                    })
            
            # Extract techniques that were detected
            for i, technique_detected in enumerate(pred_techniques[0]):
                if technique_detected == 1 and i < len(self.techniques_model.classes_):
                    technique_code = self.techniques_model.classes_[i]
                    # Look up technique name (would need mapping from code to name)
                    technique_name = technique_code  # Placeholder, ideally would have name
                    confidence = float(predprob_techniques[0][i])
                    
                    # Create technique entity
                    technique_id = self.create_node_id("ATT&CK_Technique", technique_code)
                    techniques.append({
                        'id': technique_id,
                        'type': 'ATT&CK_Technique',
                        'name': technique_name,
                        'code': technique_code,
                        'confidence': confidence
                    })
            
        except Exception as e:
            logger.warning(f"Error extracting ATT&CK entities: {e}")
            
        return tactics, techniques
        
    def _preprocess_text_for_rcatt(self, text: str) -> str:
        """
        Preprocess text for rcATT models.
        
        Args:
            text: The raw text to process
            
        Returns:
            Processed text ready for rcATT models
        """
        try:
            # Clean text according to rcATT requirements
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http(s)?:\\[0-9a-zA-Z_\.\-\\]+.', 'URL', text)
            # Remove email addresses
            text = re.sub(r'\b([a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)\b', 'email', text)
            # Remove IP addresses
            text = re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', 'IP', text)
            # Remove file paths
            text = re.sub(r'[a-zA-Z]{1}:\\[0-9a-zA-Z_\.\-\\]+', 'file', text)
            # Remove hashes
            text = re.sub(r'\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b', 'hash', text)
            # Remove non-word characters
            text = re.sub(r'\W', ' ', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
        except Exception as e:
            logger.warning(f"Error preprocessing text for rcATT: {e}")
            return ""
            
    def _confidence_propagation(self, predprob_tactics, pred_techniques, predprob_techniques):
        """
        Apply confidence propagation to techniques based on tactics confidence.
        Implementation based on rcATT algorithm.
        
        Args:
            predprob_tactics: Confidence scores for tactics
            pred_techniques: Predicted techniques (0/1)
            predprob_techniques: Confidence scores for techniques
            
        Returns:
            Updated predictions and confidence scores
        """
        # Create a copy to avoid modifying originals
        pred_techniques_corrected = pred_techniques.copy()
        predprob_techniques_corrected = predprob_techniques.copy()
        
        # Create a DataFrame for tactics confidence
        tactics_confidence_df = pd.DataFrame(data=predprob_tactics, columns=CODE_TACTICS)
        
        # For each technique
        for j in range(len(predprob_techniques[0])):
            # For each sample
            for i in range(len(predprob_techniques)):
                # Get initial confidence
                confidence = predprob_techniques[i][j]
                
                # Find related tactics and apply propagation
                for tactic_idx, tactic_code in enumerate(CODE_TACTICS):
                    # Check if this technique is related to this tactic
                    # (In a full implementation, would check TACTICS_TECHNIQUES_RELATIONSHIP_DF)
                    
                    # Apply propagation formula
                    tactic_confidence = predprob_tactics[i][tactic_idx]
                    lambda_val = 1/(np.exp(abs(confidence-tactic_confidence)))
                    predprob_techniques_corrected[i][j] = (
                        predprob_techniques_corrected[i][j] + lambda_val * tactic_confidence
                    )
                
                # Update prediction based on new confidence
                if predprob_techniques_corrected[i][j] >= 0:
                    pred_techniques_corrected[i][j] = 1
                else:
                    pred_techniques_corrected[i][j] = 0
                    
        return pred_techniques_corrected, predprob_techniques_corrected
        
    def _hanging_node(self, pred_tactics, predprob_tactics, pred_techniques, predprob_techniques, c, d):
        """
        Apply hanging node threshold algorithm to techniques predictions.
        Implementation based on rcATT algorithm.
        
        Args:
            pred_tactics: Predicted tactics (0/1)
            predprob_tactics: Confidence scores for tactics
            pred_techniques: Predicted techniques (0/1)
            predprob_techniques: Confidence scores for techniques
            c: Threshold for technique confidence
            d: Threshold for tactic confidence
            
        Returns:
            Updated technique predictions
        """
        # Create a copy to avoid modifying originals
        predprob_techniques_corrected = pred_techniques.copy()
        
        # For each sample
        for i in range(len(pred_techniques)):
            # For each technique
            for j in range(len(pred_techniques[0])):
                # For each tactic
                for k in range(len(pred_tactics[0])):
                    # Check if this technique is related to this tactic
                    # (In a full implementation, would check TACTICS_TECHNIQUES_RELATIONSHIP_DF)
                    
                    # Apply threshold-based correction
                    tech_confidence = predprob_techniques[i][j]
                    tactic_confidence = predprob_tactics[i][k]
                    
                    if 0 < tech_confidence < c and tactic_confidence < d:
                        predprob_techniques_corrected[i][j] = 0
                        
        return predprob_techniques_corrected

    def extract(self, text: str) -> Optional[KnowledgeGraph]:
        """
        Extract knowledge graph data from text using STIXnet NER.
        Integrates all STIXnet components:
        - IOC-Finder: For extracting observables
        - Knowledge Base: For location extraction from nationalities
        - rcATT: For ATT&CK tactics and techniques detection
        
        Args:
            text: The text to extract entities and relationships from
            
        Returns:
            A KnowledgeGraph object containing the extracted entities and relationships
        """
        try:
            logger.info("Starting extraction of entities using STIXnet NER")
            
            # Prepare the text for processing (fang it for IOC extraction)
            processed_text = prepare_text(text)
            
            # Track all nodes and create a mapping from entity value to node
            nodes = []
            node_map = {}  # Maps entity_type:value to node for reuse and relationship creation
            
            # === Step 1: IOC-Finder Entity Extraction ===
            # Use STIXnet IOC finder to extract entities
            iocs, pos_map = find_iocs(processed_text, parse_domain_from_url=True, parse_from_url_path=True)
            
            # Process each entity type from the IOCs
            for entity_type, entities in iocs.items():
                # Skip empty lists and complex nested structures (for now)
                if not entities or isinstance(entities, dict):
                    continue
                
                # Map entity type to a node type using the mapping
                mapped_type = ENTITY_TYPE_MAPPING.get(entity_type)
                if not mapped_type:
                    if self.verbose:
                        logger.debug(f"Skipping unmapped entity type: {entity_type}")
                    continue
                
                # Check if this node type is allowed
                if not self.is_allowed_node_type(mapped_type):
                    if self.verbose:
                        logger.debug(f"Skipping disallowed node type: {mapped_type}")
                    continue
                
                # Process each entity of this type
                for entity in entities:
                    # For hash values, determine the algorithm
                    properties = []
                    
                    # Create a unique ID for the node
                    node_id = self.create_node_id(mapped_type, entity)
                    
                    # Add properties based on entity type
                    if mapped_type == "Hash":
                        # Determine hash algorithm based on length or source
                        hash_algorithm = None
                        if entity_type == "md5s" or len(entity) == 32:
                            hash_algorithm = "MD5"
                        elif entity_type == "sha1s" or len(entity) == 40:
                            hash_algorithm = "SHA-1"
                        elif entity_type == "sha256s" or len(entity) == 64:
                            hash_algorithm = "SHA-256"
                        elif entity_type == "sha512s" or len(entity) == 128:
                            hash_algorithm = "SHA-512"
                        elif entity_type == "imphashes":
                            hash_algorithm = "IMPHASH"
                        elif entity_type == "authentihashes":
                            hash_algorithm = "AUTHENTIHASH"
                        elif entity_type == "ssdeeps":
                            hash_algorithm = "SSDEEP"
                        
                        properties.append(Property(key="algorithm", value=hash_algorithm or "Unknown"))
                        properties.append(Property(key="value", value=entity))
                    elif mapped_type == "Vulnerability" and entity.startswith("CVE-"):
                        # For CVEs, extract the ID and add it as a property
                        properties.append(Property(key="id", value=entity))
                        properties.append(Property(key="value", value=entity))
                    elif mapped_type == "CryptocurrencyAddress":
                        # Add currency type property
                        currency_type = "Bitcoin" if entity_type == "bitcoin_addresses" else "Monero"
                        properties.append(Property(key="currency_type", value=currency_type))
                        properties.append(Property(key="value", value=entity))
                    else:
                        # Default property for other entity types
                        properties.append(Property(key="value", value=entity))
                    
                    # Create the node and add it to our list
                    node = Node(
                        id=node_id,
                        type=mapped_type,
                        properties=properties
                    )
                    
                    # Store in node map for reference when creating relationships
                    node_map[f"{mapped_type}:{entity}"] = node
                    nodes.append(node)
            
            # === Step 2: Knowledge Base Location Extraction ===
            if self.is_allowed_node_type("Location"):
                logger.debug("Extracting locations using STIXnet Knowledge Base")
                location_entities = self._extract_locations(text)
                
                # Add location nodes
                for loc in location_entities:
                    # Create properties for the location
                    properties = [
                        Property(key="name", value=loc['name']),
                        Property(key="type", value=loc['entity_type'])
                    ]
                    
                    # Create the location node
                    node = Node(
                        id=loc['id'],
                        type="Location",
                        properties=properties
                    )
                    
                    # Add to node map and nodes list
                    node_map[f"Location:{loc['name']}"] = node
                    nodes.append(node)
                
                if location_entities:
                    logger.debug(f"Extracted {len(location_entities)} locations")
            
            # === Step 3: rcATT ATT&CK Extraction ===
            if (self.is_allowed_node_type("ATT&CK_Tactic") or 
                self.is_allowed_node_type("ATT&CK_Technique")):
                logger.debug("Extracting ATT&CK tactics and techniques using rcATT")
                
                tactics, techniques = self._extract_attack_entities(text)
                
                # Add tactic nodes
                if self.is_allowed_node_type("ATT&CK_Tactic"):
                    for tactic in tactics:
                        # Create properties for the tactic
                        properties = [
                            Property(key="id", value=tactic['code']),
                            Property(key="name", value=tactic['name'])
                        ]
                        
                        # Create the tactic node
                        node = Node(
                            id=tactic['id'],
                            type="ATT&CK_Tactic",
                            properties=properties
                        )
                        
                        # Add to node map and nodes list
                        node_map[f"ATT&CK_Tactic:{tactic['name']}"] = node
                        nodes.append(node)
                
                # Add technique nodes
                if self.is_allowed_node_type("ATT&CK_Technique"):
                    for technique in techniques:
                        # Create properties for the technique
                        properties = [
                            Property(key="id", value=technique['code']),
                            Property(key="name", value=technique['name'])
                        ]
                        
                        # Create the technique node
                        node = Node(
                            id=technique['id'],
                            type="ATT&CK_Technique",
                            properties=properties
                        )
                        
                        # Add to node map and nodes list
                        node_map[f"ATT&CK_Technique:{technique['code']}"] = node
                        nodes.append(node)
                
                if tactics or techniques:
                    logger.debug(f"Extracted {len(tactics)} tactics and {len(techniques)} techniques")
            
            # === Step 4: Relationship Extraction ===
            relationships = []
            
            # Prepare entity list for relation extraction
            entity_list = []
            for entity_type, entities_values in iocs.items():
                if not entities_values or isinstance(entities_values, dict):
                    continue
                
                mapped_type = ENTITY_TYPE_MAPPING.get(entity_type)
                if not mapped_type or not self.is_allowed_node_type(mapped_type):
                    continue
                
                for entity_value in entities_values:
                    node_id = self.create_node_id(mapped_type, entity_value)
                    if f"{mapped_type}:{entity_value}" in node_map:
                        entity_list.append({
                            'id': node_id,
                            'type': mapped_type,
                            'value': entity_value
                        })
            
            # Extract relationships using STIX relation extractor
            stix_relationships = self.relation_extractor.extract_relationships(entity_list, text)
            
            # Convert STIX relationships to our model's relationships
            for stix_rel in stix_relationships:
                source = stix_rel.get('source')
                target = stix_rel.get('target')
                rel_type = stix_rel.get('type')
                
                if not source or not target or not rel_type:
                    continue
                
                source_id = source.get('id')
                target_id = target.get('id')
                
                # Find the source and target nodes in our node map
                source_node = None
                target_node = None
                
                for node in nodes:
                    if node.id == source_id:
                        source_node = node
                    if node.id == target_id:
                        target_node = node
                    
                    if source_node and target_node:
                        break
                
                if source_node and target_node:
                    # Check if this relationship type is allowed
                    if self.is_allowed_relationship_type(rel_type):
                        rel = Relationship(
                            source=source_node,
                            target=target_node,
                            type=rel_type
                        )
                        relationships.append(rel)
            
            # If no relationships were found with the STIX extractor, fall back to basic relationship extraction
            if not relationships:
                logger.info("No relationships found with STIX extractor, falling back to basic extraction")
                
                # Domain to IP relationships (if both exist)
                domains = iocs.get("domains", [])
                ipv4s = iocs.get("ipv4s", [])
                
                # Add domain-to-ip relationships if we have both
                if domains and ipv4s:
                    for domain in domains:
                        domain_node_key = f"{ENTITY_TYPE_MAPPING['domains']}:{domain}"
                        if domain_node_key in node_map:
                            for ip in ipv4s:
                                ip_node_key = f"{ENTITY_TYPE_MAPPING['ipv4s']}:{ip}"
                                if ip_node_key in node_map:
                                    rel_type = "RESOLVES_TO"
                                    
                                    # Check if this relationship type is allowed
                                    if self.is_allowed_relationship_type(rel_type):
                                        rel = Relationship(
                                            source=node_map[domain_node_key],
                                            target=node_map[ip_node_key],
                                            type=rel_type
                                        )
                                        relationships.append(rel)
                
                # URLs to domains (extraction) - improved with specific domain matching
                urls = iocs.get("urls", [])
                if domains and urls:
                    # Keep track of relationships already created
                    url_domain_pairs = set()
                    
                    for url in urls:
                        # Extract domain from URL - support multiple schemes (http, https, ftp, sftp)
                        domain_match = re.search(r'(?:https?|ftp|sftp)://([^:/]+)', url)
                        if domain_match:
                            extracted_domain = domain_match.group(1)
                            
                            # Only create relationship if this specific domain exists in the domains list
                            if extracted_domain in domains:
                                url_node_key = f"{ENTITY_TYPE_MAPPING['urls']}:{url}"
                                domain_node_key = f"{ENTITY_TYPE_MAPPING['domains']}:{extracted_domain}"
                                
                                # Skip if already processed
                                pair_key = f"{url_node_key}:{domain_node_key}"
                                if pair_key in url_domain_pairs:
                                    continue
                                url_domain_pairs.add(pair_key)
                                
                                if url_node_key in node_map and domain_node_key in node_map:
                                    rel_type = "CONTAINS"
                                    
                                    # Check if this relationship type is allowed
                                    if self.is_allowed_relationship_type(rel_type):
                                        rel = Relationship(
                                            source=node_map[url_node_key],
                                            target=node_map[domain_node_key],
                                            type=rel_type
                                        )
                                        relationships.append(rel)
            
            # Add technique-to-tactic relationships
            if self.is_allowed_relationship_type("SUBTECHNIQUE_OF") and self.is_allowed_node_type("ATT&CK_Technique") and self.is_allowed_node_type("ATT&CK_Tactic"):
                # Get all tactics and techniques
                tactic_nodes = [n for n in nodes if n.type == "ATT&CK_Tactic"]
                technique_nodes = [n for n in nodes if n.type == "ATT&CK_Technique"]
                
                # For each technique, find its related tactics and add relationships
                for technique_node in technique_nodes:
                    # Extract technique code from properties
                    technique_code = None
                    for prop in technique_node.properties:
                        if prop.key == "id":
                            technique_code = prop.value
                            break
                    
                    if not technique_code:
                        continue
                    
                    # For each tactic, check if this technique belongs to it
                    for tactic_node in tactic_nodes:
                        # Extract tactic code from properties
                        tactic_code = None
                        for prop in tactic_node.properties:
                            if prop.key == "id":
                                tactic_code = prop.value
                                break
                        
                        if not tactic_code:
                            continue
                        
                        # Check if this technique belongs to this tactic
                        # In a full implementation, would check TACTICS_TECHNIQUES_RELATIONSHIP_DF
                        # For now, add a basic relationship
                        rel_type = "SUBTECHNIQUE_OF"
                        rel = Relationship(
                            source=technique_node,
                            target=tactic_node,
                            type=rel_type
                        )
                        relationships.append(rel)
            
            # Create the knowledge graph from our nodes and relationships
            knowledge_graph = KnowledgeGraph(
                nodes=nodes,
                rels=relationships
            )
            
            logger.info(f"Extracted {len(nodes)} entities and {len(relationships)} relationships")
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}", exc_info=True)
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