"""
Centralized STIX type mappings used across the codebase.
"""

# Mapping from our entity types to STIX entity types
STIX_TYPE_MAPPING = {
    "Domain": "domain-name",
    "URL": "url",
    "IPv4": "ipv4-addr",
    "IPv6": "ipv6-addr",
    "EmailAddress": "email-addr",
    "Hash": "file",
    "Vulnerability": "vulnerability",
    "FilePath": "file",
    "RegistryKey": "windows-registry-key",
    "ASN": "autonomous-system",
    "CryptocurrencyAddress": "cryptocurrency-wallet",
    "MACAddress": "mac-addr",
    "CIDR": "ipv4-addr",
    "XMPPAddress": "user-account",
    "UserAgent": "user-agent",
    "CreditCard": "payment-card",
    "Location": "location",
    "ATT&CK_Tactic": "attack-pattern",
    "ATT&CK_Technique": "attack-pattern"
}

# Inverse mapping from STIX to our entity types
STIX_TO_OUR_TYPE = {v: k for k, v in STIX_TYPE_MAPPING.items()}

# Define entity type mappings from IOC-Finder to our threat intel knowledge graph
IOC_TO_ENTITY_TYPE_MAPPING = {
    "domains": "Domain",
    "ipv4s": "IPv4",
    "ipv6s": "IPv6",
    "urls": "URL",
    "email_addresses": "EmailAddress",
    "email_addresses_complete": "EmailAddress",
    "md5s": "Hash",
    "sha1s": "Hash",
    "sha256s": "Hash",
    "sha512s": "Hash",
    "ssdeeps": "Hash",
    "imphashes": "Hash",
    "authentihashes": "Hash",
    "cves": "Vulnerability",
    "file_paths": "FilePath",
    "registry_key_paths": "RegistryKey",
    "asns": "ASN",
    "bitcoin_addresses": "CryptocurrencyAddress",
    "monero_addresses": "CryptocurrencyAddress",
    "mac_addresses": "MACAddress",
    "ipv4_cidrs": "CIDR",
    "xmpp_addresses": "XMPPAddress",
    "user_agents": "UserAgent",
    "credit-cards": "CreditCard",
    "nationality": "Location",
    "attack_tactic": "ATT&CK_Tactic",
    "attack_technique": "ATT&CK_Technique"
}

# Map entity types to properties
ENTITY_PROPERTY_MAPPINGS = {
    "Hash": [{"algorithm_key": "algorithm", "value_key": "value"}],
    "Domain": [{"value_key": "value"}],
    "IPv4": [{"value_key": "value"}],
    "IPv6": [{"value_key": "value"}],
    "URL": [{"value_key": "value"}],
    "EmailAddress": [{"value_key": "value"}],
    "Vulnerability": [{"value_key": "value", "id_key": "id"}],
    "FilePath": [{"value_key": "value"}],
    "RegistryKey": [{"value_key": "value"}],
    "ASN": [{"value_key": "value"}],
    "CryptocurrencyAddress": [{"type_key": "currency_type", "value_key": "value"}],
    "MACAddress": [{"value_key": "value"}],
    "CIDR": [{"value_key": "value"}],
    "XMPPAddress": [{"value_key": "value"}],
    "UserAgent": [{"value_key": "value"}],
    "CreditCard": [{"value_key": "value"}],
    "Location": [{"value_key": "name", "type_key": "type"}],
    "ATT&CK_Tactic": [{"id_key": "id", "value_key": "name"}],
    "ATT&CK_Technique": [{"id_key": "id", "value_key": "name"}]
}

# Relationship types mapping
RELATIONSHIP_TYPES = {
    "related_to": "RELATED_TO",
    "resolves_to": "RESOLVES_TO",
    "communicates_with": "COMMUNICATES_WITH",
    "contains": "CONTAINS",
    "affects": "AFFECTS",
    "associated_with": "ASSOCIATED_WITH",
    "has_vulnerability": "HAS_VULNERABILITY",
    "used_by": "USED_BY",
    "located_in": "LOCATED_IN",
    "uses": "USES",
    "targets": "TARGETS",
    "subtechnique_of": "SUBTECHNIQUE_OF"
}

# MITRE ATT&CK Tactics and Techniques mapping
CODE_TACTICS = [
    'TA0043', 'TA0042', 'TA0001', 'TA0002', 'TA0003', 'TA0004', 
    'TA0005', 'TA0006', 'TA0007', 'TA0008', 'TA0009', 'TA0011', 
    'TA0010', 'TA0040'
]

NAME_TACTICS = [
    'Reconnaissance', 'Resource Development', 'Initial Access', 
    'Execution', 'Persistence', 'Privilege Escalation', 
    'Defense Evasion', 'Credential Access', 'Discovery', 
    'Lateral Movement', 'Collection', 'Command and Control', 
    'Exfiltration', 'Impact'
]

# Map tactics to their code
TACTIC_TO_CODE = {name: code for name, code in zip(NAME_TACTICS, CODE_TACTICS)}