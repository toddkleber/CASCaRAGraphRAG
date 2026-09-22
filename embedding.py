import os
import re
import json
import html
import hashlib
from collections import Counter

import ollama
from dotenv import load_dotenv
from neo4j import GraphDatabase


# ============================================================
# Environment
# ============================================================

load_dotenv()


NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "bolt://localhost:7687"
)

NEO4J_USERNAME = os.getenv(
    "NEO4J_USERNAME",
    "neo4j"
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD"
)

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j"
)

EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "qwen3-embedding:0.6b"
)

VECTOR_INDEX_NAME = os.getenv(
    "KNOWLEDGE_VECTOR_INDEX_NAME",
    "cascara_knowledge_embeddings"
)

VECTOR_DIMENSIONS = int(
    os.getenv(
        "KNOWLEDGE_VECTOR_DIMENSIONS",
        "1024"
    )
)

PREFERRED_LANGUAGE = os.getenv(
    "PREFERRED_LANGUAGE",
    "en"
)

EMBED_BATCH_SIZE = int(
    os.getenv(
        "EMBED_BATCH_SIZE",
        "16"
    )
)

PREVIEW_LIMIT = int(
    os.getenv(
        "EMBED_PREVIEW_LIMIT",
        "30"
    )
)


# ============================================================
# Validation
# ============================================================

if not NEO4J_PASSWORD:
    raise ValueError(
        "NEO4J_PASSWORD is missing from .env"
    )


if not re.fullmatch(
    r"[A-Za-z_][A-Za-z0-9_]*",
    VECTOR_INDEX_NAME
):
    raise ValueError(
        "Invalid KNOWLEDGE_VECTOR_INDEX_NAME."
    )


# ============================================================
# Neo4j
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)


# ============================================================
# Indexing rules
# ============================================================

USEFUL_ITEM_LABELS = {
    "anEntity",
    "aRelationship",
    "Entity",
    "Relationship",
    "Property",
    "Enumeration",
    "EnumerationValue",
    "Link",
}


# Structural CASCaRA definitions are useful when traversing
# facts, but they do not need standalone semantic documents
# for this GraphRAG prototype.
STRUCTURAL_ITEM_IDS = {
    "cas:View",
    "cas:Outline",
    "cas:Root",
    "cas:Package",
}


STRUCTURAL_ITEM_CLASSES = {
    "cas:View",
    "cas:Outline",
    "cas:Root",
    "cas:Package",
}

# Internal link definitions used to implement reified
# relationship endpoints.
INTERNAL_LINK_IDS = {
    "cas:linksSource",
    "cas:linksTarget",
}


SCHEMA_RELATIONSHIP_TYPES = {
    "SPECIALIZES",
    "HAS_PROPERTY",

    "HAS_ENUMERATED_PROPERTY",
    "HAS_ENUMERATED_VALUE",
    "HAS_ENUMERATED_ENDPOINT",

    "HAS_ENUMERATED_SOURCE_LINK",
    "HAS_ENUMERATED_TARGET_LINK",
}


STRUCTURAL_RELATIONSHIP_TYPES = {
    "CONTAINS",
    "CAS_LISTS",
    "CAS_SHOWS",
}


# ============================================================
# Text utilities
# ============================================================

def clean_text(value):
    """
    Normalize CASCaRA text for embeddings.
    """

    if value is None:
        return ""

    value = str(value)

    value = html.unescape(
        value
    )

    # CASCaRA/wiki links:
    #
    # [[FMC:Actor]]
    # becomes:
    # FMC:Actor

    value = re.sub(
        r"\[\[([^\]]+)\]\]",
        r"\1",
        value
    )

    # Remove HTML.
    value = re.sub(
        r"<[^>]+>",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# Multilingual CASCaRA JSON
# ============================================================

def localized_value(
    json_value,
    fallback=None
):
    """
    Prefer the configured language from titleJson,
    descriptionJson, definitionJson, etc.

    Handles several reasonable JSON representations so the
    prototype is not dependent on one exact serialization.
    """

    if json_value:

        try:

            if isinstance(
                json_value,
                str
            ):
                parsed = json.loads(
                    json_value
                )
            else:
                parsed = json_value

            result = find_localized_text(
                parsed,
                PREFERRED_LANGUAGE
            )

            if result:
                return clean_text(
                    result
                )

        except (
            json.JSONDecodeError,
            TypeError
        ):
            pass

    return clean_text(
        fallback
    )


def find_localized_text(
    value,
    language
):
    """
    Search a parsed JSON structure for a localized string.
    """

    if isinstance(
        value,
        str
    ):
        return value

    if isinstance(
        value,
        list
    ):

        # First pass:
        # exact preferred language.

        for entry in value:

            if not isinstance(
                entry,
                dict
            ):
                continue

            entry_language = (
                entry.get("lang")
                or entry.get("language")
                or entry.get("@language")
            )

            if (
                entry_language
                and entry_language.lower()
                == language.lower()
            ):

                text = (
                    entry.get("value")
                    or entry.get("text")
                    or entry.get("label")
                    or entry.get("@value")
                )

                if text:
                    return str(
                        text
                    )

        # Second pass:
        # language-neutral entry.

        for entry in value:

            if not isinstance(
                entry,
                dict
            ):
                continue

            entry_language = (
                entry.get("lang")
                or entry.get("language")
                or entry.get("@language")
            )

            text = (
                entry.get("value")
                or entry.get("text")
                or entry.get("label")
                or entry.get("@value")
            )

            if (
                not entry_language
                and text
            ):
                return str(
                    text
                )

        # Third pass:
        # first usable entry.

        for entry in value:

            result = find_localized_text(
                entry,
                language
            )

            if result:
                return result

    if isinstance(
        value,
        dict
    ):

        # Some structures may look like:
        #
        # {
        #   "en": "...",
        #   "de": "..."
        # }

        if language in value:

            if isinstance(
                value[language],
                str
            ):
                return value[
                    language
                ]

        text = (
            value.get("value")
            or value.get("text")
            or value.get("label")
            or value.get("@value")
        )

        if text:

            entry_language = (
                value.get("lang")
                or value.get("language")
                or value.get("@language")
            )

            if (
                not entry_language
                or entry_language.lower()
                == language.lower()
            ):
                return str(
                    text
                )

        for nested in value.values():

            result = find_localized_text(
                nested,
                language
            )

            if result:
                return result

    return None


# ============================================================
# Human readable relationship names
# ============================================================

RELATIONSHIP_PHRASES = {
    "SPECIALIZES":
        "specializes",

    "HAS_PROPERTY":
        "has property",

    "HAS_ENUMERATED_PROPERTY":
        "has enumerated property",

    "HAS_ENUMERATED_VALUE":
        "has enumerated value",

    "HAS_ENUMERATED_ENDPOINT":
        "has enumerated endpoint",

    "HAS_ENUMERATED_SOURCE_LINK":
        "has enumerated source link",

    "HAS_ENUMERATED_TARGET_LINK":
        "has enumerated target link",

    "CONTAINS":
        "contains",

    "CAS_LISTS":
        "lists",

    "CAS_SHOWS":
        "shows",
}


def relationship_phrase(
    relationship_type
):

    if relationship_type in RELATIONSHIP_PHRASES:

        return RELATIONSHIP_PHRASES[
            relationship_type
        ]

    return (
        relationship_type
        .replace("_", " ")
        .lower()
    )


# ============================================================
# Get CASCaRA items
# ============================================================

def get_cascara_items():

    records, _, _ = driver.execute_query(
        """
        MATCH (item:CascaraItem)

        RETURN
            item.id AS id,

            item.title AS title,
            item.titleJson AS title_json,

            item.description AS description,
            item.descriptionJson AS description_json,

            item.definition AS definition,
            item.definitionJson AS definition_json,

            item.hasClass AS class,
            item.itemType AS item_type,

            item.datatype AS datatype,

            item.minCount AS min_count,
            item.maxCount AS max_count,
            item.maxLength AS max_length,

            labels(item) AS labels

        ORDER BY item.id
        """,

        database_=NEO4J_DATABASE
    )

    return [
        dict(record)
        for record in records
    ]


# ============================================================
# Get endpoints of first-class aRelationships
# ============================================================

def get_relationship_endpoints():
    """
    Return logical source and target IDs for each
    aRelationship.

    We use these to enrich the aRelationship's embedding.

    The physical endpoint edges themselves will NOT become
    separate semantic fact documents.
    """

    records, _, _ = driver.execute_query(
        """
        MATCH
            (rel:aRelationship)
            -[sourceLink]->
            (source:CascaraItem),

            (rel)
            -[targetLink]->
            (target:CascaraItem)

        WHERE
            sourceLink.linkDirection =
                'hasSourceLink'

          AND

            targetLink.linkDirection =
                'hasTargetLink'

        RETURN
            rel.id AS relationship_id,

            collect(
                DISTINCT source.id
            ) AS source_ids,

            collect(
                DISTINCT target.id
            ) AS target_ids
        """,

        database_=NEO4J_DATABASE
    )

    result = {}

    for record in records:

        result[
            record[
                "relationship_id"
            ]
        ] = {
            "source_ids":
                record["source_ids"],

            "target_ids":
                record["target_ids"],
        }

    return result


# ============================================================
# Retrieve native CASCaRA Neo4j facts
# ============================================================

def get_native_facts():
    """
    Retrieve meaningful CASCaRA-managed Neo4j relationships.

    IMPORTANT:

    We exclude ONLY the physical source/target endpoint edges
    originating from a first-class aRelationship node.

    Example excluded physical representation:

        (aRelationship)
            -[hasSourceLink]->
        (source)

        (aRelationship)
            -[hasTargetLink]->
        (target)

    The aRelationship itself is already indexed as a semantic
    relationship statement.

    We KEEP relationships such as:

        CAS_LISTS
        CAS_SHOWS
        SPECIF_PRIORITY
        SPECIALIZES
        HAS_PROPERTY
        HAS_ENUMERATED_VALUE
        CONTAINS

    even if they use linkDirection = hasTargetLink.
    """

    records, _, _ = driver.execute_query(
        """
        MATCH
            (source:CascaraItem)
            -[rel]->
            (target:CascaraItem)

        WHERE

            // ------------------------------------------------
            // Exclude only endpoint plumbing belonging to an
            // actual first-class aRelationship.
            // ------------------------------------------------

            NOT (
                source:aRelationship

                AND

                coalesce(
                    rel.linkDirection,
                    ''
                )
                IN [
                    'hasSourceLink',
                    'hasTargetLink'
                ]
            )

            // ------------------------------------------------
            // Never include our own RAG support relationships.
            // ------------------------------------------------

            AND NOT type(rel)
                STARTS WITH 'RAG_'

            AND type(rel) <> 'DESCRIBES'

            // ------------------------------------------------
            // Only CASCaRA-managed / known CASCaRA relations.
            // ------------------------------------------------

            AND (
                coalesce(
                    rel.cascaraManaged,
                    false
                ) = true

                OR

                type(rel) IN [
                    'CONTAINS',
                    'CAS_LISTS',
                    'CAS_SHOWS',
                    'SPECIALIZES',
                    'HAS_PROPERTY',
                    'HAS_ENUMERATED_PROPERTY',
                    'HAS_ENUMERATED_VALUE',
                    'HAS_ENUMERATED_ENDPOINT',
                    'HAS_ENUMERATED_SOURCE_LINK',
                    'HAS_ENUMERATED_TARGET_LINK'
                ]
            )

        RETURN

            source.id AS source_id,
            source.title AS source_title,
            source.titleJson AS source_title_json,
            source.description AS source_description,
            source.hasClass AS source_class,
            labels(source) AS source_labels,

            type(rel) AS relationship_type,
            properties(rel) AS relationship_properties,

            target.id AS target_id,
            target.title AS target_title,
            target.titleJson AS target_title_json,
            target.description AS target_description,
            target.hasClass AS target_class,
            labels(target) AS target_labels

        ORDER BY
            source_id,
            relationship_type,
            target_id
        """,

        database_=NEO4J_DATABASE
    )

    return [
        dict(record)
        for record in records
    ]


# ============================================================
# Item filtering
# ============================================================

def should_index_item(item):
    """
    Decide whether this CascaraItem should have its own
    standalone semantic document.

    Important:
    Even when an item is excluded here, relationships involving
    that item can still become fact documents.
    """

    labels = set(
        item["labels"] or []
    )

    item_id = (
        item["id"] or ""
    )

    # --------------------------------------------------------
    # Only semantic CASCaRA item categories
    # --------------------------------------------------------

    if not (
        labels
        & USEFUL_ITEM_LABELS
    ):
        return False

    # --------------------------------------------------------
    # Actual package instances
    # --------------------------------------------------------

    if (
        "Package" in labels
        or "aPackage" in labels
    ):
        return False

    # --------------------------------------------------------
    # Structural schema definitions
    # --------------------------------------------------------

    if item_id in STRUCTURAL_ITEM_IDS:
        return False


    # --------------------------------------------------------
    # Structural model instances
    #
    # Examples:
    #
    # d:Folder-Requirements
    #     hasClass = cas:Outline
    #
    # d:Diagram-...
    #     hasClass = cas:View
    # --------------------------------------------------------

    if (
        item.get("class")
        in STRUCTURAL_ITEM_CLASSES
    ):
        return False
    # --------------------------------------------------------
    # Internal source / target link definitions
    # --------------------------------------------------------

    if item_id in INTERNAL_LINK_IDS:
        return False

    # --------------------------------------------------------
    # Generated endpoint definitions such as:
    #
    # SpecIF:reads-toSource
    # SpecIF:reads-toTarget
    # oslc_rm:satisfies-toSource
    #
    # These describe implementation mechanics rather than an
    # independent piece of semantic knowledge.
    # --------------------------------------------------------

    if "Link" in labels:

        lowered_id = (
            item_id.lower()
        )

        title = localized_value(
            item["title_json"],
            item["title"]
        )

        lowered_title = (
            title.lower()
            if title
            else ""
        )

        endpoint_patterns = [
            "-tosource",
            "-totarget",
            "to source",
            "to target",
        ]

        for pattern in endpoint_patterns:

            if (
                pattern in lowered_id
                or pattern in lowered_title
            ):
                return False

    return True


# ============================================================
# Item names
# ============================================================

def build_item_lookup(
    items
):

    result = {}

    for item in items:

        title = localized_value(
            item["title_json"],
            item["title"]
        )

        description = localized_value(
            item["description_json"],
            item["description"]
        )

        result[
            item["id"]
        ] = {
            **item,

            "clean_title":
                title,

            "clean_description":
                description,

            "clean_definition":
                localized_value(
                    item[
                        "definition_json"
                    ],
                    item[
                        "definition"
                    ]
                )
        }

    return result


def display_name(
    item
):

    if not item:
        return "Unknown item"

    return (
        item.get(
            "clean_title"
        )
        or item.get(
            "clean_description"
        )
        or item.get("id")
        or "Unknown item"
    )


# ============================================================
# Item category
# ============================================================

def item_category(
    item
):

    labels = set(
        item["labels"] or []
    )

    if "aRelationship" in labels:
        return "domain_statement"

    if "anEntity" in labels:
        return "model_item"

    if (
        "Entity" in labels
        or "Relationship" in labels
        or "Property" in labels
        or "Enumeration" in labels
        or "EnumerationValue" in labels
        or "Link" in labels
    ):
        return "schema_item"

    return "item"


# ============================================================
# Build item semantic document
# ============================================================

def build_item_document(
    item,
    endpoint_map,
    item_lookup
):

    labels = set(
        item["labels"] or []
    )

    title = item[
        "clean_title"
    ]

    description = item[
        "clean_description"
    ]

    definition = item[
        "clean_definition"
    ]

    category = item_category(
        item
    )

    lines = []

    lines.append(
        "CASCaRA knowledge item"
    )

    lines.append(
        f"Knowledge category: "
        f"{category}"
    )

    lines.append(
        f"CASCaRA ID: "
        f"{item['id']}"
    )

    if title:

        lines.append(
            f"Title: {title}"
        )

    if item["class"]:

        lines.append(
            f"Class: "
            f"{item['class']}"
        )

    if item["item_type"]:

        lines.append(
            f"Item type: "
            f"{item['item_type']}"
        )

    if labels:

        lines.append(
            "CASCaRA labels: "
            + ", ".join(
                sorted(labels)
            )
        )

    if description:

        lines.append(
            f"Description: "
            f"{description}"
        )

    if definition:

        lines.append(
            f"Definition: "
            f"{definition}"
        )

    # ========================================================
    # Relationship statement enrichment
    # ========================================================

    source_ids = []
    target_ids = []

    if "aRelationship" in labels:

        endpoints = endpoint_map.get(
            item["id"],
            {}
        )

        source_ids = endpoints.get(
            "source_ids",
            []
        )

        target_ids = endpoints.get(
            "target_ids",
            []
        )

        source_names = [
            display_name(
                item_lookup.get(
                    source_id
                )
            )
            for source_id in source_ids
        ]

        target_names = [
            display_name(
                item_lookup.get(
                    target_id
                )
            )
            for target_id in target_ids
        ]

        if (
            source_names
            and target_names
        ):

            relationship_name = (
                item["class"]
                or title
                or "relationship"
            )

            lines.append(
                "Relationship statement:"
            )

            lines.append(
                f"{', '.join(source_names)} "
                f"-- {relationship_name} --> "
                f"{', '.join(target_names)}"
            )

            lines.append(
                "Source: "
                + ", ".join(
                    source_names
                )
            )

            lines.append(
                "Target: "
                + ", ".join(
                    target_names
                )
            )

    # ========================================================
    # Property metadata
    # ========================================================

    if "Property" in labels:

        if item["datatype"]:

            lines.append(
                f"Datatype: "
                f"{item['datatype']}"
            )

        if (
            item["min_count"]
            is not None
        ):

            lines.append(
                f"Minimum count: "
                f"{item['min_count']}"
            )

        if (
            item["max_count"]
            is not None
        ):

            lines.append(
                f"Maximum count: "
                f"{item['max_count']}"
            )

        if (
            item["max_length"]
            is not None
        ):

            lines.append(
                f"Maximum length: "
                f"{item['max_length']}"
            )

    return {
        "id":
            "item::"
            + item["id"],

        "kind":
            "item",

        "category":
            category,

        "title":
            title
            or item["id"],

        "cascara_id":
            item["id"],

        "relationship_type":
            item["class"]
            if "aRelationship" in labels
            else None,

        "source_ids":
            source_ids,

        "target_ids":
            target_ids,

        "text":
            "\n".join(
                lines
            ),
    }


# ============================================================
# Fact classification
# ============================================================

def classify_fact(
    relationship_type
):

    if (
        relationship_type
        in STRUCTURAL_RELATIONSHIP_TYPES
    ):
        return "structure"

    if (
        relationship_type
        in SCHEMA_RELATIONSHIP_TYPES
    ):
        return "schema"

    # Relationships such as SPECIF_PRIORITY are commonly
    # semantic/property assignments generated from CASCaRA
    # model data.

    return "property_or_semantic"


# ============================================================
# Relationship property text
# ============================================================

def useful_relationship_properties(
    properties
):
    """
    Keep only domain-level relationship properties.

    CASCaRA exporter bookkeeping should not be included
    in embedding text.
    """

    if not properties:
        return {}

    ignored = {
        "cascaraKey",
        "cascaraManaged",
        "linkItemType",
        "linkDirection",
        "linkClass",
    }

    return {
        key: value
        for key, value
        in properties.items()
        if key not in ignored
    }


def stringify_property_value(
    value
):

    if isinstance(
        value,
        str
    ):
        return clean_text(
            value
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# Stable fact ID
# ============================================================

def make_fact_id(
    source_id,
    relationship_type,
    target_id,
    properties
):

    normalized_properties = json.dumps(
        properties,
        sort_keys=True,
        default=str,
        ensure_ascii=False
    )

    raw = (
        f"{source_id}|"
        f"{relationship_type}|"
        f"{target_id}|"
        f"{normalized_properties}"
    )

    digest = hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:24]

    return (
        f"fact::{digest}"
    )


def build_fact_document(
    fact,
    item_lookup
):
    """
    Build a semantic document for a CASCaRA graph fact.

    This function is schema-driven.

    It does NOT depend on domain-specific relationship names
    such as:

        SPECIF_PRIORITY
        OSLC_RM_...
        FMC_...

    When the Neo4j relationship represents a CASCaRA link
    instance, its linkClass is resolved against the actual
    CascaraItem describing that Link.

    Example:

        relationship type:
            SPECIF_PRIORITY

        relationship metadata:
            linkClass = SpecIF:Priority

        CascaraItem:
            id = SpecIF:Priority
            title = Priority

    Semantic result:

        Data Volume -- Priority --> high
    """

    # ========================================================
    # Resolve source / target items
    # ========================================================

    source = item_lookup.get(
        fact["source_id"]
    )

    target = item_lookup.get(
        fact["target_id"]
    )

    source_name = display_name(
        source
    )

    target_name = display_name(
        target
    )


    # ========================================================
    # Raw Neo4j relationship information
    # ========================================================

    relationship_type = (
        fact["relationship_type"]
    )

    raw_relationship_properties = (
        fact.get(
            "relationship_properties"
        )
        or {}
    )


    # ========================================================
    # Resolve CASCaRA Link class
    #
    # Example:
    #
    # rel.linkClass = "SpecIF:Priority"
    #
    # item_lookup["SpecIF:Priority"]
    #     -> CascaraItem / Link
    # ========================================================

    link_class = (
        raw_relationship_properties.get(
            "linkClass"
        )
    )

    link_item = None

    if link_class:

        link_item = item_lookup.get(
            link_class
        )


    # ========================================================
    # Determine human-readable relationship name
    # ========================================================

    if link_item:

        relationship_name = (
            display_name(
                link_item
            )
        )

    elif link_class:

        # Link class exists but its schema definition is not
        # present in this particular package.
        relationship_name = (
            link_class
        )

    else:

        # Native CASCaRA metamodel/export relationship such as:
        #
        # SPECIALIZES
        # HAS_PROPERTY
        # HAS_ENUMERATED_VALUE
        # CONTAINS
        #
        # Use our generic CASCaRA phrase mapping.
        relationship_name = (
            relationship_phrase(
                relationship_type
            )
        )


    # ========================================================
    # Determine semantic category
    # ========================================================

    if (
        relationship_type
        in STRUCTURAL_RELATIONSHIP_TYPES
    ):

        category = "structure"

    elif (
        relationship_type
        in SCHEMA_RELATIONSHIP_TYPES
    ):

        category = "schema"

    elif link_class:

        # Generic CASCaRA link instance.
        #
        # The actual class can be anything supplied by the
        # package.
        category = "link_instance"

    else:

        category = (
            classify_fact(
                relationship_type
            )
        )


    # ========================================================
    # Remove exporter bookkeeping from embedding text
    # ========================================================

    semantic_properties = (
        useful_relationship_properties(
            raw_relationship_properties
        )
    )


    # ========================================================
    # Build embedding text
    # ========================================================

    lines = []

    lines.append(
        "CASCaRA graph fact"
    )

    lines.append(
        f"Fact category: "
        f"{category}"
    )


    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    lines.append(
        f"Source: "
        f"{source_name}"
    )

    if source:

        if source.get("class"):

            lines.append(
                f"Source class: "
                f"{source['class']}"
            )

        if source.get("item_type"):

            lines.append(
                f"Source item type: "
                f"{source['item_type']}"
            )


    # --------------------------------------------------------
    # Relationship
    # --------------------------------------------------------

    lines.append(
        f"Relationship: "
        f"{relationship_name}"
    )

    if link_class:

        lines.append(
            f"Relationship class: "
            f"{link_class}"
        )

    else:

        lines.append(
            f"Relationship type: "
            f"{relationship_type}"
        )


    # --------------------------------------------------------
    # Include the CASCaRA Link definition when available.
    #
    # This improves semantic search without hard-coding any
    # particular ontology.
    # --------------------------------------------------------

    if link_item:

        link_description = (
            link_item.get(
                "clean_description"
            )
        )

        link_definition = (
            link_item.get(
                "clean_definition"
            )
        )

        if link_description:

            lines.append(
                f"Relationship description: "
                f"{link_description}"
            )

        if link_definition:

            lines.append(
                f"Relationship definition: "
                f"{link_definition}"
            )


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    lines.append(
        f"Target: "
        f"{target_name}"
    )

    if target:

        if target.get("class"):

            lines.append(
                f"Target class: "
                f"{target['class']}"
            )

        if target.get("item_type"):

            lines.append(
                f"Target item type: "
                f"{target['item_type']}"
            )


    # --------------------------------------------------------
    # Human-readable statement
    # --------------------------------------------------------

    lines.append(
        "Statement: "
        f"{source_name} "
        f"-- {relationship_name} --> "
        f"{target_name}"
    )


    # --------------------------------------------------------
    # Any genuine domain properties stored on the relation
    # --------------------------------------------------------

    if semantic_properties:

        lines.append(
            "Relationship properties:"
        )

        for key, value in sorted(
            semantic_properties.items()
        ):

            lines.append(
                f"{key}: "
                f"{stringify_property_value(value)}"
            )


    # ========================================================
    # Build stable semantic document
    # ========================================================

    return {
        "id":
            make_fact_id(
                fact["source_id"],
                relationship_type,
                fact["target_id"],
                semantic_properties
            ),

        "kind":
            "fact",

        "category":
            category,

        "title":
            (
                f"{source_name} "
                f"-- {relationship_name} --> "
                f"{target_name}"
            ),

        "cascara_id":
            None,

        # Keep the Neo4j relationship type for traceability.
        "relationship_type":
            relationship_type,

        # Also keep the real CASCaRA Link class separately.
        "relationship_class":
            link_class,

        "source_ids": [
            fact["source_id"]
        ],

        "target_ids": [
            fact["target_id"]
        ],

        "text":
            "\n".join(
                lines
            ),
    }


# ============================================================
# Ollama embeddings
# ============================================================

def embed_documents(
    documents
):
    """
    Embed in batches rather than one network call per item.
    """

    total = len(
        documents
    )

    for start in range(
        0,
        total,
        EMBED_BATCH_SIZE
    ):

        batch = documents[
            start:
            start + EMBED_BATCH_SIZE
        ]

        texts = [
            document["text"]
            for document in batch
        ]

        response = ollama.embed(
            model=EMBED_MODEL,
            input=texts
        )

        embeddings = (
            response.embeddings
        )

        if (
            len(embeddings)
            != len(batch)
        ):

            raise RuntimeError(
                "Ollama returned the wrong "
                "number of embeddings."
            )

        for document, embedding in zip(
            batch,
            embeddings
        ):

            if (
                len(embedding)
                != VECTOR_DIMENSIONS
            ):

                raise ValueError(
                    "Unexpected embedding "
                    f"dimension "
                    f"{len(embedding)} "
                    f"for {document['id']}"
                )

            document[
                "embedding"
            ] = embedding

        finished = min(
            start + len(batch),
            total
        )

        print(
            f"Embedded "
            f"{finished}/{total}"
        )


# ============================================================
# Rebuild only the v2 semantic layer
# ============================================================

def clear_v2_documents():
    """
    Does NOT touch:
        CascaraItem
        RagDocument
        cascara_entity_embeddings

    Only deletes v0.2 RagKnowledge nodes.
    """

    driver.execute_query(
        """
        MATCH (doc:RagKnowledge)

        DETACH DELETE doc
        """,

        database_=NEO4J_DATABASE
    )


# ============================================================
# Store RagKnowledge documents
# ============================================================

def save_documents(
    documents
):

    rows = []

    for document in documents:

        rows.append(
            {
                "id":
                    document["id"],

                "kind":
                    document["kind"],

                "category":
                    document["category"],

                "title":
                    document["title"],

                "cascara_id":
                    document["cascara_id"],

                "relationship_type":
                    document[
                        "relationship_type"
                    ],

                "relationship_class":
                    document.get(
                        "relationship_class"
                    ),

                "text":
                    document["text"],

                "embedding":
                    document[
                        "embedding"
                    ],
            }
        )

    driver.execute_query(
        """
        UNWIND $rows AS row

        MERGE (
            doc:RagKnowledge {
                id: row.id
            }
        )

        SET
            doc.kind =
                row.kind,

            doc.category =
                row.category,

            doc.title =
                row.title,

            doc.cascaraId =
                row.cascara_id,

            doc.relationshipType =
                row.relationship_type,
            
            doc.relationshipClass =
                row.relationship_class,

            doc.text =
                row.text,

            doc.embedding =
                row.embedding,

            doc.embeddingModel =
                $embedding_model,

            doc.embeddingSchemaVersion =
                2
        """,

        rows=rows,

        embedding_model=EMBED_MODEL,

        database_=NEO4J_DATABASE
    )


# ============================================================
# Connect item documents to CASCaRA items
# ============================================================

def create_item_links(
    documents
):

    item_rows = [
        {
            "doc_id":
                document["id"],

            "cascara_id":
                document["cascara_id"],
        }

        for document in documents

        if (
            document["kind"]
            == "item"

            and

            document["cascara_id"]
        )
    ]

    if item_rows:

        driver.execute_query(
            """
            UNWIND $rows AS row

            MATCH (
                doc:RagKnowledge {
                    id: row.doc_id
                }
            )

            MATCH (
                item:CascaraItem {
                    id: row.cascara_id
                }
            )

            MERGE
                (doc)
                -[:RAG_DESCRIBES]->
                (item)
            """,

            rows=item_rows,

            database_=NEO4J_DATABASE
        )


# ============================================================
# Connect semantic documents to source / target nodes
# ============================================================

def create_endpoint_links(
    documents
):

    endpoint_rows = []

    for document in documents:

        if (
            not document[
                "source_ids"
            ]
            and
            not document[
                "target_ids"
            ]
        ):
            continue

        endpoint_rows.append(
            {
                "doc_id":
                    document["id"],

                "source_ids":
                    document[
                        "source_ids"
                    ],

                "target_ids":
                    document[
                        "target_ids"
                    ],
            }
        )

    if not endpoint_rows:
        return

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    driver.execute_query(
        """
        UNWIND $rows AS row

        MATCH (
            doc:RagKnowledge {
                id: row.doc_id
            }
        )

        UNWIND
            row.source_ids
            AS source_id

        MATCH (
            source:CascaraItem {
                id: source_id
            }
        )

        MERGE
            (doc)
            -[:RAG_SOURCE]->
            (source)
        """,

        rows=endpoint_rows,

        database_=NEO4J_DATABASE
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    driver.execute_query(
        """
        UNWIND $rows AS row

        MATCH (
            doc:RagKnowledge {
                id: row.doc_id
            }
        )

        UNWIND
            row.target_ids
            AS target_id

        MATCH (
            target:CascaraItem {
                id: target_id
            }
        )

        MERGE
            (doc)
            -[:RAG_TARGET]->
            (target)
        """,

        rows=endpoint_rows,

        database_=NEO4J_DATABASE
    )


# ============================================================
# Constraint
# ============================================================

def create_constraint():

    driver.execute_query(
        """
        CREATE CONSTRAINT
            rag_knowledge_id
        IF NOT EXISTS

        FOR (doc:RagKnowledge)

        REQUIRE doc.id IS UNIQUE
        """,

        database_=NEO4J_DATABASE
    )


# ============================================================
# Vector index
# ============================================================

def create_vector_index():

    query = f"""
    CREATE VECTOR INDEX
        {VECTOR_INDEX_NAME}
    IF NOT EXISTS

    FOR (doc:RagKnowledge)

    ON doc.embedding

    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`:
                {VECTOR_DIMENSIONS},

            `vector.similarity_function`:
                'cosine'
        }}
    }}
    """

    driver.execute_query(
        query,
        database_=NEO4J_DATABASE
    )


# ============================================================
# Index status
# ============================================================

def show_vector_index():

    records, _, _ = driver.execute_query(
        """
        SHOW VECTOR INDEXES

        YIELD
            name,
            state,
            populationPercent,
            labelsOrTypes,
            properties

        WHERE name = $index_name

        RETURN
            name,
            state,
            populationPercent,
            labelsOrTypes,
            properties
        """,

        index_name=VECTOR_INDEX_NAME,

        database_=NEO4J_DATABASE
    )

    print()
    print(
        "=== V2 VECTOR INDEX ==="
    )

    if not records:

        print(
            "Index not found."
        )

        return

    for record in records:

        print(
            "Name:",
            record["name"]
        )

        print(
            "State:",
            record["state"]
        )

        print(
            "Population:",
            record[
                "populationPercent"
            ]
        )

        print(
            "Labels:",
            record[
                "labelsOrTypes"
            ]
        )

        print(
            "Properties:",
            record[
                "properties"
            ]
        )


# ============================================================
# Preview
# ============================================================

def print_preview(
    documents
):

    print()
    print(
        "=== SEMANTIC DOCUMENT PREVIEW ==="
    )

    for index, document in enumerate(
        documents[
            :PREVIEW_LIMIT
        ],
        start=1
    ):

        print()
        print(
            f"{index}. "
            f"[{document['kind']}] "
            f"[{document['category']}]"
        )

        print(
            document["title"]
        )

        print(
            "ID:",
            document["id"]
        )

        print(
            "---"
        )

        print(
            document["text"]
        )

    if (
        len(documents)
        > PREVIEW_LIMIT
    ):

        print()
        print(
            f"... "
            f"{len(documents) - PREVIEW_LIMIT} "
            f"additional documents not shown."
        )


# ============================================================
# Main
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Connectivity
        # ----------------------------------------------------

        driver.verify_connectivity()

        print(
            "Connected to Neo4j successfully."
        )

        print(
            "Embedding model:",
            EMBED_MODEL
        )

        print(
            "Knowledge vector index:",
            VECTOR_INDEX_NAME
        )

        print(
            "Preferred language:",
            PREFERRED_LANGUAGE
        )

        # ----------------------------------------------------
        # Read graph
        # ----------------------------------------------------

        print()
        print(
            "Reading CASCaRA graph..."
        )

        items = get_cascara_items()

        item_lookup = (
            build_item_lookup(
                items
            )
        )

        endpoint_map = (
            get_relationship_endpoints()
        )

        native_facts = (
            get_native_facts()
        )

        print(
            "CascaraItems:",
            len(items)
        )

        print(
            "First-class relationship statements:",
            len(endpoint_map)
        )

        print(
            "Native CASCaRA facts:",
            len(native_facts)
        )

        # ----------------------------------------------------
        # Item documents
        # ----------------------------------------------------

        documents = []

        for raw_item in items:

            item = item_lookup[
                raw_item["id"]
            ]

            if not should_index_item(
                item
            ):
                continue

            documents.append(
                build_item_document(
                    item,
                    endpoint_map,
                    item_lookup
                )
            )

        item_document_count = len(
            documents
        )

        # ----------------------------------------------------
        # Fact documents
        # ----------------------------------------------------

        for fact in native_facts:

            documents.append(
                build_fact_document(
                    fact,
                    item_lookup
                )
            )

        fact_document_count = (
            len(documents)
            - item_document_count
        )

        print()
        print(
            "=== DOCUMENT COUNTS ==="
        )

        print(
            "Item documents:",
            item_document_count
        )

        print(
            "Fact documents:",
            fact_document_count
        )

        print(
            "Total semantic documents:",
            len(documents)
        )

        # ----------------------------------------------------
        # Category counts
        # ----------------------------------------------------

        category_counts = Counter(
            document["category"]
            for document
            in documents
        )

        print()
        print(
            "=== CATEGORIES ==="
        )

        for category, count in sorted(
            category_counts.items()
        ):

            print(
                f"{category}: {count}"
            )

        # ----------------------------------------------------
        # Preview BEFORE spending time embedding
        # ----------------------------------------------------

        print_preview(
            documents
        )

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        print()
        print(
            "Generating embeddings..."
        )

        embed_documents(
            documents
        )

        # ----------------------------------------------------
        # Build v2 layer
        # ----------------------------------------------------

        print()
        print(
            "Rebuilding v0.2 semantic layer..."
        )

        create_constraint()

        clear_v2_documents()

        save_documents(
            documents
        )

        create_item_links(
            documents
        )

        create_endpoint_links(
            documents
        )

        # ----------------------------------------------------
        # Vector index
        # ----------------------------------------------------

        print()
        print(
            "Creating / checking vector index..."
        )

        create_vector_index()

        show_vector_index()

        # ----------------------------------------------------
        # Complete
        # ----------------------------------------------------

        print()
        print(
            "=================================="
        )

        print(
            "CASCaRA knowledge indexing complete."
        )

        print(
            "=================================="
        )

        print()
        print(
            "v0.1 was NOT modified."
        )

        print(
            "Old index:"
        )

        print(
            "  cascara_entity_embeddings"
        )

        print()
        print(
            "New v0.2 index:"
        )

        print(
            f"  {VECTOR_INDEX_NAME}"
        )

    finally:

        driver.close()


if __name__ == "__main__":
    main()