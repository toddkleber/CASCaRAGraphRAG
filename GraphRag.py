import os
import re

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

CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "qwen3:8b"
)

VECTOR_INDEX_NAME = os.getenv(
    "KNOWLEDGE_VECTOR_INDEX_NAME",
    "cascara_knowledge_embeddings"
)


# ============================================================
# Retrieval configuration
# ============================================================

VECTOR_TOP_K = int(
    os.getenv(
        "KNOWLEDGE_VECTOR_TOP_K",
        "10"
    )
)

# Top vector documents used to establish graph entry points.
SEED_DOCUMENTS = int(
    os.getenv(
        "KNOWLEDGE_SEED_DOCUMENTS",
        "5"
    )
)

# Number of semantic graph-expansion rounds.
MAX_DEPTH = int(
    os.getenv(
        "KNOWLEDGE_MAX_DEPTH",
        "4"
    )
)

# Maximum new connected documents kept at each depth.
MAX_DOCS_PER_DEPTH = int(
    os.getenv(
        "KNOWLEDGE_DOCS_PER_DEPTH",
        "10"
    )
)

# Hard safety limits.
MAX_EVIDENCE_DOCS = int(
    os.getenv(
        "KNOWLEDGE_MAX_EVIDENCE",
        "40"
    )
)

MAX_GRAPH_ITEMS = int(
    os.getenv(
        "KNOWLEDGE_MAX_GRAPH_ITEMS",
        "100"
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
# Embedding
# ============================================================

def embed_question(question):

    response = ollama.embed(
        model=EMBED_MODEL,
        input=question
    )

    return response.embeddings[0]


# ============================================================
# Initial vector retrieval
# ============================================================

def vector_search(
    query_embedding,
    top_k
):

    query = f"""
    MATCH (doc:RagKnowledge)

    SEARCH doc IN (
        VECTOR INDEX {VECTOR_INDEX_NAME}
        FOR $query_embedding
        LIMIT $top_k
    )

    SCORE AS score

    OPTIONAL MATCH
        (doc)-[:RAG_DESCRIBES]->
        (described:CascaraItem)

    OPTIONAL MATCH
        (doc)-[:RAG_SOURCE]->
        (source:CascaraItem)

    OPTIONAL MATCH
        (doc)-[:RAG_TARGET]->
        (target:CascaraItem)

    RETURN
        doc.id AS document_id,
        doc.kind AS kind,
        doc.category AS category,
        doc.title AS title,

        doc.cascaraId AS cascara_id,

        doc.relationshipType
            AS relationship_type,

        doc.relationshipClass
            AS relationship_class,

        doc.text AS text,

        score,

        collect(
            DISTINCT described.id
        ) AS described_ids,

        collect(
            DISTINCT source.id
        ) AS source_ids,

        collect(
            DISTINCT target.id
        ) AS target_ids

    ORDER BY score DESC
    """

    records, _, _ = driver.execute_query(
        query,

        query_embedding=query_embedding,
        top_k=top_k,

        database_=NEO4J_DATABASE
    )

    return [
        dict(record)
        for record in records
    ]


# ============================================================
# Connected semantic graph search
# ============================================================

def connected_documents(
    frontier_ids,
    query_embedding,
    limit
):
    """
    Find RagKnowledge documents touching any CascaraItem in
    the current frontier.

    This is the graph part of GraphRAG.

    We do NOT need to know whether the underlying relationship
    is SpecIF, FMC, OSLC, or some other ontology.

    embedding_v2.py has already mapped the CASCaRA semantics
    into:

        RAG_DESCRIBES
        RAG_SOURCE
        RAG_TARGET
    """

    if not frontier_ids:
        return []

    records, _, _ = driver.execute_query(
        """
        MATCH
            (doc:RagKnowledge)
            -[semanticLink]->
            (anchor:CascaraItem)

        WHERE
            type(semanticLink) IN [
                'RAG_DESCRIBES',
                'RAG_SOURCE',
                'RAG_TARGET'
            ]

            AND

            anchor.id IN $frontier_ids

        WITH DISTINCT doc

        WITH
            doc,
            vector.similarity.cosine(
                doc.embedding,
                $query_embedding
            ) AS score

        OPTIONAL MATCH
            (doc)-[:RAG_DESCRIBES]->
            (described:CascaraItem)

        OPTIONAL MATCH
            (doc)-[:RAG_SOURCE]->
            (source:CascaraItem)

        OPTIONAL MATCH
            (doc)-[:RAG_TARGET]->
            (target:CascaraItem)

        RETURN
            doc.id AS document_id,
            doc.kind AS kind,
            doc.category AS category,
            doc.title AS title,

            doc.cascaraId AS cascara_id,

            doc.relationshipType
                AS relationship_type,

            doc.relationshipClass
                AS relationship_class,

            doc.text AS text,

            score,

            collect(
                DISTINCT described.id
            ) AS described_ids,

            collect(
                DISTINCT source.id
            ) AS source_ids,

            collect(
                DISTINCT target.id
            ) AS target_ids

        ORDER BY score DESC

        LIMIT $limit
        """,

        frontier_ids=list(
            frontier_ids
        ),

        query_embedding=query_embedding,

        limit=limit,

        database_=NEO4J_DATABASE
    )

    return [
        dict(record)
        for record in records
    ]


# ============================================================
# Document -> graph items
# ============================================================

def document_item_ids(document):

    result = []

    for field in (
        "described_ids",
        "source_ids",
        "target_ids",
    ):

        for item_id in (
            document.get(field)
            or []
        ):

            if (
                item_id
                and item_id not in result
            ):
                result.append(
                    item_id
                )

    return result


# ============================================================
# GraphRAG retrieval
# ============================================================

def retrieve_graph_context(question):

    query_embedding = (
        embed_question(
            question
        )
    )

    # --------------------------------------------------------
    # Step 1: vector retrieval
    # --------------------------------------------------------

    vector_results = vector_search(
        query_embedding,
        VECTOR_TOP_K
    )

    print()
    print(
        "=== INITIAL VECTOR SEARCH ==="
    )

    for index, result in enumerate(
        vector_results,
        start=1
    ):

        print(
            f"{index}. "
            f"{result['title']} "
            f"({result['score']:.4f}) "
            f"[{result['category']}]"
        )


    # --------------------------------------------------------
    # Step 2: establish semantic graph seeds
    # --------------------------------------------------------

    seed_docs = vector_results[
        :SEED_DOCUMENTS
    ]

    evidence = {}
    visited_items = set()
    frontier = set()


    for document in seed_docs:

        evidence[
            document["document_id"]
        ] = {
            **document,
            "origin": "vector"
        }

        for item_id in document_item_ids(
            document
        ):

            frontier.add(
                item_id
            )

            visited_items.add(
                item_id
            )


    print()
    print(
        "=== GRAPH SEEDS ==="
    )

    for item_id in sorted(
        frontier
    ):

        print(
            item_id
        )


    # --------------------------------------------------------
    # Step 3: graph expansion
    # --------------------------------------------------------

    traversal_log = []

    for depth in range(
        1,
        MAX_DEPTH + 1
    ):

        if not frontier:
            break

        if (
            len(evidence)
            >= MAX_EVIDENCE_DOCS
        ):
            break

        if (
            len(visited_items)
            >= MAX_GRAPH_ITEMS
        ):
            break


        # Retrieve more than we intend to keep because some
        # candidates may already have been seen.

        query_limit = (
            MAX_DOCS_PER_DEPTH * 3
        )

        candidates = connected_documents(
            frontier,
            query_embedding,
            query_limit
        )


        new_documents = []

        for document in candidates:

            if (
                document["document_id"]
                in evidence
            ):
                continue

            new_documents.append(
                document
            )

            if (
                len(new_documents)
                >= MAX_DOCS_PER_DEPTH
            ):
                break


        next_frontier = set()


        for document in new_documents:

            if (
                len(evidence)
                >= MAX_EVIDENCE_DOCS
            ):
                break

            evidence[
                document["document_id"]
            ] = {
                **document,
                "origin":
                    f"graph_depth_{depth}"
            }

            for item_id in document_item_ids(
                document
            ):

                if (
                    item_id
                    not in visited_items
                ):

                    next_frontier.add(
                        item_id
                    )


        traversal_log.append(
            {
                "depth":
                    depth,

                "frontier_count":
                    len(frontier),

                "candidate_count":
                    len(candidates),

                "kept_count":
                    len(new_documents),

                "next_frontier_count":
                    len(next_frontier),
            }
        )


        visited_items.update(
            next_frontier
        )

        frontier = next_frontier


    # --------------------------------------------------------
    # Step 4: final semantic ordering
    # --------------------------------------------------------

    evidence_list = list(
        evidence.values()
    )

    evidence_list.sort(
        key=lambda item:
            item["score"],
        reverse=True
    )


    print()
    print(
        "=== GRAPH TRAVERSAL ==="
    )

    for entry in traversal_log:

        print()
        print(
            f"Depth {entry['depth']}:"
        )

        print(
            "  Frontier items:",
            entry[
                "frontier_count"
            ]
        )

        print(
            "  Candidate documents:",
            entry[
                "candidate_count"
            ]
        )

        print(
            "  Documents kept:",
            entry[
                "kept_count"
            ]
        )

        print(
            "  New graph items:",
            entry[
                "next_frontier_count"
            ]
        )


    print()
    print(
        "=== FINAL EVIDENCE ==="
    )

    for index, document in enumerate(
        evidence_list,
        start=1
    ):

        print(
            f"{index}. "
            f"{document['title']} "
            f"({document['score']:.4f}) "
            f"[{document['category']}] "
            f"<{document['origin']}>"
        )


    return evidence_list


# ============================================================
# LLM context
# ============================================================

def build_llm_context(
    evidence
):

    blocks = []

    for index, document in enumerate(
        evidence,
        start=1
    ):

        block = []

        block.append(
            f"[Evidence {index}]"
        )

        block.append(
            f"Kind: "
            f"{document['kind']}"
        )

        block.append(
            f"Category: "
            f"{document['category']}"
        )

        block.append(
            f"Semantic score: "
            f"{document['score']:.4f}"
        )

        if document.get(
            "relationship_class"
        ):

            block.append(
                "Relationship class: "
                + document[
                    "relationship_class"
                ]
            )

        elif document.get(
            "relationship_type"
        ):

            block.append(
                "Relationship type: "
                + document[
                    "relationship_type"
                ]
            )

        block.append(
            document["text"]
        )

        blocks.append(
            "\n".join(
                block
            )
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# Answer
# ============================================================

def answer_question(
    question,
    evidence
):

    context = build_llm_context(
        evidence
    )

    system_prompt = """
You answer questions using only the supplied CASCaRA graph
evidence.

The evidence was retrieved using semantic vector search and
CASCaRA graph traversal.

Important rules:

1. Treat CASCaRA graph statements as authoritative evidence.

2. Vector similarity scores only indicate retrieval relevance.
   A higher score does NOT mean a statement is more true or
   more correct.

3. Prefer evidence whose source, relationship, and target
   directly address the user's question.

4. Ignore retrieved facts that are unrelated to the question.

5. Do not invent facts that are not present in the supplied
   evidence.

6. A first-class CASCaRA relationship statement such as:

       A -- relationship --> B

   represents a factual relationship in the graph.

7. Schema facts, structural facts, link instances, and model
   statements may all be useful depending on the question.



Answer clearly and concisely.
""".strip()


    user_prompt = f"""
QUESTION:

{question}


CASCaRA GRAPH EVIDENCE:

{context}
""".strip()


    response = ollama.chat(
        model=CHAT_MODEL,

        messages=[
            {
                "role":
                    "system",

                "content":
                    system_prompt,
            },

            {
                "role":
                    "user",

                "content":
                    user_prompt,
            },
        ],
    )


    return response.message.content


# ============================================================
# Main
# ============================================================

def main():

    try:

        driver.verify_connectivity()

        print(
            "Connected to Neo4j successfully."
        )

        print()
        print(
            "=== CONFIGURATION ==="
        )

        print(
            "Neo4j:",
            NEO4J_URI
        )

        print(
            "Database:",
            NEO4J_DATABASE
        )

        print(
            "Embedding model:",
            EMBED_MODEL
        )

        print(
            "Chat model:",
            CHAT_MODEL
        )

        print(
            "Knowledge index:",
            VECTOR_INDEX_NAME
        )

        print(
            "Vector top K:",
            VECTOR_TOP_K
        )

        print(
            "Seed documents:",
            SEED_DOCUMENTS
        )

        print(
            "Maximum depth:",
            MAX_DEPTH
        )

        print(
            "Documents per depth:",
            MAX_DOCS_PER_DEPTH
        )

        print()
        print(
            "CASCaRA GraphRAG Prototype v0.2"
        )

        print(
            "Type 'exit' to quit."
        )


        while True:

            print()

            question = input(
                "Question: "
            ).strip()


            if not question:
                continue


            if question.lower() in {
                "exit",
                "quit"
            }:
                break


            evidence = (
                retrieve_graph_context(
                    question
                )
            )


            print()
            print(
                "=== AI ANSWER ==="
            )

            answer = answer_question(
                question,
                evidence
            )

            print(
                answer
            )


    finally:

        driver.close()


if __name__ == "__main__":
    main()