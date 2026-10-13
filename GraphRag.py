import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.llm import OllamaLLM

from cascara_retriever import CascaraGraphRetriever


load_dotenv()


# ============================================================
# Environment
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")

VECTOR_INDEX_NAME = os.getenv(
    "KNOWLEDGE_VECTOR_INDEX_NAME",
    "cascara_knowledge_embeddings",
)

GDS_GRAPH_NAME = os.getenv("GDS_GRAPH_NAME", "cascara_rag_v04")

VECTOR_TOP_K = int(os.getenv("KNOWLEDGE_VECTOR_TOP_K", "10"))
SEED_DOCUMENTS = int(os.getenv("KNOWLEDGE_SEED_DOCUMENTS", "5"))
APOC_MAX_LEVEL = int(os.getenv("APOC_MAX_LEVEL", "7"))
APOC_LIMIT_PER_SEED = int(os.getenv("APOC_LIMIT_PER_SEED", "100"))
FINAL_EVIDENCE_LIMIT = int(os.getenv("FINAL_EVIDENCE_LIMIT", "15"))

SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.65"))
GDS_WEIGHT = float(os.getenv("GDS_WEIGHT", "0.25"))
DISTANCE_WEIGHT = float(os.getenv("DISTANCE_WEIGHT", "0.10"))

OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "4096"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

DEBUG_RETRIEVAL = os.getenv("DEBUG_RETRIEVAL", "true").lower() in {
    "true",
    "1",
    "yes",
}


if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is missing from .env")


# ============================================================
# Prompt
# ============================================================

SYSTEM_INSTRUCTIONS = """
You answer questions using only the supplied CASCaRA graph evidence.

The evidence was retrieved from a CASCaRA knowledge graph.

Rules:
1. Treat explicit CASCaRA graph statements as authoritative.
2. Retrieval scores are relevance signals only. A higher score does not make a statement more true.
3. Prefer evidence whose source, relationship, and target directly answer the question.
4. Ignore unrelated structural or schema facts.
5. Do not invent information.
6. Follow multi-hop facts when necessary.
7. If the supplied evidence is insufficient, respond exactly:
The retrieved CASCaRA graph does not contain enough information to determine that.
8. Do not mention evidence numbers, vector scores, APOC, GDS, PageRank, internal IDs, or retrieval mechanics unless the user explicitly asks how the result was obtained.

Answer naturally and concisely.
""".strip()

PROMPT_TEMPLATE = """
Context:
{context}

Examples:
{examples}

Question:
{query_text}

Answer:
""".strip()


def verify_plugins(driver):
    apoc_records, _, _ = driver.execute_query(
        "RETURN apoc.version() AS version",
        database_=NEO4J_DATABASE,
    )

    gds_records, _, _ = driver.execute_query(
        "RETURN gds.version() AS version",
        database_=NEO4J_DATABASE,
    )

    return apoc_records[0]["version"], gds_records[0]["version"]


def build_rag(driver):
    embedder = OllamaEmbeddings(model=EMBED_MODEL)

    llm = OllamaLLM(
        model_name=CHAT_MODEL,
        model_params={
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT,
            }
        },
    )

    retriever = CascaraGraphRetriever(
        driver=driver,
        embedder=embedder,
        index_name=VECTOR_INDEX_NAME,
        neo4j_database=NEO4J_DATABASE,
        gds_graph_name=GDS_GRAPH_NAME,
        seed_documents=SEED_DOCUMENTS,
        apoc_max_level=APOC_MAX_LEVEL,
        apoc_limit_per_seed=APOC_LIMIT_PER_SEED,
        final_evidence_limit=FINAL_EVIDENCE_LIMIT,
        semantic_weight=SEMANTIC_WEIGHT,
        gds_weight=GDS_WEIGHT,
        distance_weight=DISTANCE_WEIGHT,
        debug=DEBUG_RETRIEVAL,
    )

    projection = retriever.rebuild_gds_projection()

    prompt_template = RagTemplate(
        template=PROMPT_TEMPLATE,
        expected_inputs=["context", "query_text", "examples"],
        system_instructions=SYSTEM_INSTRUCTIONS,
    )

    rag = GraphRAG(
        retriever=retriever,
        llm=llm,
        prompt_template=prompt_template,
    )

    return rag, projection


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        print("Connected to Neo4j successfully.")

        apoc_version, gds_version = verify_plugins(driver)

        print()
        print("=== PLUGINS ===")
        print("APOC:", apoc_version)
        print("GDS:", gds_version)

        rag, projection = build_rag(driver)

        print()
        print("=== GDS PROJECTION ===")
        print("Graph:", projection.get("graphName"))
        print("Nodes:", projection.get("nodeCount"))
        print("Relationships:", projection.get("relationshipCount"))

        print()
        print("=== CONFIGURATION ===")
        print("Neo4j:", NEO4J_URI)
        print("Database:", NEO4J_DATABASE)
        print("Embedding model:", EMBED_MODEL)
        print("Chat model:", CHAT_MODEL)
        print("Knowledge index:", VECTOR_INDEX_NAME)
        print("Vector top K:", VECTOR_TOP_K)
        print("Seed documents:", SEED_DOCUMENTS)
        print("APOC max level:", APOC_MAX_LEVEL)
        print("Final evidence:", FINAL_EVIDENCE_LIMIT)

        print()
        print("CASCaRA GraphRAG Prototype v0.4")
        print("Neo4j GraphRAG + Ollama + APOC + GDS")
        print("Type 'exit' to quit.")

        while True:
            print()
            question = input("Question: ").strip()

            if not question:
                continue

            if question.lower() in {"exit", "quit"}:
                break

            response = rag.search(
                query_text=question,
                retriever_config={
                    "top_k": VECTOR_TOP_K,
                    "final_evidence_limit": FINAL_EVIDENCE_LIMIT,
                },
                return_context=True,
                response_fallback="The retrieved CASCaRA graph does not contain enough information to determine that.",
            )

            print()
            print("=== AI ANSWER ===")
            print(response.answer)

    finally:
        driver.close()


if __name__ == "__main__":
    main()