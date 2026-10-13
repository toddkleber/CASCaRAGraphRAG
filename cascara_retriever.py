import re
from typing import Any, Optional

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResultItem


class CascaraGraphRetriever(Retriever):
    """
    CASCaRA-aware GraphRAG retriever for Neo4j GraphRAG v0.4.

    Retrieval flow:
        1. Embed the question with Neo4j's OllamaEmbeddings.
        2. Run Neo4j vector search over (:RagKnowledge.embedding).
        3. Convert top vector hits into CASCaRA seed nodes.
        4. Expand connected evidence with APOC over the normalized semantic layer.
        5. Run GDS Personalized PageRank from the query seed nodes.
        6. Rerank by semantic similarity + GDS score + APOC distance.
        7. Return focused context items to Neo4j's GraphRAG pipeline.

    This assumes embedding.py/index_cascara.py has already created:
        (:RagKnowledge { text, embedding, ... })
        (:RagKnowledge)-[:RAG_DESCRIBES]->(:CascaraItem)
        (:RagKnowledge)-[:RAG_SOURCE]->(:CascaraItem)
        (:RagKnowledge)-[:RAG_TARGET]->(:CascaraItem)
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        embedder: OllamaEmbeddings,
        index_name: str,
        neo4j_database: Optional[str] = None,
        gds_graph_name: str = "cascara_rag_v04",
        seed_documents: int = 5,
        apoc_max_level: int = 7,
        apoc_limit_per_seed: int = 100,# limit for apoc.path.expandConfig() per seed node
        final_evidence_limit: int = 15,# limit for final reranked evidence returned to GraphRAG
        semantic_weight: float = 0.65,
        gds_weight: float = 0.25,
        distance_weight: float = 0.10,
        debug: bool = True,#this is for debugging purposes, set to False in production
    ) -> None:
        super().__init__(driver, neo4j_database=neo4j_database)

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", index_name):
            raise ValueError("Invalid Neo4j vector index name.")

        self.embedder = embedder
        self.index_name = index_name
        self.gds_graph_name = gds_graph_name
        self.seed_documents = seed_documents
        self.apoc_max_level = apoc_max_level
        self.apoc_limit_per_seed = apoc_limit_per_seed
        self.final_evidence_limit = final_evidence_limit
        self.semantic_weight = semantic_weight
        self.gds_weight = gds_weight
        self.distance_weight = distance_weight
        self.debug = debug

    # ========================================================
    # Neo4j GraphRAG hook
    # ========================================================

    def get_search_results(
        self,
        query_text: Optional[str] = None,
        query_vector: Optional[list[float]] = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RawSearchResult:
        """
        Neo4j GraphRAG calls this method through Retriever.search().
        It must return RawSearchResult(records=[neo4j.Record, ...]).
        """

        if query_vector is None:
            if not query_text:
                raise ValueError("query_text or query_vector is required.")

            query_vector = self.embedder.embed_query(query_text)

        vector_records = self._vector_search(
            query_embedding=query_vector,
            top_k=top_k,
        )

        seed_ids = self._get_seed_item_ids(
            vector_records[: self.seed_documents]
        )

        apoc_distances = self._apoc_expand(seed_ids)
        gds_scores = self._gds_personalized_pagerank(seed_ids)

        candidate_ids = {
            record["document_id"]
            for record in vector_records
            if record.get("document_id")
        }

        candidate_ids.update(apoc_distances.keys())

        evidence_records = self._get_final_candidate_records(
            candidate_ids=candidate_ids,
            query_embedding=query_vector,
            apoc_distances=apoc_distances,
            gds_scores=gds_scores,
            limit=kwargs.get("final_evidence_limit", self.final_evidence_limit),
        )

        if self.debug:
            self._print_debug(
                vector_records=vector_records,
                seed_ids=seed_ids,
                apoc_distances=apoc_distances,
                gds_scores=gds_scores,
                evidence_records=evidence_records,
            )

        return RawSearchResult(
            records=evidence_records,
            metadata={
                "seed_ids": seed_ids,
                "vector_results": len(vector_records),
                "apoc_candidates": len(apoc_distances),
                "gds_candidates": len(gds_scores),
                "final_evidence": len(evidence_records),
            },
        )

    def get_result_formatter(self):
        return self._format_record

    @staticmethod
    def _format_record(record: neo4j.Record) -> RetrieverResultItem:
        metadata = {
            "document_id": record.get("document_id"),
            "kind": record.get("kind"),
            "category": record.get("category"),
            "title": record.get("title"),
            "cascara_id": record.get("cascara_id"),
            "relationship_type": record.get("relationship_type"),
            "relationship_class": record.get("relationship_class"),
            "semantic_score": record.get("semantic_score"),
            "gds_score": record.get("gds_score"),
            "apoc_distance": record.get("apoc_distance"),
            "final_score": record.get("final_score"),
        }

        content = (
            f"Title: {record.get('title')}\n"
            f"Category: {record.get('category')}\n"
            f"Kind: {record.get('kind')}\n"
            f"Text:\n{record.get('text')}"
        )

        return RetrieverResultItem(
            content=content,
            metadata=metadata,
        )

    # ========================================================
    # Vector search
    # ========================================================

    def _vector_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[neo4j.Record]:
        query = f"""
        MATCH (doc:RagKnowledge)

        SEARCH doc IN (
            VECTOR INDEX {self.index_name}
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
            doc.relationshipType AS relationship_type,
            doc.relationshipClass AS relationship_class,
            doc.text AS text,
            score AS semantic_score,
            collect(DISTINCT described.id) AS described_ids,
            collect(DISTINCT source.id) AS source_ids,
            collect(DISTINCT target.id) AS target_ids

        ORDER BY semantic_score DESC
        """

        records, _, _ = self.driver.execute_query(
            query,
            query_embedding=query_embedding,
            top_k=top_k,
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )

        return records

    @staticmethod
    def _get_seed_item_ids(
        vector_records: list[neo4j.Record],
    ) -> list[str]:
        seed_ids: list[str] = []

        for record in vector_records:
            for field in (
                "described_ids",
                "source_ids",
                "target_ids",
            ):
                for item_id in record.get(field) or []:
                    if item_id and item_id not in seed_ids:
                        seed_ids.append(item_id)

        return seed_ids

    # ========================================================
    # APOC graph expansion
    # ========================================================

    def _apoc_expand(
        self,
        seed_ids: list[str],
    ) -> dict[str, int]:
        if not seed_ids:
            return {}

        records, _, _ = self.driver.execute_query(
            """
            UNWIND $seed_ids AS seed_id

            MATCH (seed:CascaraItem {id: seed_id})

            CALL apoc.path.expandConfig(
                seed,
                {
                    relationshipFilter:
                        'RAG_DESCRIBES|RAG_SOURCE|RAG_TARGET',
                    minLevel: 1,
                    maxLevel: $max_level,
                    bfs: true,
                    uniqueness: 'NODE_PATH',
                    limit: $limit_per_seed
                }
            )
            YIELD path

            WITH
                last(nodes(path)) AS endpoint,
                length(path) AS distance

            WHERE endpoint:RagKnowledge

            RETURN
                endpoint.id AS document_id,
                min(distance) AS apoc_distance
            """,
            seed_ids=seed_ids,
            max_level=self.apoc_max_level,
            limit_per_seed=self.apoc_limit_per_seed,
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )

        return {
            record["document_id"]: record["apoc_distance"]
            for record in records
        }

    # ========================================================
    # GDS Personalized PageRank
    # ========================================================

    def rebuild_gds_projection(self) -> dict[str, Any]:
        """
        Rebuild the in-memory GDS graph used by this retriever.
        Safe to call each time GraphRag.py starts.
        """

        self.driver.execute_query(
            """
            CALL gds.graph.drop($graph_name, false)
            YIELD graphName
            RETURN graphName
            """,
            graph_name=self.gds_graph_name,
            database_=self.neo4j_database,
        )

        records, _, _ = self.driver.execute_query(
            """
            CALL gds.graph.project(
                $graph_name,
                ['CascaraItem', 'RagKnowledge'],
                {
                    RAG_DESCRIBES: {
                        type: 'RAG_DESCRIBES',
                        orientation: 'UNDIRECTED'
                    },
                    RAG_SOURCE: {
                        type: 'RAG_SOURCE',
                        orientation: 'UNDIRECTED'
                    },
                    RAG_TARGET: {
                        type: 'RAG_TARGET',
                        orientation: 'UNDIRECTED'
                    }
                }
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """,
            graph_name=self.gds_graph_name,
            database_=self.neo4j_database,
        )

        if not records:
            return {}

        return dict(records[0])

    def _gds_personalized_pagerank(
        self,
        seed_ids: list[str],
    ) -> dict[str, float]:
        if not seed_ids:
            return {}

        records, _, _ = self.driver.execute_query(
            """
            MATCH (seed:CascaraItem)
            WHERE seed.id IN $seed_ids

            WITH collect(seed) AS seeds

            CALL gds.pageRank.stream(
                $graph_name,
                {
                    sourceNodes: seeds,
                    maxIterations: 20,
                    dampingFactor: 0.85,
                    scaler: 'MinMax'
                }
            )
            YIELD nodeId, score

            WITH
                gds.util.asNode(nodeId) AS node,
                score

            WHERE node:RagKnowledge

            RETURN
                node.id AS document_id,
                score AS gds_score
            """,
            seed_ids=seed_ids,
            graph_name=self.gds_graph_name,
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )

        return {
            record["document_id"]: float(record["gds_score"] or 0.0)
            for record in records
        }

    # ========================================================
    # Final reranking
    # ========================================================

    def _get_final_candidate_records(
        self,
        candidate_ids: set[str],
        query_embedding: list[float],
        apoc_distances: dict[str, int],
        gds_scores: dict[str, float],
        limit: int,
    ) -> list[neo4j.Record]:
        if not candidate_ids:
            return []

        records, _, _ = self.driver.execute_query(
            """
            MATCH (doc:RagKnowledge)
            WHERE doc.id IN $candidate_ids

            WITH
                doc,
                vector.similarity.cosine(
                    doc.embedding,
                    $query_embedding
                ) AS semantic_score,
                coalesce($gds_scores[doc.id], 0.0) AS gds_score,
                $apoc_distances[doc.id] AS apoc_distance

            WITH
                doc,
                semantic_score,
                gds_score,
                apoc_distance,
                CASE
                    WHEN apoc_distance IS NULL THEN 0.0
                    ELSE 1.0 / toFloat(apoc_distance)
                END AS distance_score

            WITH
                doc,
                semantic_score,
                gds_score,
                apoc_distance,
                distance_score,
                (
                    $semantic_weight * semantic_score
                    + $gds_weight * gds_score
                    + $distance_weight * distance_score
                ) AS final_score

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
                doc.relationshipType AS relationship_type,
                doc.relationshipClass AS relationship_class,
                doc.text AS text,
                semantic_score,
                gds_score,
                apoc_distance,
                distance_score,
                final_score,
                collect(DISTINCT described.id) AS described_ids,
                collect(DISTINCT source.id) AS source_ids,
                collect(DISTINCT target.id) AS target_ids

            ORDER BY final_score DESC
            LIMIT $limit
            """,
            candidate_ids=list(candidate_ids),
            query_embedding=query_embedding,
            apoc_distances=apoc_distances,
            gds_scores=gds_scores,
            semantic_weight=self.semantic_weight,
            gds_weight=self.gds_weight,
            distance_weight=self.distance_weight,
            limit=limit,
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )

        return records

    # ========================================================
    # Debug printing
    # ========================================================

    @staticmethod
    def _score_text(value: Any) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _print_debug(
        self,
        vector_records: list[neo4j.Record],
        seed_ids: list[str],
        apoc_distances: dict[str, int],
        gds_scores: dict[str, float],
        evidence_records: list[neo4j.Record],
    ) -> None:
        print()
        print("=== INITIAL VECTOR SEARCH ===")
        for index, record in enumerate(vector_records, start=1):
            print(
                f"{index}. {record.get('title')} "
                f"({self._score_text(record.get('semantic_score'))}) "
                f"[{record.get('category')}]"
            )

        print()
        print("=== CASCaRA SEEDS ===")
        for seed_id in seed_ids:
            print(seed_id)

        print()
        print("APOC candidate documents:", len(apoc_distances))
        print("GDS ranked documents:", len(gds_scores))

        print()
        print("=== FINAL RERANKED EVIDENCE ===")
        for index, record in enumerate(evidence_records, start=1):
            print(f"{index}. {record.get('title')}")
            print(
                "   final="
                f"{self._score_text(record.get('final_score'))} "
                "semantic="
                f"{self._score_text(record.get('semantic_score'))} "
                "gds="
                f"{self._score_text(record.get('gds_score'))} "
                "distance="
                f"{record.get('apoc_distance') if record.get('apoc_distance') is not None else '-'}"
            )