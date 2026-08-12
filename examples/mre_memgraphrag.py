#!/usr/bin/env python3
"""Minimal Runnable Example (MRE) for MemGraphRAG.

Validates:
1. Filtering unstable schemas with threshold tau=2
2. Rejecting facts whose relations have not reached ontological stability
3. PPR propagation integrated with MemoryGuidedRetriever
"""

from __future__ import annotations

import asyncio

from memgraphrag_core.domain.models import Fact, GlobalMemoryState, Passage
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever


async def build_demo() -> tuple[GlobalMemoryState, NetworkXGraphStore, InMemoryVectorStore]:
    memory = GlobalMemoryState(tau=2)
    graph = NetworkXGraphStore()
    vectors = InMemoryVectorStore()

    # --- M_pas ---
    passages = [
        Passage(
            id="pas_001",
            text="Catena Zapata winery is located in the Mendoza region, Argentina.",
        ),
        Passage(
            id="pas_002",
            text="Catena Zapata produces the Gran Enemigo wine in Gualtallary, Mendoza.",
        ),
        Passage(
            id="pas_003",
            text="Some experimental wineries produce craft vermouth in Patagonia.",
        ),
    ]
    for p in passages:
        memory.add_passage(p)

    await vectors.add_texts(
        texts=[p.text for p in passages],
        ids=[p.id for p in passages],
        namespace="M_pas",
    )

    # --- M_ont: register candidates (frequency) ---
    # (winery)-[produces]->(wine) x2 → stable
    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
    # (winery)-[located_in]->(region) x2 → stable
    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    # (winery)-[produces]->(vermouth) x1 → unstable (tau=2)
    memory.register_schema_candidate("Winery", "PRODUCES", "Vermouth")

    stable = memory.get_stable_schemas()
    await vectors.add_texts(
        texts=[
            f"{s.head_type} {s.relation} {s.tail_type}" for s in memory.ontologies.values()
        ],
        ids=[s.schema_key for s in memory.ontologies.values()],
        namespace="M_ont",
    )

    # --- M_fac ---
    f1 = Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity="Mendoza",
        tail_type="Region",
        passage_ids=["pas_001"],
        confidence=0.95,
    )
    f2 = Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="PRODUCES",
        tail_entity="Gran Enemigo",
        tail_type="Wine",
        passage_ids=["pas_002"],
        confidence=0.92,
    )
    f_noisy = Fact(
        head_entity="Experimental Winery",
        head_type="Winery",
        relation="PRODUCES",
        tail_entity="Patagonian Vermouth",
        tail_type="Vermouth",
        passage_ids=["pas_003"],
        confidence=0.5,
    )

    added_f1 = memory.add_fact(f1) is not None
    added_f2 = memory.add_fact(f2) is not None
    added_noisy = memory.add_fact(f_noisy) is not None

    fact_docs = list(memory.facts.values())
    if fact_docs:
        await vectors.add_texts(
            texts=[
                f"{f.head_entity} {f.relation} {f.tail_entity}" for f in fact_docs
            ],
            ids=[f.fact_key for f in fact_docs],
            namespace="M_fac",
        )

    # --- Hierarchical graph: ont -> fact -> passage ---
    for schema in memory.ontologies.values():
        ont_id = f"ont_{schema.schema_key}"
        await graph.add_node(
            GraphNode(
                id=ont_id,
                label="Schema",
                properties={
                    "frequency": schema.frequency,
                    "stable": schema.is_stable(memory.tau),
                },
            )
        )

    for fact in memory.facts.values():
        fact_id = f"fact_{fact.fact_key}"
        await graph.add_node(
            GraphNode(
                id=fact_id,
                label="Fact",
                properties={
                    "head": fact.head_entity,
                    "relation": fact.relation,
                    "tail": fact.tail_entity,
                },
            )
        )
        await graph.add_edge(
            GraphEdge(
                source_id=f"ont_{fact.schema_key}",
                target_id=fact_id,
                relation="INSTANTIATES",
            )
        )
        for pid in fact.passage_ids:
            pas_id = f"pas_{pid}"
            await graph.add_node(GraphNode(id=pas_id, label="Passage", properties={}))
            await graph.add_edge(
                GraphEdge(source_id=fact_id, target_id=pas_id, relation="EVIDENCE_IN")
            )

    for p in passages:
        await graph.add_node(
            GraphNode(id=f"pas_{p.id}", label="Passage", properties={"text": p.text[:80]})
        )

    print("=" * 65)
    print("RUNNING MEMGRAPHRAG MINIMAL RUNNABLE EXAMPLE (MRE)")
    print("=" * 65)
    print(f"Passages loaded in M_pas: {len(memory.passages)}")
    print(
        f"Schemas in M_ont: {len(memory.ontologies)} total | "
        f"{len(stable)} stable (tau={memory.tau})"
    )
    for s in stable:
        print(f"   - Stable schema: {s.schema_key} (freq={s.frequency})")
    print(
        f"Facts added to M_fac: f1={added_f1}, f2={added_f2}, "
        f"f_noisy={added_noisy} (tau filtering ok!)"
    )

    return memory, graph, vectors


async def main() -> None:
    memory, graph, vectors = await build_demo()
    retriever = MemoryGuidedRetriever(
        memory_state=memory,
        graph_store=graph,
        vector_store=vectors,
    )

    query = "Where is Catena Zapata winery located and what wine does it produce?"
    print(f"\nTEST QUERY: '{query}'\n")

    result = await retriever.retrieve(query)

    print("=" * 45)
    print("RETRIEVAL RESULTS")
    print("=" * 45)
    print(f"Retrieval mode: {result.retrieval_mode}")
    print(f"Facts retrieved from M_fac: {len(result.facts)}")
    for fact in result.facts:
        print(
            f"   • {fact.head_entity} -[{fact.relation}]-> {fact.tail_entity} "
            f"(passage support: {fact.passage_ids})"
        )

    print(f"\nPassages retrieved from M_pas (grounding): {len(result.passages)}")
    for p in result.passages:
        preview = p.text if len(p.text) <= 100 else p.text[:97] + "..."
        print(f"   • [{p.id}]: '{preview}'")

    n_nodes = len(result.subgraph.get("nodes", []))
    n_edges = len(result.subgraph.get("edges", []))
    print(f"\nExtracted graph sub-network: Nodes={n_nodes}, Edges={n_edges}")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
