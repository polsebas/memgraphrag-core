# Informe de Alineación Teórica y Arquitectónica

**Proyecto:** `memgraphrag-core`  
**Referencia:** paper original de MemGraphRAG (memoria global de 3 capas + PPR personalizado + fusión híbrida)  
**Fecha:** 2026-08-12  
**Suite continua:** `tests/test_paper_compliance.py` (también `pytest test_paper_compliance.py` / `python tests/test_paper_compliance.py`)  
**Resultado de la corrida:** 17/17 asserts del paper en verde; `pytest tests/` → 54 passed.

---

## Resumen ejecutivo

El código implementa de forma fiel el modelo de memoria \(M_{pas}, M_{ont}, M_{fac}\), la regla de estabilidad \(f(\text{schema}) \ge \tau\), el PageRank personalizado

\[
p = (1 - \alpha_{ppr}) \cdot p_0 + \alpha_{ppr} \cdot A^{T} p
\]

y la fusión híbrida

\[
S_{final} = \alpha_{hybrid} \cdot \mathrm{Norm}(S_{vec}) + (1 - \alpha_{hybrid}) \cdot \mathrm{Norm}(S_{ppr}).
\]

No quedó ninguna **incongruencia** bloqueante. Las diferencias que siguen son **desviaciones menores** (dirección almacenada de `INSTANTIATES`, PPR bidireccional por defecto, namespace auxiliar `M_ent`).

En esta auditoría se aplicaron parches para: (1) vocabulario canónico de nodos/aristas compartido por ingesta e indexación, (2) grounding del subgrafo contra los pasajes recuperados, (3) escritura explícita de la ecuación de PPR en el motor.

---

## Matriz de requisitos

Leyenda: **Conforme** · **Desviación menor** · **Incongruencia**.

| Eje | Requisito del paper | Evidencia en código | Estado |
|-----|---------------------|---------------------|--------|
| 1 | \(M_{pas}\): almacén de evidencia primaria (pasajes) | `Passage` + `GlobalMemoryState.passages` + namespace vectorial `M_pas` | **Conforme** |
| 1 | \(M_{ont}\): esquemas `(head_type, relation, tail_type)` | `Schema` + `ontologies` + namespace `M_ont` | **Conforme** |
| 1 | \(M_{fac}\): instancias `(head_entity, head_type, relation, tail_entity, tail_type)` | `Fact` + `facts` + namespace `M_fac` | **Conforme** |
| 1 | Separación estricta entre capas | Tres diccionarios distintos; ids de pasaje / schema_key / fact_key no se mezclan | **Conforme** |
| 1 | Namespace vectorial propio por capa | `InMemoryVectorStore` / Chroma: una colección por namespace | **Conforme** |
| 1 | Indexación de cada capa en el grafo | Nodos `Passage`, `Schema`, `Fact` con prefijos `pas_`, `ont_`, `fact_` | **Conforme** |
| 1 | Solo tres capas de memoria | Existe `M_ent` para canonicalizar entidades (no es capa del paper) | **Desviación menor** |
| 2 | \(f(\text{schema}) \ge \tau\) | `Schema.is_stable(tau)` ≡ `frequency >= tau` | **Conforme** |
| 2 | \(M_{fac}\) rechaza hechos con \(f < \tau\) | `GlobalMemoryState.add_fact` retorna `None`; no hay write a vector/grafo | **Conforme** |
| 2 | Transición candidato → esquema estable al alcanzar \(\tau\) | `register_schema_candidate` incrementa `frequency`; `get_stable_schemas()` | **Conforme** |
| 2 | Candidatos inestables no instancian hechos | Cubierto por `add_fact`; el candidato sí se registra en \(M_{ont}\) | **Conforme** |
| 3 | Nodos `Passage`, `Schema`, `Fact` | Constantes `NODE_*` usadas por ingesta e indexación | **Conforme** |
| 3 | `Fact --INSTANTIATES--> Schema` (o \(M_{fac} \leftrightarrow M_{ont}\)) | Se persiste `Schema --INSTANTIATES--> Fact`; PPR recorre en ambos sentidos | **Desviación menor** |
| 3 | `Fact --EVIDENCE_IN--> Passage` (o \(M_{fac} \leftrightarrow M_{pas}\)) | `Fact --EVIDENCE_IN--> Passage` en ambos writers | **Conforme** |
| 3 | Misma nomenclatura/dirección en todo el pipeline | Ingesta e indexación unificados; `SUPPORTED_BY` solo como alias de peso | **Conforme** |
| 4 | \(p_0\) desde similitud vectorial en las 3 capas | `LayeredPPRCalculator.build_seed_weights` | **Conforme** |
| 4 | \(p = (1-\alpha_{ppr}) p_0 + \alpha_{ppr} A^{T} p\) | Power iteration en `NetworkXGraphStore._run_ppr_sync`; residual L1 \(< 10^{-6}\) | **Conforme** |
| 4 | Subgrafo con soporte en pasajes recuperados (grounding) | `_facts_grounded_in_passages` + `_filter_grounded_subgraph` | **Conforme** |
| 4 | Grafo dirigido vs. PPR bidireccional | `bidirectional=True` por defecto (lectura \(\leftrightarrow\) del paper) | **Desviación menor** |
| 5 | \(S_{final} = \alpha_{hybrid}\mathrm{Norm}(S_{vec})+(1-\alpha_{hybrid})\mathrm{Norm}(S_{ppr})\) | `HybridReranker.fuse` | **Conforme** |
| 5 | \(\mathrm{Norm}\) evita dominio por escala | min-max (default) y z-score+logística; test de escalas 0.95 vs 1000 | **Conforme** |

---

## Detalle por eje

### 1. Modelo de memoria de 3 capas

`GlobalMemoryState` mantiene tres almacenes disjuntos:

- \(M_{pas}\): texto fuente (`Passage.id`, `Passage.text`).
- \(M_{ont}\): triples de tipo `(head_type, relation, tail_type)` con frecuencia observada.
- \(M_{fac}\): triples de instancia ligados a `schema_key` y `passage_ids`.

La indexación vectorial usa `LAYER_M_PAS` / `LAYER_M_ONT` / `LAYER_M_FAC`. El grafo materializa la misma partición con labels `Passage` / `Schema` / `Fact`.

**Desviación menor:** el resolver de entidades usa el namespace auxiliar `M_ent`. No participa del PPR ni de la memoria global del paper; es un índice de canonicalización.

### 2. Estabilidad ontológica \(\tau\)

La regla implementada es exactamente \(f(\text{schema}) \ge \tau\):

```python
def is_stable(self, tau: int) -> bool:
    return self.frequency >= tau
```

`add_fact` rechaza de forma categórica si el esquema no existe o no es estable. El candidato pasa a estable cuando `register_schema_candidate` lleva `frequency` a \(\tau\); recién ahí \(M_{fac}\) acepta la instancia.

Los writers (ingesta e indexación) **no** sincronizan al vector store `M_fac` ni crean nodos `Fact` si `add_fact` devolvió `None`.

### 3. Topología jerárquica

Convención canónica del repositorio (unificada en ingesta e indexación):

```text
Schema --INSTANTIATES--> Fact --EVIDENCE_IN--> Passage
```

**Desviación menor de dirección:** el enunciado del paper lista `Fact --INSTANTIATES--> Schema`. El código guarda el sentido *el esquema instancia al hecho*, que es la lectura dual. El PPR corre con `bidirectional=True`, así que la propagación \(M_{fac} \leftrightarrow M_{ont}\) y \(M_{fac} \leftrightarrow M_{pas}\) coincide con la formulación \(\leftrightarrow\) del paper. No hay drift `SUPPORTED_BY` vs `EVIDENCE_IN` entre pipelines.

### 4. Recuperación y Personalized PageRank

El vector de semilla \(p_0\) combina hits de similitud en las tres capas con pesos \(\omega_{ont}, \omega_{fac}, \omega_{pas}\) (default 0.2 / 0.5 / 0.3).

La iteración de potencia implementa

\[
p \leftarrow (1-\alpha)\,p_0 + \alpha\,A^{T}p
\]

con \(A\) la adyacencia ponderada por tipo de arista, normalizada por fila. Los nodos colgantes teleportan según \(p_0\) (completitud estándar para que \(A\) sea estocástica). El test de punto fijo mide residual L1 \(< 10^{-6}\).

El subgrafo de salida solo incluye:

- pasajes recuperados,
- hechos con al menos un `passage_id` en ese conjunto,
- esquemas de esos hechos,
- aristas `INSTANTIATES` / `EVIDENCE_IN` cuyos extremos están en ese conjunto.

### 5. Reranking híbrido

`HybridReranker` aplica \(\mathrm{Norm}\) (min-max por defecto) **antes** de mezclar. El test de dominio de escala usa \(S_{vec}\in\{0.95, 0.10\}\) y \(S_{ppr}\in\{1, 1000\}\): sin normalizar gana el ítem estructural por magnitud; con min-max gana el léxico cuando \(\alpha_{hybrid}=0.6\), como predice la fórmula.

---

## Discrepancias encontradas

| ID | Hallazgo | Severidad | Tratamiento |
|----|----------|-----------|-------------|
| D1 | `INSTANTIATES` persistido como Schema→Fact, no Fact→Schema | Desviación menor | Documentada; PPR bidireccional hace equivalentes ambas lecturas. No se invirtió para no romper el grafo ya unificado. |
| D2 | Subgrafo podía incluir hechos sin evidencia en los pasajes del top-k | Incongruencia (grounding) | **Parcheado** (ver P2). |
| D3 | Vocabulario de aristas/nodos repetido como strings sueltos (riesgo de drift `SUPPORTED_BY`) | Desviación menor | **Parcheado** (ver P1). |
| D4 | Ecuación de PPR correcta pero escrita como \(\alpha A^{T}p + (1-\alpha)p_0\) | Cosmético | **Parcheado** (ver P3) para copiar el orden del paper. |
| D5 | Namespace `M_ent` extra | Desviación menor | Se deja: no es una cuarta capa de memoria. |
| D6 | Candidatos inestables se indexan en `M_ont` (vector + nodo Schema) | Desviación menor | Coherente con acumular \(f(\text{schema})\); no instancian hechos. |
| D7 | `SUPPORTED_BY` sigue en `DEFAULT_RELATION_WEIGHTS` | Desviación menor | Alias de migración para grafos viejos; los writers ya no lo emiten. |

---

## Parches aplicados en esta auditoría

### P1 — Vocabulario canónico del paper (`domain/models.py`)

Se centralizaron namespaces, labels y aristas:

- `LAYER_M_PAS`, `LAYER_M_ONT`, `LAYER_M_FAC`
- `NODE_PASSAGE`, `NODE_SCHEMA`, `NODE_FACT`
- `EDGE_INSTANTIATES`, `EDGE_EVIDENCE_IN` (`EDGE_SUPPORTED_BY` solo como alias de peso)

`IngestionPipeline`, `IndexingOrchestrator`, `LayeredPPRCalculator` y `MemoryGuidedRetriever` consumen esas constantes. Ingesta e indexación escriben la misma jerarquía.

### P2 — Grounding del subgrafo (`retrieval/retriever.py`)

Antes, el subgrafo se armaba con los ids del ranking híbrido aunque un hecho no tuviera evidencia entre los pasajes devueltos.

Ahora:

1. Se adjuntan pasajes de evidencia si hay cupo en `top_k_passages` (comportamiento previo).
2. `_facts_grounded_in_passages` descarta hechos sin soporte en ese conjunto.
3. `_filter_grounded_subgraph` recorta nodos y aristas; `EVIDENCE_IN` solo puede apuntar a pasajes recuperados.

### P3 — Ecuación de PPR (`infrastructure/networkx_graph_store.py`)

La actualización quedó escrita en el orden del paper:

```python
new_scores = (1.0 - alpha) * personalization + alpha * new_scores
```

con comentario explícito \(p = (1-\alpha_{ppr})\,p_0 + \alpha_{ppr}\,A^{T}p\).

---

## Cómo correr la matriz

```bash
pytest tests/test_paper_compliance.py
python tests/test_paper_compliance.py
pytest test_paper_compliance.py          # entrypoint en la raíz del repo
pytest tests/                            # suite completa del paquete
```

Clases de la matriz:

| Clase | Eje |
|-------|-----|
| `TestAxis1ThreeLayerMemory` | Capas, namespaces, labels |
| `TestAxis2OntologicalStability` | \(\tau\), rechazo, transición a estable |
| `TestAxis3HierarchicalGraph` | Aristas canónicas en ingesta e indexación |
| `TestAxis4PersonalizedPageRank` | \(p_0\), punto fijo de PPR, grounding |
| `TestAxis5HybridReranking` | Fórmula \(S_{final}\) y anti-dominancia de escala |

---

*Fin del informe.*
