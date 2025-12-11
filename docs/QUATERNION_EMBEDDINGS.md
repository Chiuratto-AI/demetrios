# Quaternion Knowledge Graph Embeddings

Demetrios provides native support for quaternion embeddings based on the paper:

> **Quaternion Knowledge Graph Embeddings**  
> Zhang, S., et al. (2019)  
> arXiv:1904.10281

## Overview

Quaternion embeddings represent entities and relations in knowledge graphs using 4D hypercomplex numbers. This provides:

- **More degrees of freedom** than complex embeddings (4D vs 2D rotations)
- **Latent inter-dependencies** captured via the Hamilton product
- **Better geometric interpretability** for relational patterns
- **Modeling of symmetry, anti-symmetry, and inversion**

## Core Concept

In quaternion embeddings:
- **Entities** are represented as quaternions: `h, t : quat`
- **Relations** are modeled as rotations: `r : quat`
- **Scoring**: `score(h, r, t) = <h ⊗ r, t>`

Where `⊗` is the Hamilton product and `<,>` is the inner product.

## Operations

### Hamilton Product

The key operation that captures inter-dependencies between all quaternion components:

```d
let transformed = hamilton_product(head, relation);
```

Mathematical definition for `q1 ⊗ q2`:
```
(a1 + b1i + c1j + d1k) * (a2 + b2i + c2j + d2k)
= (a1a2 - b1b2 - c1c2 - d1d2)
+ (a1b2 + b1a2 + c1d2 - d1c2)i
+ (a1c2 - b1d2 + c1a2 + d1b2)j  
+ (a1d2 + b1c2 - c1b2 + d1a2)k
```

### Scoring Function

```d
// Score a triple (head, relation, tail)
let score = quat_score(head, relation, tail);

// Equivalent to:
let transformed = hamilton_product(head, relation);
let score = quat_inner_product(transformed, tail);
```

### Embedding Initialization

```d
// Initialize entity/relation embedding
let entity = quat_embed_init(entity_id);

// Normalize to unit quaternion
let entity_norm = quat_normalize_embed(entity);
```

### Inner Product

```d
// Compute similarity between quaternion embeddings
let similarity = quat_inner_product(q1, q2);
```

### Vector Rotation

```d
// Rotate a 3D vector by quaternion
let rotated = quat_rotate_vec(rotation_quat, vector);
```

## Complete Example

```d
// Knowledge Graph Embedding with Quaternions

fn create_entity_embedding(entity_id: i32) -> quat {
    let q = quat_embed_init(entity_id);
    return quat_normalize_embed(q);
}

fn create_relation_embedding(relation_id: i32) -> quat {
    let q = quat_embed_init(relation_id);
    return quat_normalize_embed(q);
}

fn score_triple(head: quat, relation: quat, tail: quat) -> f32 {
    let transformed = hamilton_product(head, relation);
    return quat_inner_product(transformed, tail);
}

fn main() -> i32 {
    // Create embeddings
    let alice = create_entity_embedding(1);
    let bob = create_entity_embedding(2);
    let company = create_entity_embedding(3);
    
    let works_at = create_relation_embedding(100);
    let knows = create_relation_embedding(101);
    
    // Score triples
    let score1 = score_triple(alice, works_at, company);
    let score2 = score_triple(alice, knows, bob);
    
    return 0;
}
```

## Relational Patterns

Quaternion embeddings can model:

| Pattern | Description | How |
|---------|-------------|-----|
| Symmetry | `r(x,y) ⟺ r(y,x)` | Relation as reflection |
| Anti-symmetry | `r(x,y) ⟹ ¬r(y,x)` | Non-commutative Hamilton product |
| Inversion | `r1(x,y) ⟺ r2(y,x)` | Conjugate relation |
| Composition | `r1(x,y) ∧ r2(y,z) ⟹ r3(x,z)` | Product of rotations |

## SIMD Optimization

The Hamilton product and all quaternion operations are SIMD-optimized using 128-bit vector registers (F32X4). This provides significant speedup for:

- Batch scoring of triples
- Training quaternion embeddings
- Knowledge graph completion tasks

## API Reference

| Function | Signature | Description |
|----------|-----------|-------------|
| `hamilton_product` | `(quat, quat) -> quat` | Hamilton product q1 ⊗ q2 |
| `quat_score` | `(quat, quat, quat) -> f32` | Score triple |
| `quat_embed_init` | `(i32) -> quat` | Initialize embedding |
| `quat_normalize_embed` | `(quat) -> quat` | Normalize to unit |
| `quat_inner_product` | `(quat, quat) -> f32` | Inner product |
| `quat_rotate_vec` | `(quat, vec3) -> vec3` | Rotate vector |

## References

1. Zhang, S., Tay, Y., Yao, L., & Liu, Q. (2019). Quaternion Knowledge Graph Embeddings. *NeurIPS 2019*. arXiv:1904.10281
