# Epistemic Agents in Demetrios

## Overview

Day 31 introduces **LLM-Augmented Ontology Evolution & Epistemic Agents** to the Demetrios compiler. This transforms Demetrios from a static type checker into a **living epistemic ecosystem** where:

1. **LLMs generate** new ontology fragments for gaps in L3-L4
2. **Agents autonomously** query, revise, and evolve knowledge
3. **Runtime guards** enforce epistemic integrity beyond compile-time
4. **Evolution calculus** adapts ontologies via Bayesian/MCMC methods

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Demetrios Epistemic Ecosystem                     │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │     LLM      │    │   Epistemic  │    │    Knowledge         │  │
│  │  Integration │───▶│    Agents    │───▶│      Base            │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Ontology    │    │  Evolution   │    │    Type Checker      │  │
│  │  Generator   │    │  Calculus    │    │    Integration       │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. LLM Integration (`compiler/src/llm/`)

Multi-provider LLM client infrastructure supporting:

- **OpenAI** (GPT-4, GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude 3.5 Sonnet, Claude 3 Opus)
- **Ollama** (Llama 3.1, Mistral, CodeLlama)

#### Configuration

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4-turbo"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL="claude-sonnet-4-20250514"

# Ollama (local)
export OLLAMA_HOST="http://127.0.0.1:11434"
export OLLAMA_MODEL="llama3.1:8b"
```

#### Usage

```rust
use demetrios::llm::{LLMClientRegistry, LLMRequest};

// Initialize from environment
let registry = LLMClientRegistry::from_env();

// Query the default provider
let request = LLMRequest::new("Classify 'scaffold' in BFO")
    .with_system("You are an ontology expert")
    .with_temperature(0.2);

let response = registry.default_client()?.query(&request)?;
println!("Response: {}", response.content);
println!("Confidence: {:.2}", response.estimated_confidence());
```

### 2. Ontology Generation (`compiler/src/ontology/llm_gen/`)

LLM-powered ontology fragment generation from natural language.

#### Task Types

| Task | Description | Typical Accuracy |
|------|-------------|------------------|
| Term Extraction | Extract domain terms from text | ~85% |
| Term Typing | Classify terms into BFO categories | ~78% |
| Taxonomy Discovery | Find is-a relationships | ~72% |
| Relation Extraction | Find non-taxonomic relations | ~65% |
| Definition Generation | Create Aristotelian definitions | ~70% |

#### Usage

```rust
use demetrios::ontology::llm_gen::{OntologyGenerator, GenerationConfig};
use demetrios::llm::LLMClientRegistry;

let registry = LLMClientRegistry::from_env();
let config = GenerationConfig::for_domain("biomaterials")
    .with_min_confidence(0.8);

let generator = OntologyGenerator::with_config(registry, config);

let fragment = generator.generate_from_text(
    "Porous scaffolds support bone regeneration through cell migration",
    "biomaterials"
)?;

println!("Generated {} classes", fragment.classes.len());
println!("Average confidence: {:.2}", fragment.average_confidence());
```

### 3. Epistemic Agents (`compiler/src/epistemic/agents/`)

Autonomous agents for knowledge management.

#### Agent Types

| Agent | Purpose | Autonomy Level |
|-------|---------|----------------|
| Query | Answer questions about knowledge | Read-only |
| Revise | Update knowledge with new evidence | Supervised |
| Evolve | Adapt ontologies over time | Semi-autonomous |
| Generate | Create new knowledge from descriptions | LLM-assisted |

#### Usage

```rust
use demetrios::epistemic::agents::{
    EpistemicAgent, AgentTask, KnowledgeBase, Evidence, EvolutionObjective
};

// Create agent with knowledge base
let kb = KnowledgeBase::new();
let agent = EpistemicAgent::new(kb)
    .with_llm(LLMClientRegistry::from_env());

// Submit query task
let task = AgentTask::query("what is a scaffold")
    .with_priority(Priority::High);
let id = agent.submit(task);

// Submit revision task
let evidence = Evidence::new("experiment", "New data supports hypothesis")
    .with_confidence(0.9);
let task = AgentTask::revise("hypothesis", evidence);
agent.submit(task);

// Submit evolution task
let task = AgentTask::evolve("biomaterials", EvolutionObjective::Consistency);
agent.submit(task);

// Execute pending tasks
while agent.pending_count() > 0 {
    if let Some(result) = agent.execute_next()? {
        println!("Task completed: {:?}", result);
    }
}
```

### 4. Evolution Calculus (`compiler/src/epistemic/evolution.rs`)

MCMC-inspired ontology evolution using Metropolis-Hastings acceptance.

#### Algorithm

```
1. Initialize ontology state
2. Calculate initial fitness
3. For each iteration:
   a. Propose mutation (add/remove axiom, adjust confidence, etc.)
   b. Calculate new fitness
   c. Accept with probability:
      - 1.0 if new_fitness > old_fitness
      - exp((new - old) / temperature) otherwise
   d. Cool down temperature
4. Return when converged or max iterations reached
```

#### Mutation Operators

| Operator | Description | Impact |
|----------|-------------|--------|
| AddAxiom | Add new relationship | Medium |
| RemoveAxiom | Remove relationship | Medium |
| AdjustConfidence | Change belief confidence | Low |
| PromoteBelief | Convert belief to fact | High |
| DemoteFact | Convert fact to belief | High |
| MergeClasses | Combine equivalent classes | High |
| SplitClass | Divide class into subclasses | High |

#### Usage

```rust
use demetrios::epistemic::evolution::{
    EvolutionEngine, EvolutionConfig, OntologyState
};

let config = EvolutionConfig {
    max_iterations: 1000,
    initial_temperature: 1.0,
    cooling_rate: 0.995,
    convergence_threshold: 0.001,
    ..Default::default()
};

let mut engine = EvolutionEngine::new(config);
let initial_state = OntologyState::new();

let result = engine.evolve(initial_state);

println!("Iterations: {}", result.iterations);
println!("Converged: {}", result.converged);
println!("Fitness: {:.4} -> {:.4}", result.initial_fitness, result.final_fitness);
println!("Acceptance rate: {:.2}%", result.acceptance_rate * 100.0);
```

### 5. Confidence Estimation (`compiler/src/llm/confidence.rs`)

Epistemic confidence estimation from LLM responses based on linguistic markers.

#### Indicators

| Indicator | Effect on Confidence |
|-----------|---------------------|
| Hedging phrases ("might", "possibly") | Decrease |
| Certainty phrases ("definitely", "always") | Increase (with caution) |
| Uncertainty markers ("I'm not sure") | Strong decrease |
| Source references ("according to") | Increase |
| Reasoning markers ("therefore", "because") | Slight increase |

#### Usage

```rust
use demetrios::llm::confidence::{analyze_confidence, indicators_to_confidence};

let response = "Based on BFO, a scaffold is definitely a material entity (BFO:0000040). 
This is because it has spatial extent and persists through time.";

let indicators = analyze_confidence(response);
let confidence = indicators_to_confidence(&indicators);

println!("Confidence: {:.2}", confidence);
println!("Hedges: {}", indicators.hedge_count);
println!("Certainty: {}", indicators.certainty_count);
println!("Sources: {}", indicators.source_references);
```

## Integration with Type Checking

The epistemic agent system integrates with Demetrios's type checker through the `Knowledge[τ,ε,δ,Φ]` type:

```demetrios
// Epistemic type with LLM-generated ontology binding
let scaffold_mass: Knowledge[
    content = f64,
    τ = (2024, Lab, Experiment),
    ε = (confidence: 0.95, source: Measurement),
    δ = BFO:0000040,  // material entity - validated by agent
    Φ = [sensor → calibration → conversion]
] = measure_mass(sample);

// Agent can revise confidence based on new evidence
revise scaffold_mass with evidence {
    source: "new_experiment",
    confidence: 0.98
};
```

## Prompt Templates

The system includes research-backed prompt templates for ontology tasks:

### Term Extraction
```
Extract ontology terms from the following text. Output as JSON.

Domain context: {domain}
Text: {text}

Output format:
{
  "terms": [
    {"term": "...", "type_hint": "class|property|individual", "confidence": 0.0-1.0}
  ]
}
```

### Term Typing (BFO Classification)
```
Classify the following term according to BFO categories.

Term: {term}
Domain: {domain}
Context: {context}

Think step by step:
1. Does this entity have spatial parts and persist through time?
2. If continuant: Is it independent or dependent?
3. What BFO class best describes it?
```

## Research Foundation

This implementation is based on:

1. **LLMs4OL Challenge 2025** (ISWC) - Hybrid LLM + domain embeddings for ontology learning
2. **arXiv:2411.06528** "Epistemic Integrity in LLMs" - Confidence estimation from linguistic markers
3. **NeurIPS 2025** Embodied Agent Interface - World models and planning for autonomous reasoning
4. **Dapoigny & Barlatier** (Applied Ontology 2012) - Dependent types for ontology representation
5. **CMAM Algorithm** (2025) - CMA-ES + Metropolis for Bayesian inference

## Feature Flag

Enable LLM integration with the `llm` feature:

```toml
[dependencies]
demetrios = { version = "0.31.0", features = ["llm"] }
```

Or build with:

```bash
cargo build --features llm
```

## CLI Usage

```bash
# Run with LLM (requires API key)
OPENAI_API_KEY=... cargo run -- check example.d --with-agents

# Generate ontology from text
ANTHROPIC_API_KEY=... cargo run -- generate-ontology \
    --text "Scaffolds support cell migration" \
    --domain biomaterials

# Evolve ontology
cargo run -- evolve-ontology \
    --target biomaterials \
    --objective consistency \
    --iterations 1000
```

## Future Directions

- **Multi-agent collaboration**: Multiple agents working together on complex tasks
- **Human-in-the-loop**: Interactive approval for high-impact changes
- **Continuous learning**: Agents that improve from feedback
- **Distributed evolution**: Parallel ontology evolution across clusters
- **Fine-tuned models**: Domain-specific LLMs for ontology tasks

---

*"In Demetrios, code doesn't just run — it knows, questions, and evolves."*
