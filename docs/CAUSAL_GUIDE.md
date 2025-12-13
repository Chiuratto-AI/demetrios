# Causal Inference in Demetrios

## Overview

Demetrios provides **native causal reasoning** as a language feature. Rather than relegating causal inference to libraries, it's integrated into the type system, allowing compile-time causal checking and automated do-calculus.

## The Causal Hierarchy

Pearl's causal hierarchy has three levels:

1. **Association** (P): What does the data show?
2. **Intervention** (do): What happens if we act?
3. **Counterfactuals** (⟨⟩): What would have happened?

Demetrios supports all three natively.

## Level 1: Causal DAGs

A **causal directed acyclic graph (DAG)** represents assumed causal relationships.

### Specifying a DAG

```d
// UC Berkeley admissions example
causal model admissions {
    nodes: [Gender, Department, Admission];

    // Causal edges
    Gender -> Department;
    Gender -> Admission;
    Department -> Admission;
}
```

### Reading a DAG

- **X → Y** means X causally influences Y
- **Paths** represent causal pathways
- **Backdoor paths** are confounding (non-causal) paths

Example with confounder:

```
    Ability
      / \
     /   \
    v     v
Education—>Income
```

Path from Education to Income:
- **Direct path**: Education → Income (causal effect)
- **Backdoor path**: Education ← Ability → Income (confounding)

## Level 2: Interventions (do-calculus)

The **do-operator** represents external intervention, removing confounding.

### Observational vs Interventional

```d
// Observational: P(Income | Education)
// Includes confounding from Ability
let p_income_given_education = observe(income, education);

// Interventional: P(Income | do(Education))
// Breaks Education ← Ability path
let p_income_given_do = do(model, Education = education_value);
```

### Identifying Causal Effects

Use the **backdoor criterion** to identify when a causal effect is identifiable:

```d
// Is the effect of Education on Income identifiable?
// Need to block all backdoor paths

// Backdoor paths through Ability
// Solution: Adjust (stratify) by Ability

let ate = estimate_ate(
    data,
    treatment = "Education",
    outcome = "Income",
    adjust = ["Ability"]  // Block backdoor paths
);
```

### Three Rules of do-calculus

#### Rule 1: Ignoring Observations

If Y is independent of X given Z and W (in intervened graph):
```
P(y | do(x), z, w) = P(y | do(x), w)
```

#### Rule 2: Ignoring Interventions

If there's no causal path from X to Y given Z:
```
P(y | do(x), z) = P(y | z)
```

#### Rule 3: Causal Sufficiency

If all confounders are measured and adjusted:
```
P(y | do(x)) = Σ_z P(y | x, z) P(z)
```

## Level 3: Counterfactuals

**Counterfactuals** reason about what would have happened under a different intervention.

### Structure

```
Observed: Alice without education earns $30k
Counterfactual: If Alice had education, she'd earn $30k + $15k = $45k
```

### Three-Step Process

1. **Abduction**: Infer unobserved variables from observation
2. **Action**: Perform intervention in causal model
3. **Prediction**: Predict outcome with new variables

```d
fn counterfactual_prediction(
    model: CausalModel,
    observed: Evidence,     // What we saw
    intervention: Intervention,  // What we'd change
    query: String           // What we want to predict
) -> f64 {
    // Step 1: Infer latent variables from observation
    let latent = abduct(model, observed);

    // Step 2: Apply intervention to model
    let intervened_model = do(model, intervention);

    // Step 3: Predict with same latent variables
    let prediction = predict(intervened_model, latent, query);

    return prediction;
}
```

## Applications

### 1. Simpson's Paradox Resolution

**Paradox**: An association reverses when stratifying by a confounder.

```d
// Marginal (unadjusted):
// Males: 60% admission rate
// Females: 50% admission rate
// Appears males favored

// Stratified by Department:
// Males: 35% (applied mostly to selective departments)
// Females: 45% (applied mostly to less selective departments)
// Actually females have higher rates per department!

// Causal solution: Department is a confounder
// P(Admission | Gender) ≠ P(Admission | do(Gender))

let confounders = dag.backdoor_criterion("Gender", "Admission");
// Returns: ["Department"]

let true_effect = estimate_ate(data, "Gender", "Admission", adjust=confounders);
```

### 2. Treatment Effect Heterogeneity

Estimate effects for subgroups:

```d
fn conditional_ate(
    data: DataFrame,
    treatment: String,
    outcome: String,
    subgroup_var: String,
    subgroup_values: [Value]
) -> [f64] {
    let mut effects = [];

    for value in subgroup_values {
        let subgroup_data = data.filter(subgroup_var == value);
        let effect = estimate_ate(subgroup_data, treatment, outcome);
        effects.push(effect);
    }

    return effects;
}

// Example: Does education effect depend on initial ability?
let effects_by_ability = conditional_ate(
    data,
    "Education",
    "Income",
    "Ability",
    ["Low", "Medium", "High"]
);
```

### 3. Mediation Analysis

Decompose effect into direct and indirect:

```
    Treatment
       |
       |---> Mediator ---> Outcome
       |                  /
       |_________________/

Total effect = Direct + Indirect (through mediator)
```

```d
fn mediation_analysis(
    model: CausalModel,
    treatment: String,
    mediator: String,
    outcome: String
) -> MediationResult {
    // Total effect: treatment → outcome
    let total_effect = estimate_ate(model, treatment, outcome);

    // Direct effect: treatment → outcome (fixing mediator)
    let direct_effect = estimate_ate(
        model.fix_mediator(mediator),
        treatment,
        outcome
    );

    // Indirect effect: treatment → mediator → outcome
    let indirect_effect = total_effect - direct_effect;

    return MediationResult {
        total_effect,
        direct_effect,
        indirect_effect,
        proportion_mediated: indirect_effect / total_effect
    };
}
```

### 4. Instrumental Variables

When you have unmeasured confounders, use an instrument:

```
    Confounder (unmeasured)
         |
         v
Instrument ----> Treatment ----> Outcome
                     ^
                     |
              (arrow from confounder)
```

Instrument must:
- Affect outcome only through treatment
- Be independent of confounder

```d
fn iv_estimate(
    data: DataFrame,
    instrument: String,
    treatment: String,
    outcome: String
) -> f64 {
    // IV estimator: IV assumes no unmeasured confounding
    // β_IV = Cov(Instrument, Outcome) / Cov(Instrument, Treatment)

    let cov_iz = covariance(data[instrument], data[outcome]);
    let cov_it = covariance(data[instrument], data[treatment]);

    return cov_iz / cov_it;
}
```

### 5. Policy Evaluation

Compare potential policies:

```d
fn evaluate_policy(
    model: CausalModel,
    policy: Policy,
    outcome: String,
    target_population: String
) -> f64 {
    // What would be the average outcome if we implemented this policy?
    // P(outcome | do(policy))

    let outcome_dist = do(model, policy);
    let average_outcome = mean(outcome_dist[target_population]);

    return average_outcome;
}

// Compare two education policies
let policy_a = Policy::IncreaseAccess { increase_percentage: 20 };
let policy_b = Policy::ImproveQuality { improve_rating: 2 };

let effect_a = evaluate_policy(model, policy_a, "Income", "Low-income");
let effect_b = evaluate_policy(model, policy_b, "Income", "Low-income");

if effect_a > effect_b {
    println("Policy A is more effective");
} else {
    println("Policy B is more effective");
}
```

## Type System Integration

Causal types prevent incorrect reasoning:

```d
// Error: Can't compare observational and causal probabilities
let p_obs = observe(income, education);  // Type: P(Income | Education)
let p_causal = do(model, education);     // Type: P(Income | do(Education))
let invalid = p_obs + p_causal;          // ✗ TYPE ERROR

// Correct: Convert to same level
let p_causal_equiv = convert_to_causal(p_obs, model);
let valid = p_causal_equiv + p_causal;   // ✓ OK
```

## Common Pitfalls

### 1. Confusing Association with Causation

```d
// ✗ WRONG: Observing correlation means causation
let correlation = pearson(smoking, lung_cancer);
println("Smoking causes cancer: {}", correlation > 0);

// ✓ RIGHT: Use causal model and do-calculus
let model = causal_model;
let effect = estimate_ate(data, "Smoking", "LungCancer", adjust=confounders);
println("Causal effect of smoking: {}", effect);
```

### 2. Forgetting Unmeasured Confounders

```d
// ✗ WRONG: Assuming no unmeasured confounders
let effect = estimate_ate(data, "Treatment", "Outcome");

// ✓ RIGHT: Document assumptions
let model = CausalModel {
    // Assume these are the only confounders:
    confounders: ["Age", "Baseline"],
    // Note: Cost/access not included - potential bias
    unknown_confounders: ["Cost", "Healthcare Access"]
};

let effect = estimate_ate(data, "Treatment", "Outcome", adjust=model.confounders);
println("Note: Estimates may be biased if unobserved confounders exist");
```

### 3. Over-adjusting (Collider Bias)

```d
// ✗ WRONG: Adjusting for colliders opens backdoor paths
//   X → Collider ← Z
//   Adjusting for Collider induces spurious X-Z correlation

let effect = estimate_ate(data, "X", "Y", adjust=["Collider"]);

// ✓ RIGHT: Only adjust for confounders on open paths
let confounders = dag.backdoor_criterion("X", "Y");
let effect = estimate_ate(data, "X", "Y", adjust=confounders);
```

## Identifiability

Check if a causal effect is identifiable from observational data:

```d
fn is_identifiable(
    dag: CausalDAG,
    treatment: String,
    outcome: String
) -> (bool, Option<Vec<String>>) {
    // Check backdoor criterion
    let confounders = dag.backdoor_criterion(treatment, outcome);

    // If all confounders are measurable, effect is identifiable
    if confounders.is_empty() {
        return (true, None);
    }

    // If confounders contain unmeasured variables, not identifiable
    if confounders.contains_unmeasured() {
        return (false, Some(confounders));
    }

    return (true, Some(confounders));
}
```

## See Also

- [AUTODIFF_GUIDE.md](AUTODIFF_GUIDE.md) - Gradient computation
- [UNITS_GUIDE.md](UNITS_GUIDE.md) - Physical units
- [Examples](../examples/) - wave2_* examples
- Pearl, J. (2009). Causality: Models, Reasoning and Inference
- Angrist & Pischke (2009). Mostly Harmless Econometrics
