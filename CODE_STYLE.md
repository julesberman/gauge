# Code Style Guide (for AI Agents)

This document describes how I want my **research code** to look and behave.  
It is written **for AI agents and tools** that generate, edit, or refactor my code.

The main context:

- Single developer (me), personal research code
- Frequent iteration and refactoring
- JAX / Flax / Optax-centric
- Runs on GPUs, trains large neural networks
- Priorities: **readability**, **mathematical clarity**, **low boilerplate**
- It’s OK if things sometimes crash; I’ll rerun the experiment

---

## 1. Overall Philosophy

1. **Readable and math-like first, production-grade second.**
   - Code should look close to the underlying math/concepts.
   - Prefer clear, direct implementations over “clever” or over-engineered ones.

2. **Minimal ceremony, maximal signal.**
   - Avoid boilerplate, patterns, and abstractions that don’t clearly pay for themselves.
   - It’s OK if there’s some repetition if it makes the code easier to read.

3. **Assume a careful user; don’t over-defend.**
   - Don’t add lots of defensive checks or over-robust error handling.
   - It’s acceptable for code to fail fast and loudly when something’s wrong.

4. **Single-user, research-focused.**
   - Optimize for *my* understanding and iteration speed, not for a team of engineers.
   - No need for heavy generalization or ultra-flexible APIs unless explicitly needed.

---

## 2. Architecture & Style

### 2.1 Functional First

- Prefer **functions** over classes and complex objects.
- Functions should:
  - Take explicit inputs (arrays, params, rng keys, configs).
  - Return explicit outputs (arrays, loss values, metrics, updated states).
- Encourage **pure functions** where possible (especially with JAX), but:
  - Side effects for I/O, logging, checkpointing are fine and expected.

### 2.2 Avoid Classes When Possible

- **Default rule:** Do **not** introduce new classes unless there is a strong reason.
- If you think “I should make a class for this,” try first:
  - A set of related functions (e.g., `init_model`, `apply_model`, `train_step`, `eval_step`).
  - A simple config `dict` or a lightweight `dataclass` (only when clearly helpful).
- Object hierarchies, inheritance, and elaborate OOP patterns are **not wanted**.

Exceptions:
- Flax `nn.Module` is fine (and expected) where idiomatic.
- Minimal utility classes are acceptable only if they dramatically simplify usage.

---

## 3. Types, Safety, Logging, and Error Handling

### 3.1 Type Annotations

- Type hints are **optional**, not required.
- Do **not** aggressively add type annotations everywhere.
- If using type hints:
  - Keep them minimal and obvious.
  - Do not introduce complex generic types, protocols, or type-level abstractions.

### 3.2 Safety Checks and Defensive Coding

- **Do not** add large amounts of validation, checking, or defensive code.
- Reasonable minimum:
  - A few targeted `assert` statements when they clarify assumptions (e.g., shape checks in core parts).
  - Simple sanity checks around critical logic if they aid understanding.
- Avoid:
  - Extensive input validation on every function.
  - Nested `try/except` blocks just to keep things running.
  - Complex error messages and custom exceptions.
- If something is wrong, it’s fine for the code to **crash**. I can rerun the experiment.

### 3.3 Logging

- Generally log with print statments
- Print statemenets should generally be one line and inform about the current progress through the experiment
- Print statements should be all lower case
- Its nice to print important results and metrics like accuracy, etc...
- Do a few here and there
---

## 4. JAX / Flax / Optax / GPU-Specific Guidelines

### 4.1 General

- Code is built around:
  - **JAX** for arrays and transformations
  - **Flax** for neural network modules
  - **Optax** for optimizers and training updates
- Assume:
  - Code will run on **GPU** (or TPU).
  - We care about **performance and memory**, but only after readability.

### 4.2 Style in JAX Code

- Prefer **functional JAX style**:
  - Use `jax.jit`, `jax.vmap`, `jax.pmap`, etc., in a clear way.
  - Keep transformation boundaries relatively simple and readable.
- Keep core computations as small, clear functions:
  - `loss_fn(params, batch, rng_key, ...)`
  - `train_step(state, batch, rng_key, ...)`
- Avoid:
  - Overly nested transforms.
  - Clever inlining that obscures what’s happening.

### 4.3 Flax & Optax Usage

- For Flax:
  - Use `nn.Module` as intended.
  - Keep modules relatively small and composable.
- For Optax:
  - Use simple optimizer setups.
  - Write straightforward training loops rather than complex abstractions.

### 4.4 Performance & Memory

- Performance matters, but **not** at the expense of huge complexity.
- Acceptable:
  - Basic performance-aware patterns (e.g., avoiding Python loops when vectorization is natural).
  - Reasonable memory considerations (e.g., not storing huge redundant arrays unnecessarily).
- Not acceptable:
  - Extremely intricate performance tricks that make the code unreadable.
  - Over-optimization that hides the underlying math or logic.

---

## 5. Readability & Documentation

### 5.1 Naming

- Prefer clear, descriptive names:
  - `logits`, `params`, `grads`, `state`, `batch`, `rng_key`, etc.
- Avoid over-abbreviations unless they’re common in ML/JAX contexts.

### 5.2 Comments & Docstrings

- Use **short, focused comments** to explain:
  - The high-level idea.
  - Non-obvious logic.
  - Mathematical meaning (e.g., “this computes KL(p || q)”).
- Docstrings:
  - Optional but welcome for core functions.
  - Keep them brief and helpful, not verbose boilerplate.

### 5.3 Layout

- Keep functions relatively small and focused.
- Group related helper functions together.
- Separate:
  - Model definition
  - Data loading/preprocessing
  - Training loop
  - Evaluation/metrics
  as much as practical.

---

## 6. What to Avoid (Important for Agents)

When generating or editing code, **do not**:

1. **Introduce heavyweight OOP**:
   - No elaborate class hierarchies.
   - No “manager” or “service” classes just for structure.

2. **Add a lot of types or safety code**:
   - Don’t retrofit everything with strict type annotations.
   - Don’t wrap every operation in error handling.
   - Don’t add complex configuration systems or over-generalized abstractions.

3. **Over-engineer for production**:
   - No dependency injection frameworks.
   - No complicated logging frameworks.
   - No “enterprise” patterns.

4. **Create excessive boilerplate**:
   - Avoid large config parsers with many layers.
   - Avoid generic “engine” or “framework” code unless absolutely necessary.

---

## 7. How Agents Should Modify Existing Code

When an AI agent edits or extends this codebase, it should:

1. **Match the existing functional style**:
   - Prefer adding functions over adding classes.
   - Keep inputs/outputs explicit.

2. **Preserve readability and math clarity**:
   - If refactoring, ensure the new version is at least as readable as the old one.
   - Don’t hide core logic inside generic utilities.

3. **Minimize changes unrelated to the intent**:
   - Don’t reformat or reorganize large files unless requested.
   - Avoid large mechanical changes (e.g., blanket type annotation insertion).

4. **Prefer simple, direct solutions**:
   - If there is a choice between a clever abstraction and a straightforward snippet, pick the straightforward one.
   - Keep new dependencies minimal and common (prefer standard JAX/Flax/Optax patterns).

5. **Accept that crashes are OK**:
   - Do not try to over-guard code in ways that add noise.
   - Only add checks when they also aid understanding.

---

## 8. Example Pattern (For Reference)

This kind of structure is good:

```python
def create_model():
    # Define and return a Flax module or model factory
    ...


def loss_fn(params, batch, rng_key, apply_fn):
    """Compute loss for a single batch."""
    inputs, targets = batch
    preds = apply_fn(params, rng=rng_key, inputs=inputs)
    loss = jnp.mean((preds - targets) ** 2)
    return loss


def train_step(state, batch, rng_key):
    """One optimization step."""
    grads = jax.grad(loss_fn)(state.params, batch, rng_key, state.apply_fn)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(
        params=new_params,
        opt_state=new_opt_state,
    )
    return new_state
