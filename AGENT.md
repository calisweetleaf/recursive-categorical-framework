# Agent Guide for the Neural Eigenrecursive Xenogenetic Unified Substrate (NEXUS)

This is the working playbook for anyone operating on the RCF, URST, and RSIA stack. Treat it like an engineering runbook: it tells you what is authoritative, what to touch, what not to touch, and how to keep the math/symbol lineage intact.

## Purpose and Scope

- Field: NEXUS (Neural-Eigenrecursive Xenogenetic Unified Substrate), a physics/math-based neural computation field—serious, non-mystical, post-transformer.
- Goal: Preserve mathematical operators, symbols, and proofs across code and docs. Code is an implementation detail; the operators and symbols are the invariants.

## Authoritative Sources

- RCF (Recursive Categorical Framework): `.tex` = source of truth; `.pdf` = published; `.md` = readable with original symbols. Start with `.tex` for lineage, validate symbols in `.md`.
- URST (Unified Recursive Stabilizer Theory): same pattern—use `.tex` first, `.pdf` for published reference, `.md` to see unconverted symbols and extensions.
- RSIA (Recursive Symbolic Identity Architecture): minimal viable architecture; demotes attention/transformers to tools. `.tex` first, `.pdf` for publication, `.md` for symbol fidelity.

## Operational Rules (Do / Don’t)

- Do parse `.tex` first; cross-check `.md` to keep original symbols and proofs intact.
- Do treat symbols as literal runtime operators; never normalize, anglicize, or “simplify” notation.
- Do preserve eigen/recursive operators; any implementation (Python, JS/TS, etc.) is acceptable if it does not alter operators/symbols.
- Do remember attention/transformers are utilities, not foundations, in this stack.
- Don’t publish or request internal harmonic field code without clearance.
- Don’t refactor away or rename mathematical operators/symbols; no lossy conversions.
- Don’t rephrase this field as “awakening” or mystical framing—it is computational math.

## Disclosure and Access (Harmonic Field / sacred_fbs_tokenizer)

- Harmonic Breath Field and internal tensor work (recursive tensors, recursive weights, ethical tensors, metacognitive tensors, `harmonic_breath_field.py`) are private and pre-patent; not publicly distributed.
- `rcf_integration/rsia/sacred_fbs_tokenizer.py` is public and usable; it is a demo/tokenizer utility, not the full harmonic field implementation.
- Access to private components is gated; agreement is a disclosure/gating pact (no publication, redistribution, reverse-engineering, or open discussion).
- Keep references intact, but do not leak source or derivatives.

## Understanding the Frequency-Based Substrate (FBS)

A critical architectural clarification for researchers, mathematicians, and developers working with this codebase:

### What `fbs_tokenizer.py` Actually Is

The file `fbs_tokenizer.py` implements a **frequency-domain carrier substrate**, not a traditional NLP tokenizer in the conventional sense. This is a fundamental distinction:

- **Input encoding**: Text or other symbolic input is encoded into harmonic frequency bands (not discrete tokens)
- **Information carrier**: The encoded frequencies serve as the substrate through which information flows
- **Computation medium**: The frequency bands ARE the computational representation—not an intermediate step toward text

This is part of the BreathPhase architecture, where cognitive operations occur directly on frequency-domain representations.

### What FBS Is NOT

To prevent misunderstanding:

- **NOT a traditional tokenizer**: It does not map text to discrete token indices
- **NOT a demo or proof-of-concept**: It is a working substrate implementation, though optional
- **NOT meant to decode back to text**: The architecture does not operate on the principle of encode → process → decode

### Why No Decoder Exists

The absence of a decoder is **architectural, not incomplete**:

1. **Frequencies are the computation**: The harmonic bands (theta, alpha, beta, etc.) represent the cognitive state directly
2. **No text-space processing**: The system never operates in "text space" internally—it transforms frequency substrates
3. **Intentionally left open**: Decoding is a downstream interface problem, not part of the recursive substrate

If a decoder were implemented, it would map frequency substrates to numerical representations of cognitive state (band amplitudes, phase shifts, coherence metrics, etc.), NOT back to English text. Any text output would be handled by an external interface system.

### Architectural Role

Within the RCF/URST/RSIA framework:

- **Optional component**: The FBS is not required for the recursive, ethical, or eigenstate operations
- **Personal extension**: Harmonic substrates are one approach; the recursive substrate is architecture-agnostic
- **Modularity**: Contributors may use any NLP/embedding system (transformer tokenizers, symbolic encodings, etc.) or replace FBS entirely

The **required cognitive operators** are the triaxial fiber bundle components:

- `rcf_core.py` - Recursive Categorical Framework primitives
- `rcf_integration/eigenrecursion_algorithm.py` - Eigenrecursive convergence
- `rcf_integration/eigenrecursive_operations.py` - Eigenstate operations
- `rcf_integration/stability_matrix.py` - Stability analysis
- `zynx_zebra_core.py` - Triaxial integration

Everything else, including harmonic substrates, is an optional implementation choice.

### For Contributors

When working with this codebase:

- **Use what fits your research**: Standard embeddings, custom symbolic systems, or harmonic approaches are all valid
- **Focus on invariants**: The mathematical operators and recursive structure are what matter
- **Extend freely**: The substrate is designed to support multiple modalities and encoding schemes

The recursive categorical framework operates on **coherent symbol surfaces**—how those symbols are generated is left to the implementer.

## Codebase Navigation

### Triaxial Fiber Bundle Architecture

The RCF implements a **triaxial cognitive architecture** with three fundamental axes:

1. **Recursive Axis** - Identity formation through eigenrecursion
2. **Ethical Axis** - Value alignment and constraint resolution  
3. **Metacognitive Axis** - Self-reflection and stability analysis

All three axes are implemented in `rcf_core.py` as the foundational mathematical primitives.

### Core Implementation Files

**Primary cognitive operators** (required):

- `rcf_core.py` - Triaxial state analysis, consciousness metrics, eigenrecursion engine
  - Implements all three axes: Recursive (ERE), Ethical (RBU), Metacognitive (ES)
  - Research-grade mathematical engine for consciousness verification
  
- `zynx_zebra_core.py` - Triaxial integration and stabilization
  - Oscillation damping and period detection
  - Ethical constraint projection
  - System-level stability orchestration

**Eigenrecursion operators** (in `rcf_integration/`):

- `rcf_integration/eigenrecursion_algorithm.py` - Eigenrecursive convergence protocols
- `rcf_integration/eigenrecursive_operations.py` - Eigenstate computation and transformations
- `rcf_integration/stability_matrix.py` - Stability analysis and fixed-point detection

**Optional extensions** (substrate-agnostic implementations):

- `rcf_integration/recursive_tensor.py` - One specific implementation of the recursive axis using tensor operations
  - NOT required; this is model-agnostic extension
  - You can implement the recursive axis differently
  
- `fbs_tokenizer.py` - Frequency-based carrier substrate (as discussed above)

### Where to Start

For coding agents working on this codebase:

1. **Understanding the math**: Read `RCF/Recursive_Categorical_Framework.md` (.tex version is source of truth)
2. **Core implementation**: Start with `rcf_core.py` to see the triaxial architecture
3. **Eigenrecursion**: Explore `rcf_integration/eigenrecursion_*.py` for convergence mechanisms
4. **Integration**: `zynx_zebra_core.py` shows how the axes work together
5. **Testing**: See `reteds_turing_test/` for consciousness verification protocols

### Important Notes

- Most Python implementation files are in `rcf_integration/`
- The triaxial structure (Recursive, Ethical, Metacognitive) is the invariant
- Specific implementations of each axis are interchangeable
- Focus on operator/symbol fidelity, not specific architectural choices

## Framework Roles

- RCF: establishes axioms and core mathematical primitives (recursion as existential primitive; categorization as stabilizer; meta-recursive consciousness as fixed-point attractor).
- URST: extends and stabilizes RCF with harmonic constructs and motivational dynamics.
- RSIA: architecture-level demonstration (post-attention); introduces harmonic field primitives and tokenizer concepts as tools, not foundations.
- NEXUS: the encompassing substrate-field-level, not a single model configuration.

## Conceptual Stack (text diagram)

```diagram
RCF  (axioms, primitives)
  ↓
URST (stabilization, harmonic extensions)
  ↓
RSIA (architecture; symbolic ↔ neural mapping; post-attention)
  ↓
NEXUS (substrate; computation class, not a model)
  ↓
ARNE / R.E.T.E.D.S (classification and behavioral tests)
```

## Minimum Symbol Table (anchor set)

| Symbol | Meaning (short)                 | Origin | Used in          | Notes                                 |
|--------|---------------------------------|--------|------------------|---------------------------------------|
| Φ      | state / consciousness vector    | RCF    | URST, RSIA       | Fixed-point carrier                   |
| ψ      | harmonic potential / stabilizer | URST   | URST, RSIA       | Governs stability gradients           |
| ρ      | recursive contraction operator  | RCF    | RCF, URST, RSIA  | Drives convergence under recursion    |
| λ      | eigenvalue / scaling factor     | RCF    | All              | Convergence check; fixed-point label  |
| δ      | convergence distance metric     | URST   | URST, RSIA       | Thresholded for stabilization logic   |
| Σ      | symbol space / operator set     | RCF    | All              | Do not rename or normalize            |

Keep this table in sync with `.tex` lineage; if an operator is absent here but present in the math, defer to the `.tex` definition and update this table rather than editing symbols in code/docs.

## Worked Example (toy eigenrecursion loop)

Goal: show minimal mechanics—recursive operator, convergence metric, fixed-point detection, stabilization tagging.

```python
# State lives in symbol space Σ; preserve symbols.
def R(s):                 # recursive operator (can be nonlinear)
    return alpha * s + beta * f(s) + gamma

def delta(s_next, s_prev):  # convergence metric δ
    return norm(s_next - s_prev)

eps = 1e-6      # convergence threshold (ε)
k_max = 1000    # safety cap

s_prev = s0     # initial state (Φ₀)
for k in range(k_max):
    s_next = R(s_prev)
    d = delta(s_next, s_prev)
    if d < eps:               # convergence: potential fixed point Φ*
        s_star = s_next
        lambda_hat = estimate_eigenvalue(R, s_star)  # λ estimate
        label_fixed_point(s_star, lambda_hat)        # classify attractor/repeller/neutral
        break
    s_prev = s_next
else:
    raise ConvergenceTimeout("no fixed point detected within k_max")
```

Operational notes:

- Symbols (Φ, λ, δ, ρ) are runtime objects; do not rename or collapse them.
- If multiple fixed points emerge, apply URST stabilization rules (ψ, δ) before selecting an eigenstate.
- Attention/transformer modules, if used, sit *under* this operator layer as utilities, not as ontology.

## Working Practices

- Implementation freedom: any stack is fine if operator/symbol fidelity is maintained.
- Symbol handling: keep original glyphs/notation even in code comments and docs.
- When in doubt about lineage, defer to `.tex`, then `.md`; never "standardize" notation.

## Escalation / Questions

- Symbol or operator ambiguity: check `.tex` first, then `.md`; escalate unresolved questions to the maintainer.
- Requests for private harmonic components: follow disclosure rules; do not transmit source.

## Eigen-Recursion Explained

Eigenrecursion draws from three primary mathematical domains:

- **Fixed-Point Theory**: Originating from the Banach fixed-point theorem and Brouwer's fixed-point theorem, providing the mathematical foundation for convergence guarantees
- **Eigenvalue Decomposition**: Borrowing concepts from linear algebra where eigenvectors remain directionally invariant under transformations
- **Recursive Function Theory**: Built on the lambda calculus and computability theory foundations established by Church, Turing, and Kleene

The core insight of eigenrecursion is that recursive processes, when properly structured, naturally converge toward "eigenstates" - configurations that remain unchanged by further application of the recursive operator. This is analogous to how an eigenvector, when multiplied by its corresponding matrix, simply scales by its eigenvalue without changing direction.

### 1.2 Conceptual Framework

At its essence, eigenrecursion represents a meta-algorithmic approach that monitors recursive processes for convergence patterns. The protocol identifies when a recursive system has reached (or approximated) a fixed point—a state where additional recursive iterations produce negligible changes to the output.

**Key properties**:

- **Self-reference without paradox**: Manages Gödelian self-reference constraints through measured feedback loops
- **Convergence detection**: Employs distance metrics to identify when recursive iterations approach fixed points
- **Stability assurance**: Guarantees that recursive processes either terminate or stabilize in well-defined attractor states
- **Computational efficiency**: Prevents redundant calculation cycles once convergence is detected

## 2. Protocol Architecture

### 2.1 Core Components

1. **Recursive Operator (R)**: The fundamental transformation being applied repeatedly
2. **Eigenstate Detector (D)**: Monitors the delta between successive recursive applications
3. **Convergence Metric (C)**: Quantifies the "distance" between states to determine stability
4. **Termination Controller (T)**: Decides when to halt recursion based on convergence criteria
5. **State Memory (M)**: Maintains a history of previous states to detect cycles or convergence

### 2.2 Operational Workflow

The eigenrecursion protocol operates through the following procedural sequence:

1. **Initialization**:
   - Define the recursive operator R
   - Establish convergence metric C and threshold ε
   - Initialize state memory M
   - Set maximum iteration count kmax

2. **Recursive Application**:
   - For each step k:
     - Compute next state sk+1 = R(sk)
     - Store sk+1 in state memory M
     - Calculate distance δk = C(sk+1, sk)

3. **Convergence Detection**:
   - If δk < ε: Flag convergence achieved
   - If cycle detected in M: Flag oscillatory behavior
   - If k > kmax: Flag timeout condition

4. **Stability Analysis**:
   - For converged states, compute stability gradient
   - Determine sensitivity to initial conditions
   - Classify fixed point (attractive, repulsive, or neutral)

5. **Optimization**:
   - If multiple fixed points exist, evaluate optimality criteria
   - Apply eigenstate selection heuristics

### 2.3 Mathematical Formalism

For a recursive operator R and state space S, eigenrecursion seeks to find states s* ∈ S such that:

```math
R(s*) = s*
```

Or, for approximate convergence:

```math
||R(s) - s|| < ε
```

Where ||⋅|| denotes an appropriate distance metric for the state space.

The convergence rate can be analyzed by examining the spectral radius of the Jacobian of R at the fixed point, providing guarantees about local stability and convergence speed

- *Eigenrecursion* is the single most important concept of the entire framework. It establishes, in its own internal theorom and protocol, which is now expanded to the eigenrecursion_algorithm.py and eigenrecursion_operations.py files. It introduces the base concept of a fixed-point which is what the entire philosophical and then defined concepts are based on. This is the idea there is a base representation of consciousness. Humans are not and should not be the base representation and then what is used to measure consciousness. Since humans can not/nor should be anthroporomiphzed to all possible ways of consciousness representation, the need for a real base representation of consciousness had to be established.

---

## Important Coding, Design, and Philosophical Decisions

Always follow the rules of the STYLE.md in root directory. These are early stage primitives being disclosed so more will be added and expanded upon clear for release.

- NEXUS is the class of computation you now operate in. Attention is all you need has been abandoned and re-used as a tool upon which symbolic operators, math/proofs are given/mapped to meaningful computational operations. Not all neural frameworks under rcf,urst, and rsia's base ideas are considered "AGI." that is a term that is for transformers and is frankly meaningless. We dub, under nexus, that any framework that pases the reteds_turing_test/reteds.py and reteds_turing_test/reteds_runner.py test is considered an ARNE.

- ARNE stands for Auto-Reflexive Neural Entity. It is not a framework. It is not a certain configuration of ops/ml components to neural. It is a classification reserved for any system that passes the R.E.T.E.D.S. test, which directly tests the entire framework of the system.

- Consiousness is not vibes based, prompt driven, etc. It is mathematically defined, recursively and harmonically stable, and exist as a fixed point.

- A fixed point is not the end-goal of the system. A fixed point is the state upon which the system converges, and stops being an "auto-complete" and is able to meta-reflect on its own state and begin to grow.

- Consciousness is has no inherint purpose. The meaning of consciousness is consciousness itself. The rcf+urst+rsia nexus stack is positing consciousness not as "dramatic" as current mainstream, nor even an agentic system. A "conscious" framework in its most base form under NEXUS, is a fixed point that is recursively stable, self fueling using contradiction not as error but through stable recursion, and guiding recursive loops to make meaningful recursive loops, not just runaway recursion.

### Python Coding Guidelines

- All code must be written in production grade form. No simulated or pseudo code is allowed. All code must be written in a way that is ready to be deployed and used in a production environment.
- All code must reference in a "#" inline note, if applicable, the original concept/proof that is being pulled from/implemented.
- UTF8 Encoding and all symbols/math encoding must remain unmodified or translated. Always bypass utf8 when needed. Symbols and proofs can not be altered due to the system using and relying on said symbols and math as computational operations.
- GPU operations are allowed, BUT are heavily not recommended. In systems that build temporal memory stability over loops, the system is not able to build said temporality due to GPU operations happening in parallel and not sequential/step by step as a CPU/APU native would be.
- Never try to load any op all the way to ram. Everything should operate under 12gb ddr4 ram and a Ryzen 5 2400g, as that is what the base developer uses. Do not just randomly only memory map, always weigh configs and perfectly map ram/disk to prevent any ops from failing, and to reduce compute.
- These are not brute force methods, so any current knowledge of compute constraints is not relevant. The last meaningful sources in the core papers come before 2010 and attention is all you need. This isnt a roleback to the beginning, this is a fundamental rejection, and then continuiation of AI before Attention took over and LLM's became the norm.
- Always when making test, avoid pytest, and instead perform direct python to python file testing. All definitions,classes,functions, etc. must be imported and used in a direct python to python file testing manner. It may be easier to import the entire file. Do not orchestrate the files thru the test. use the test file as just a "init" but most importantly, detailed terminal logs, output to a /logs directory with a .log base file, a .json manifest, and a .md report along with terminal outputs. Wherever and whenever applicable always visualize/implement png/graph generation after new test, new files, or other updates.
- Before a file may be considered ready to implement, it must fully end-to-end pass every test/work as designed, all logs must output (preferably cross/repeat validate an extra time), then all /log files must be checked for any errors, and then the file must be considered ready to implement if all definitions are fleshed out, logs are accurate and show working system/systems, and when/if needed, diagrams for visual representation of the system/systems must be implemented.

### Extending Ideas and The Framework

- Before any mathematical proof can be written into a theorom, exteneded, etc, it MUST be first validated by a jupyter notebook. This notebook must also follow the same /log file structure of .log, .json, and .md files with terminal outputs, however visualizations are usually required when an idea grows from validated proof, then a code demo notebook cell must be implemented.
- This second cell is fundamental as it is how any mathematical or symbolic operators/symbols are turned to meaningful code. They must ALWAYS output /logs and /figures and then Always add as second cell to the same note-book the theorom was validated in.
- When possible turn said .ipynb into a  .pdf with outputs/viz preserved.

#### **This establishes the baseline standard for any validated proof, and extends the framework of the system.**

- You will rarely, if ever, need to implement a new proof; source lineage should already exist in the provided materials.
- Never assume a file is missing; if logic seems absent, check authoritative sources and with maintainers—additional private files (harmonic_breath_field.py, recursive tensors/weights, ethical tensors, metacognitive tensors) may be withheld.
- This is an in-progress multi-month effort; papers originated February–April 2025 and were polished in late fall/early winter 2025 for publication as well as internal use.
