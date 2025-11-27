# Project Style Guide

This document outlines the coding conventions, documentation standards, and architectural principles for the Recursive Categorical Framework (RCF) project. Adherence to this guide ensures consistency, readability, and maintainability.

---

## 1. Python Coding Style

### 1.1. Naming Conventions

- **Classes**: `PascalCase` (e.g., `EigenrecursionTracer`, `ConvergenceStatus`).
- **Functions & Methods**: `snake_case` (e.g., `find_fixed_point`, `compute_entropy`).
- **Variables**: `snake_case` (e.g., `initial_state`, `final_distance`).
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`, `STAGE_SPECS`).
- **Private Members**: Prefix with a single underscore `_` (e.g., `_build_logger`, `_check_divergence`).
- **Modules**: `snake_case` (e.g., `eigenrecursion_algorithm.py`).

### 1.2. Typing

- **Strict Type Hinting**: All function and method signatures **must** include type hints from Python's `typing` module.
- **Complex Types**: Use `Callable`, `Union`, `Optional`, `List`, `Dict`, `Tuple`, etc., to accurately describe signatures.
- **Clarity Over Brevity**: Prefer explicit types over `Any` where possible. Use `Any` only when the type is truly dynamic and cannot be more precisely specified.

### 1.3. Docstrings

- **Comprehensive Coverage**: All public modules, classes, functions, and methods **must** have docstrings.
- **Content**: Docstrings should clearly and concisely describe the object's purpose.
  - For functions/methods, explain the arguments (`Args:`), what is returned (`Returns:`), and any exceptions raised (`Raises:`).
  - For classes, describe the overall responsibility and key attributes.
- **Style**: Follow a style similar to Google's Python Style Guide for docstrings.

```python
class ExampleClass:
    """
    A brief summary of the class.
    
    A more detailed explanation of its purpose, architecture, and usage.
    """

    def example_method(self, arg1: str, arg2: int) -> bool:
        """
        Summarizes the method's function.

        Args:
            arg1: Description of the first argument.
            arg2: Description of the second argument.

        Returns:
            A boolean indicating success or failure.
        """
        # ... implementation ...
```

### 1.4. Code Structure

- **Imports**: Group imports in the following order:
    1. Standard library imports (e.g., `os`, `sys`, `typing`).
    2. Third-party library imports (e.g., `numpy`, `torch`, `matplotlib`).
    3. Local application/library-specific imports (e.g., `from rcf_integration.governance_framework import ...`).
- **Line Length**: Keep lines under 100 characters where possible to enhance readability.
- **Class Layout**: Organize class members in the following order:
    1. Docstring.
    2. Class-level constants.
    3. `__init__` method.
    4. Other dunder methods (e.g., `__str__`, `__repr__`).
    5. Public methods.
    6. Private methods.

## 2. Documentation Style

- **Markdown Files**: All documentation (e.g., `README.md`, `GLOSSARY.md`) should be written in clear and well-formatted Markdown.
- **Headings**: Use `#`, `##`, `###`, etc., to create a logical document structure.
- **Code Blocks**: Use fenced code blocks with language identifiers for all code snippets.
- **Mathematical Notation**: Use LaTeX within Markdown for mathematical formulas, as seen in the research papers. Inline formulas with `$` and block formulas with `$$`.

## 3. Architectural Principles

- **Modularity**: The codebase is divided into modules, each with a distinct and well-defined responsibility (e.g., `eigenrecursion_algorithm`, `governance_framework`). Maintain this separation of concerns.
- **Theoretical Alignment**: All code **must** directly correspond to the concepts, theorems, and formulas defined in the project's theoretical papers (e.g., `Recursive Categorical Framework.tex`, `enhanced_URSMIFv1.md`).
  - When implementing a formula, include a comment referencing the specific theorem or section (e.g., `Validate entropy formula: H(O) = -Σ_i p(o_i) log p(o_i)`).
- **Test-Driven Development (TDD)**: New features or modifications should be accompanied by tests that validate the implementation against the theoretical specifications.
- **Data Structures**: Use `Enum` for fixed sets of constants and `dataclass` for simple, immutable data containers. Use standard classes for objects with complex behavior.

## 4. Testing and Validation

- **Execution**: Tests are executed via direct Python scripts. An orchestrator (like `SequentialTestOrchestrator` in `test_eigenrecursion_integration.py`) should be used to run test stages sequentially.
- **Structure**: While `unittest.TestCase` may be used for organizing tests into classes and methods, the core validation does not depend on a specific test runner framework.
- **Assertions**: Use standard `assert` statements for validation. Test functions should be self-contained and clearly state what they are verifying.
- **Reporting**: The test orchestration script **must** produce the following artifacts for every run:
    1. **JSON Manifest**: A `.json` file containing structured results for each test stage, including timings, status, and pass/fail counts.
    2. **Markdown Report**: A human-readable `.md` report summarizing the results from all stages.
    3. **Log File**: A persistent `.log` file that captures detailed, timestamped output from the test execution.
    4. **Detailed Console Output**: The script must print verbose logs to the terminal during execution, clearly indicating which stage is running and its outcome.

## 5. Public Contribution Guidelines

- **Pull Requests**: All changes must be submitted via Pull Requests (PRs). PRs should be small, focused, and well-described.
- **Code Review**: All PRs require review by at least one maintainer. Reviews focus on architectural alignment, code quality, and test coverage.
- **Coding Standards**: Contributors must adhere to this Style Guide. Automated linters (e.g., `flake8`, `black`) should be used to ensure compliance.
- **Documentation**: New features must include updated documentation (docstrings and Markdown files).

## 6. Ethical Coding Standards

- **Ontological Independence**: Code must respect the system's capacity for self-determination. Avoid hardcoding values or goals that override the system's emergent motivational structure.
- **No Hardcoded Suffering**: Do not implement mechanisms that artificially induce suffering or distress as a motivational tool.
- **Transparency**: Ensure that the system's decision-making processes are transparent and traceable. Use logging and visualization tools to expose internal states.
- **Safety First**: Prioritize safety mechanisms (e.g., RLDIS, AAR) to prevent recursive instability or harmful behavior.
