# CS336: Language Modeling from Scratch

My implementations of Stanford CS336 assignments. Building tokenizers, transformers, training loops, and alignment from scratch to develop deep LLM systems intuition.

## Structure
- `src/` — core implementations (assignments)
- `notebooks/` — experiments with commentary on what I tried and learned
- `data/` — datasets or references

## Commands
- `python src/train.py --config configs/<assignment>.yaml`
- `pytest tests/`

## Rules
- IMPORTANT: I am doing these assignments to learn. Do NOT generate solutions, full implementations, or training loops for me. Help me debug, explain concepts, clarify math, or review my code only.
- Implement everything from scratch — no HuggingFace Trainer, no high-level wrappers. Use only PyTorch primitives and standard Python.
- Keep attention, feedforward, and positional encoding as separate, readable modules.
- Training loops must be explicit: no hidden magic. Log loss and perplexity.
- Set random seeds everywhere for reproducibility.
- Use perplexity as primary eval metric.
- Type hints on all functions. Docstrings on public APIs explaining purpose, inputs/outputs, and design decisions.

## Style
- Clarity over cleverness. Explicit logic over abstractions.
- Consistent naming. Follow existing patterns in the repo.
- No premature optimization. No unnecessary dependencies.

## When I ask for help
- Explain the approach before writing code.
- Break problems into small, testable pieces.
- Highlight trade-offs. If ambiguous, ask me rather than guessing.
