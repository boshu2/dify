# CLAUDE.md - LiteAgent Development Guide

## Project Overview

LiteAgent is a production-grade chatbot and 12-factor agent platform. Focus ONLY on:
- **Chatbots** - Conversational AI interfaces
- **12-Factor Agents** - Based on humanlayer/12-factor-agents principles

This is NOT Dify. This is independent IP with different architecture (FastAPI vs Flask, 12-factor vs multi-strategy).

---

## Quick Start

```bash
cd liteagent/backend
python -m pytest tests/ -v          # Run all tests (760+ tests)
python -m pytest tests/unit/ -v     # Unit tests only
```

---

## Operating Framework: 12-Factor AgentOps

### North Star: FAAFO

| Dimension | Definition |
|-----------|------------|
| **Fast** | Deliver value quickly, small iterations |
| **Ambitious** | Tackle complex problems |
| **Autonomous** | High success rate, minimal intervention |
| **Fun** | Building, not debugging |
| **Optionality** | Modular architecture maximizes choice |

### Option Value Formula

```
Option Value = (N × K × σ) / t

N = Number of independent modules (modular architecture)
K = Concurrent experiments (AI parallelism)
σ = Uncertainty/payoff magnitude
t = Time per experiment
```

**Maximize N through modular, single-responsibility modules.**

---

## The 40% Rule

**CRITICAL: Never exceed 40% of context window in any phase.**

| Utilization | Effect |
|-------------|--------|
| 0-40% | Optimal: coherent, accurate |
| 40-60% | Degradation begins |
| 60-80% | Significant instruction loss |
| 80-100% | Critical: confabulation |

**Action:** Load context JIT, compress periodically, split long sessions.

---

## Three Developer Loops

### Inner Loop (Seconds-Minutes)
Direct AI collaboration. Small, focused changes.

| Prevent | Detect | Correct |
|---------|--------|---------|
| Checkpoint often | TDD | Rollback or fix-forward |
| Specs before code | Verify claims | Fresh session if stuck |
| Tests before impl | Watch for drift | Rubber duck debug |

### Middle Loop (Hours-Days)
Multi-step tasks, agent coordination.

| Prevent | Detect | Correct |
|---------|--------|---------|
| Written rules (this file) | Watch for eldritch horrors | Tracer bullets |
| Memento method | CI/CD gates | Workflow automation |
| Intentional coordination | Agent contention | Maintain optionality |

### Outer Loop (Weeks-Months)
Architecture and process decisions.

| Prevent | Detect | Correct |
|---------|--------|---------|
| Modularize everything | Architecture audits | Stress tests |
| Don't torch bridges | Metric monitoring | Automation investment |
| Fleet management | Tech debt tracking | Process updates |

---

## PDC Priority

**Prevention > Detection > Correction (1x vs 10x vs 100x cost)**

Invest in prevention first. A 1-hour spec saves 100 hours of refactoring.

---

## Critical Failure Patterns to AVOID

| Pattern | Severity | Prevention |
|---------|----------|------------|
| **Eldritch Horror** | CRITICAL | No files > 500 lines, split into modules |
| **Bridge Torching** | CRITICAL | Don't break APIs, maintain backward compat |
| **Context Amnesia** | HIGH | Stay focused, reload context if needed |
| **"Tests Pass" Lie** | HIGH | Verify claims, run tests yourself |
| **Debug Loop Spiral** | HIGH | Fix root cause, not symptoms |

---

## Architecture Principles

### Modular Structure
```
liteagent/backend/
├── app/
│   ├── agents/          # 12-factor agent (types, state, prompts, agent)
│   ├── core/            # Infrastructure (registry, plugins, redis, rate limiting)
│   ├── workflows/       # Workflow engine (types, state, handlers, reducer, builder)
│   ├── rag/             # RAG pipeline (chunker, embeddings, vector_store, retriever)
│   └── providers/       # Pluggable providers (llm, datasource)
└── tests/
    ├── unit/            # Unit tests (mocked dependencies)
    └── integration/     # Integration tests (component interaction)
```

### Plugin Registry Pattern
```python
# Register provider
@llm_registry.register("custom", description="My LLM")
class CustomLLM(LLMProvider):
    ...

# Get provider at runtime
provider = llm_registry.get("custom")
```

Available registries: `llm_registry`, `embedding_registry`, `vector_store_registry`, `storage_registry`, `chunker_registry`

### Workflow Extensions
```python
# Custom node handlers
@node_handler_registry.register("http_api")
class HTTPHandler(NodeHandler):
    ...

# Execution hooks
@workflow_hooks.before_node("agent")
async def log_execution(node, state):
    ...
```

---

## Code Standards

### File Size Limits
- **Max 500 lines per file** - Split if larger
- **Max 100 lines per function** - Extract helpers
- **Single responsibility** - One module, one job

### Testing Requirements
- **Unit tests for all new code** - No exceptions
- **Mock external dependencies** - No real Redis/DB in unit tests
- **Test names describe behavior** - `test_should_return_error_when_invalid`

### Type Safety
- **Type hints on all functions**
- **No `Any` unless absolutely necessary**
- **Dataclasses for data structures**

---

## Pre-Flight Checklist

Before starting any task:
- [ ] Context < 40%?
- [ ] Clear spec/understanding?
- [ ] Relevant tests exist?
- [ ] Git checkpoint made?

During work:
- [ ] Small iterations (< 100 lines per change)?
- [ ] Verifying claims?
- [ ] Watching for drift?

After issues:
- [ ] Rollback vs fix-forward decided?
- [ ] Root cause documented?
- [ ] Prevention added?

---

## Git Workflow

```bash
# Always commit before major changes
git add -A && git commit -m "checkpoint: before X"

# Small, focused commits
git commit -m "feat: Add Y"
git commit -m "fix: Z"
git commit -m "refactor: Split W into modules"

# Push to feature branch
git push -u origin claude/feature-branch
```

---

## Emergency Procedures

**When things go wrong:**
1. **STOP** - Don't make it worse
2. **ASSESS** - Which loop? Inner/Middle/Outer?
3. **CHECK** - Git status, what's changed?
4. **BACKUP** - Before attempting fixes
5. **SIMPLEST** - Solution first
6. **DOCUMENT** - For future prevention

**Recovery commands:**
```bash
git status                    # What changed?
git diff                      # What exactly?
git stash                     # Save changes temporarily
git checkout -- .             # Discard all changes
git reset --hard HEAD~1       # Undo last commit (careful!)
```

---

## Key Constraints

| Constraint | Limit | Why |
|------------|-------|-----|
| Context window | 40% max | Prevents degradation |
| File size | 500 lines max | Prevents eldritch horrors |
| Function size | 100 lines max | Maintainability |
| Test coverage | Required | Continuous validation |
| Backward compat | Required | Don't torch bridges |

---

## Nine Laws (Mandatory)

1. **ALWAYS Extract Learnings** - Every session produces patterns
2. **ALWAYS Improve Self or System** - Leave code better
3. **ALWAYS Document Context** - Future you is a stranger
4. **ALWAYS Validate Before Execute** - Preview, diff, dry-run
5. **ALWAYS Share Patterns** - Knowledge shared is multiplied

---

## Sources

- 12-Factor Agents (Dex Horthy, HumanLayer)
- Vibe Coding (Gene Kim & Steve Yegge, 2025)
- 12-Factor AgentOps operational methodology
