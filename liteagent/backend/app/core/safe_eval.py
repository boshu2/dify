"""
Safe expression evaluation using simpleeval.

Replaces dangerous eval() calls with a sandboxed evaluator.
"""
import math
from typing import Any

from simpleeval import EvalWithCompoundTypes, DEFAULT_FUNCTIONS


# Safe functions available in expressions
SAFE_FUNCTIONS = {
    **DEFAULT_FUNCTIONS,
    # Type conversions
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "len": len,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    # Math
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    # String
    "lower": lambda s: s.lower() if isinstance(s, str) else s,
    "upper": lambda s: s.upper() if isinstance(s, str) else s,
    "strip": lambda s: s.strip() if isinstance(s, str) else s,
    # Type checking
    "isinstance": isinstance,
    "type": type,
}


def create_evaluator(names: dict[str, Any] | None = None) -> EvalWithCompoundTypes:
    """
    Create a configured safe evaluator.

    Args:
        names: Variable names available in expressions.

    Returns:
        Configured SimpleEval instance.
    """
    evaluator = EvalWithCompoundTypes()
    evaluator.functions = SAFE_FUNCTIONS.copy()

    if names:
        evaluator.names = names.copy()
    else:
        evaluator.names = {}

    return evaluator


def safe_eval(expression: str, context: dict[str, Any] | None = None) -> Any:
    """
    Safely evaluate a Python expression.

    This is a drop-in replacement for eval() that uses simpleeval
    to prevent code injection attacks.

    Args:
        expression: The expression to evaluate.
        context: Variables available in the expression.

    Returns:
        The result of the expression.

    Raises:
        Exception: If evaluation fails.
    """
    evaluator = create_evaluator(context)
    return evaluator.eval(expression)


def safe_eval_condition(condition: str, context: dict[str, Any] | None = None) -> bool:
    """
    Safely evaluate a boolean condition.

    Args:
        condition: The condition expression to evaluate.
        context: Variables available in the expression.

    Returns:
        Boolean result of the condition.
    """
    try:
        result = safe_eval(condition, context)
        return bool(result)
    except Exception:
        return False


def safe_eval_math(expression: str) -> float | int:
    """
    Safely evaluate a mathematical expression.

    Only allows numbers and basic math operators.

    Args:
        expression: Mathematical expression like "2 + 2" or "10 * 5".

    Returns:
        The numeric result.

    Raises:
        ValueError: If expression contains invalid characters.
    """
    from simpleeval import SimpleEval

    # For math expressions, use a minimal evaluator with no variables
    evaluator = SimpleEval()
    evaluator.functions = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
    }
    evaluator.names = {
        "pi": math.pi,
        "e": math.e,
    }
    return evaluator.eval(expression)
