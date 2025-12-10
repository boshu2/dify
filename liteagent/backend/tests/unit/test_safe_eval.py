"""
Tests for safe_eval module.
"""
import pytest

from app.core.safe_eval import safe_eval, safe_eval_condition, safe_eval_math


class TestSafeEval:
    """Tests for the safe_eval function."""

    def test_simple_arithmetic(self):
        """Test basic arithmetic expressions."""
        assert safe_eval("1 + 1") == 2
        assert safe_eval("10 - 5") == 5
        assert safe_eval("3 * 4") == 12
        assert safe_eval("10 / 2") == 5

    def test_with_variables(self):
        """Test expressions with variable context."""
        assert safe_eval("x + y", {"x": 5, "y": 3}) == 8
        assert safe_eval("name", {"name": "test"}) == "test"

    def test_comparison(self):
        """Test comparison operations."""
        assert safe_eval("x > 5", {"x": 10}) is True
        assert safe_eval("x < 5", {"x": 10}) is False
        assert safe_eval("x == 10", {"x": 10}) is True

    def test_boolean_operations(self):
        """Test boolean operations."""
        assert safe_eval("x and y", {"x": True, "y": True}) is True
        assert safe_eval("x or y", {"x": False, "y": True}) is True
        assert safe_eval("not x", {"x": False}) is True

    def test_list_access(self):
        """Test list indexing."""
        assert safe_eval("items[0]", {"items": [1, 2, 3]}) == 1
        assert safe_eval("items[1]", {"items": ["a", "b", "c"]}) == "b"

    def test_dict_access(self):
        """Test dictionary access."""
        assert safe_eval("data['key']", {"data": {"key": "value"}}) == "value"

    def test_builtin_functions(self):
        """Test allowed builtin functions."""
        assert safe_eval("len(items)", {"items": [1, 2, 3]}) == 3
        assert safe_eval("str(42)") == "42"
        assert safe_eval("int('42')") == 42
        assert safe_eval("abs(-5)") == 5

    def test_dangerous_code_blocked(self):
        """Test that dangerous code patterns are blocked."""
        # These should raise exceptions
        with pytest.raises(Exception):
            safe_eval("__import__('os')")

        with pytest.raises(Exception):
            safe_eval("open('/etc/passwd')")

        with pytest.raises(Exception):
            safe_eval("eval('1+1')")


class TestSafeEvalCondition:
    """Tests for the safe_eval_condition function."""

    def test_true_conditions(self):
        """Test conditions that evaluate to True."""
        assert safe_eval_condition("True") is True
        assert safe_eval_condition("1 == 1") is True
        assert safe_eval_condition("x > 0", {"x": 5}) is True

    def test_false_conditions(self):
        """Test conditions that evaluate to False."""
        assert safe_eval_condition("False") is False
        assert safe_eval_condition("1 == 2") is False
        assert safe_eval_condition("x < 0", {"x": 5}) is False

    def test_invalid_condition_returns_false(self):
        """Test that invalid conditions return False instead of raising."""
        assert safe_eval_condition("undefined_var") is False
        assert safe_eval_condition("syntax error here") is False

    def test_truthy_values(self):
        """Test truthy value conversion."""
        assert safe_eval_condition("1") is True
        assert safe_eval_condition("'hello'", {}) is True
        assert safe_eval_condition("0") is False
        assert safe_eval_condition("''", {}) is False


class TestSafeEvalMath:
    """Tests for the safe_eval_math function."""

    def test_basic_operations(self):
        """Test basic math operations."""
        assert safe_eval_math("2 + 2") == 4
        assert safe_eval_math("10 - 3") == 7
        assert safe_eval_math("5 * 6") == 30
        assert safe_eval_math("20 / 4") == 5

    def test_complex_expressions(self):
        """Test complex mathematical expressions."""
        assert safe_eval_math("(2 + 3) * 4") == 20
        assert safe_eval_math("2 ** 3") == 8
        assert safe_eval_math("10 % 3") == 1

    def test_math_functions(self):
        """Test mathematical functions."""
        assert safe_eval_math("abs(-5)") == 5
        assert safe_eval_math("round(3.7)") == 4
        assert safe_eval_math("min(1, 2, 3)") == 1
        assert safe_eval_math("max(1, 2, 3)") == 3

    def test_math_constants(self):
        """Test math constants are available."""
        import math
        assert safe_eval_math("pi") == pytest.approx(math.pi)
        assert safe_eval_math("e") == pytest.approx(math.e)

    def test_no_variable_access(self):
        """Test that arbitrary variables cannot be accessed."""
        with pytest.raises(Exception):
            safe_eval_math("x + 1")
