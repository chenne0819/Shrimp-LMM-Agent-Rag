from __future__ import annotations

import ast
from typing import Final

_ALLOWED_BINOPS: Final[dict[type[ast.operator], object]] = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}
_ALLOWED_UNARYOPS: Final[dict[type[ast.unaryop], object]] = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def _evaluate_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        operation = _ALLOWED_BINOPS[type(node.op)]
        return operation(
            _evaluate_arithmetic(node.left),
            _evaluate_arithmetic(node.right),
        )

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        operation = _ALLOWED_UNARYOPS[type(node.op)]
        return operation(_evaluate_arithmetic(node.operand))

    raise ValueError("只支援基本四則運算、括號與次方。")


def calculate_math(expression: str) -> str:
    """Evaluate arithmetic expressions for the calculator skill."""
    print(f"[工具] 數學計算: {expression}")
    try:
        parsed = ast.parse(expression, mode="eval")
        return str(_evaluate_arithmetic(parsed))
    except Exception as exc:
        return f"計算失敗: {exc}"
