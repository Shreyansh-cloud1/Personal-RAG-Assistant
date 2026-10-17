"""Helpers for passing vectors through raw SQL without a pgvector adapter."""


def to_pgvector_literal(values) -> str:
    """Format a numeric sequence as pgvector's textual input format."""
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"
