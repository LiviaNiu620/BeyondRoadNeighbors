"""Metadata attached to every run record."""


def result_metadata(evaluation_kind: str) -> dict:
    """Identify the kind of experiment that produced a record (e.g. ``cooperative_outer``)."""
    return {"evaluation_kind": evaluation_kind}
