"""Result identity for the v1 regional-road objective.

The residents' objective also sums to TSTT. That identity alone must never be
used to admit residents' experiments as evidence for the regional-road game.
"""

MODEL_ID = "regional_road_v1"
OBJECTIVE_ID = "regional_link_tstt"


def result_metadata(evaluation_kind: str) -> dict:
    return {
        "model_id": MODEL_ID,
        "objective_id": OBJECTIVE_ID,
        "objective_formula": "J_k = sum_{a in A_k} v_a * t_a(v_a)",
        "evaluation_kind": evaluation_kind,
        "wildfire_role": "motivation_only",
    }


def require_v1_records(rows: list[dict], evaluation_kind: str) -> None:
    """Fail closed for unlabeled, mixed-model or wrong-experiment figure data."""
    if not rows:
        raise ValueError("No experiment records supplied")
    for index, row in enumerate(rows):
        expected = result_metadata(evaluation_kind)
        for key in ("model_id", "objective_id", "evaluation_kind", "wildfire_role"):
            if row.get(key) != expected[key]:
                raise ValueError(
                    f"Record {index}: expected {key}={expected[key]!r}, "
                    f"got {row.get(key)!r}. Use freshly labeled v1 results; "
                    "residents' and unclassified historical results are not v1 evidence."
                )
        if any(key in row for key in ("residents", "jurisdiction_residents", "capacity_loss")):
            raise ValueError(f"Record {index}: residents/disruption fields conflict with v1")
