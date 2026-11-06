"""
src/router/rule.py
------------------
Rule-based router v1.

Reads thresholds from configs/router.yaml and maps a QualityVector
to one of the six enhancement actions.

Rules are evaluated top-to-bottom; the first matching rule wins.
If no rule fires, the default_action from config is returned.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any

import yaml

from src.quality.metrics import QualityVector
from src.enhance import ENHANCEMENT_NAMES

# Supported comparison operators
_OPS: dict[str, Any] = {
    ">":  operator.gt,
    ">=": operator.ge,
    "<":  operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


class RuleRouter:
    """Rule-based router loaded from a YAML config file.

    Parameters
    ----------
    config_path : str | Path
        Path to router.yaml.
    """

    def __init__(self, config_path: str | Path = "configs/router.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.rules: list[dict] = cfg.get("rules", [])
        self.default_action: str = cfg.get("default_action", "none")

        # Validate action names
        all_actions = {r["action"] for r in self.rules} | {self.default_action}
        for action in all_actions:
            if action not in ENHANCEMENT_NAMES:
                raise ValueError(
                    f"Router config contains unknown action {action!r}. "
                    f"Valid: {ENHANCEMENT_NAMES}"
                )

    def route(self, qv: QualityVector) -> str:
        """Return the enhancement action name for the given quality vector.

        Parameters
        ----------
        qv : QualityVector

        Returns
        -------
        str
            Enhancement name, one of ENHANCEMENT_NAMES.
        """
        qv_dict = qv.to_dict()

        for rule in self.rules:
            conditions = rule.get("conditions", [])
            action = rule["action"]

            if all(
                _OPS[op](qv_dict[field], threshold)
                for field, op, threshold in conditions
            ):
                return action

        return self.default_action

    def route_with_reason(self, qv: QualityVector) -> tuple[str, str]:
        """Return (action, reason_description) for the given quality vector.

        Parameters
        ----------
        qv : QualityVector

        Returns
        -------
        tuple[str, str]
            (action_name, explanation_of_which_rule_fired)
        """
        qv_dict = qv.to_dict()

        for rule in self.rules:
            conditions = rule.get("conditions", [])
            action = rule["action"]

            if all(
                _OPS[op](qv_dict[field], threshold)
                for field, op, threshold in conditions
            ):
                cond_strs = [
                    f"{field} ({qv_dict[field]:.2f}) {op} {threshold}"
                    for field, op, threshold in conditions
                ]
                reason = f"Rule fired: {' and '.join(cond_strs)}"
                return action, reason

        return self.default_action, f"Default action: no rules fired (default: {self.default_action})"
