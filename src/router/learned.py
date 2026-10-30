"""
src/router/learned.py
---------------------
Learned router v2 (FR-07).

Trains a decision tree or logistic regression from a quality vector
(5 features) to an oracle enhancement label.

Usage
-----
    from src.router.learned import LearnedRouter

    router = LearnedRouter(config_path="configs/router.yaml")
    router.fit(oracle_df)                       # pandas DataFrame from oracle.py
    action = router.predict(quality_vector)     # QualityVector → str
    router.save("results/checkpoints/router.pkl")

    # Later:
    router2 = LearnedRouter.load("results/checkpoints/router.pkl")
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.quality.metrics import QualityVector
from src.enhance import ENHANCEMENT_NAMES

FEATURE_COLS = ["brightness", "contrast", "sharpness", "noise", "haze"]


class LearnedRouter:
    """Scikit-learn router trained on oracle labels.

    Parameters
    ----------
    config_path : str | Path
        Path to router.yaml; reads learned.model, max_depth, random_state.
    """

    def __init__(self, config_path: str | Path = "configs/router.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        lcfg = cfg.get("learned", {})

        model_type  = lcfg.get("model", "decision_tree")
        random_state = lcfg.get("random_state", 42)

        if model_type == "decision_tree":
            self._clf = DecisionTreeClassifier(
                max_depth=lcfg.get("max_depth", 5),
                random_state=random_state,
            )
        elif model_type == "logistic_regression":
            self._clf = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown learned router model: {model_type!r}")

        self._fitted = False

    def fit(self, oracle_df: pd.DataFrame) -> "LearnedRouter":
        """Fit on a DataFrame produced by oracle.py.

        Parameters
        ----------
        oracle_df : pd.DataFrame
            Must contain columns in FEATURE_COLS and 'oracle_action'.

        Returns
        -------
        self
        """
        X = oracle_df[FEATURE_COLS].values.astype(np.float32)
        y = oracle_df["oracle_action"].values
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, qv: QualityVector) -> str:
        """Return the predicted enhancement for a single quality vector."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        x = qv.to_array().reshape(1, -1)
        return str(self._clf.predict(x)[0])

    def evaluate(self, oracle_df: pd.DataFrame) -> dict[str, float]:
        """Compute accuracy and regret against oracle labels.

        Regret = mean(oracle_score − router_score_for_chosen_action)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluate().")

        X = oracle_df[FEATURE_COLS].values.astype(np.float32)
        y_true = oracle_df["oracle_action"].values
        y_pred = self._clf.predict(X)

        acc = float(accuracy_score(y_true, y_pred))

        # Regret per frame
        regrets = []
        for _, row in oracle_df.iterrows():
            chosen = row.get(f"score_{y_pred[_]}", np.nan) if False else None
            # Use predicted action's score column
            pass

        # Simplified: iterate rows
        regrets = []
        for pred, (_, row) in zip(y_pred, oracle_df.iterrows()):
            oracle_score = row["oracle_score"]
            router_score = row.get(f"score_{pred}", oracle_score)
            regrets.append(float(oracle_score) - float(router_score))

        return {"accuracy": acc, "mean_regret": float(np.mean(regrets))}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRouter":
        with open(path, "rb") as f:
            return pickle.load(f)
