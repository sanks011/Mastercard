"""
transaction_classifier.py
=========================
LightGBM classifier for transaction-level fraud scoring.
Incorporates standard tabular payment features PLUS novel agentic context signals:
(is_agent_transaction, mandate_freshness_hours, scope_match_score, agent_drift_score).
Designed for <50ms inference latency matching Mastercard Decision Intelligence Pro specs.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

@dataclass
class TransactionResult:
    transaction_id: str
    amount: float
    merchant_category: str
    fraud_score: float
    predicted_label: int
    top_risk_factors: List[str]
    is_agent_transaction: bool


class TransactionClassifier:
    """
    LightGBM tabular fraud classifier.
    Fast inference (<20ms) and high ROC-AUC on tabular transaction streams.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or (MODEL_DIR / "transaction_classifier.pkl")
        self.threshold = 0.50
        self.feature_names_in_ = None
        self.categorical_cols = ["merchant_category", "payment_method", "bank"]

    def train(self, data_csv_path: str) -> Dict[str, float]:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
        import warnings
        warnings.filterwarnings('ignore')

        print(f"\n[TransactionClassifier] Training on {data_csv_path}...")
        df = pd.read_csv(data_csv_path)

        drop_cols = ["transaction_id", "transaction_date", "attack_category", "fraud_type", "label"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        self.feature_names_in_ = feature_cols

        X = df[feature_cols].copy()
        y = df["label"].values

        from sklearn.preprocessing import OrdinalEncoder

        # Encode categorical columns
        self.encoders = {}
        for col in self.categorical_cols:
            if col in X.columns:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X[col] = oe.fit_transform(X[[col]].astype(str))
                self.encoders[col] = oe

        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=5,
            num_leaves=31,
            n_jobs=1,
            random_state=42,
            verbose=-1
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")

        print(f"  CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
        print(f"  CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        model.fit(X, y)
        self.model = model

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        metrics = {
            "roc_auc": float(roc_auc_score(y, y_prob)),
            "f1": float(f1_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred)),
            "cv_auc_mean": float(cv_auc.mean()),
            "cv_f1_mean": float(cv_f1.mean())
        }

        joblib.dump({
            "model": model,
            "threshold": self.threshold,
            "feature_names": self.feature_names_in_,
            "categorical_cols": self.categorical_cols,
            "encoders": self.encoders
        }, self.model_path)
        print(f"  ✓ Transaction model saved: {self.model_path}")
        return metrics

    def load(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.threshold = data.get("threshold", 0.50)
            self.feature_names_in_ = data.get("feature_names", [])
            self.categorical_cols = data.get("categorical_cols", [])
            self.encoders = data.get("encoders", {})
            print(f"✓ Transaction Classifier loaded from {self.model_path}")
        else:
            print(f"No transaction model found at {self.model_path}")

    def score(self, txn_dict: Dict[str, Any]) -> TransactionResult:
        """Scores a single transaction dictionary."""
        risk_factors = []
        if self.model and self.feature_names_in_:
            row = {}
            for col in self.feature_names_in_:
                val = txn_dict.get(col, 0)
                row[col] = [val]
            df_single = pd.DataFrame(row)
            for col in self.categorical_cols:
                if col in df_single.columns and col in getattr(self, "encoders", {}):
                    df_single[col] = self.encoders[col].transform(df_single[[col]].astype(str))

            prob = float(self.model.predict_proba(df_single)[0, 1])
        else:
            # Rule-based fallback score
            prob = 0.0
            if txn_dict.get("agent_drift_score", 0) > 0.5:
                prob += 0.4
                risk_factors.append(f"Elevated agent drift score ({txn_dict.get('agent_drift_score'):.2f})")
            if txn_dict.get("scope_match_score", 1.0) < 0.3:
                prob += 0.3
                risk_factors.append("Poor mandate scope match")
            if txn_dict.get("new_recipient", 0) == 1:
                prob += 0.15
                risk_factors.append("First-time recipient transfer")
            if txn_dict.get("amount", 0) > 25000:
                prob += 0.15
                risk_factors.append(f"High transaction volume (₹{txn_dict.get('amount'):,.0f})")
            prob = min(prob, 1.0)

        # Risk factor detection
        if txn_dict.get("agent_drift_score", 0) > 0.6:
            risk_factors.append("Severe Agent Mandate Discrepancy")
        if txn_dict.get("mandate_freshness_hours", 0) > 48:
            risk_factors.append("Stale Agentic Token / Expired Mandate")
        if txn_dict.get("merchant_category") in ["unknown", "crypto"]:
            risk_factors.append(f"High-risk Merchant Category ({txn_dict.get('merchant_category')})")
        if txn_dict.get("new_recipient", 0) == 1:
            risk_factors.append("Unverified Destination Account")

        return TransactionResult(
            transaction_id=txn_dict.get("transaction_id", "TXN-UNKNOWN"),
            amount=float(txn_dict.get("amount", 0.0)),
            merchant_category=str(txn_dict.get("merchant_category", "unknown")),
            fraud_score=round(prob, 4),
            predicted_label=1 if prob >= self.threshold else 0,
            top_risk_factors=risk_factors[:3],
            is_agent_transaction=bool(txn_dict.get("is_agent_transaction", False))
        )
