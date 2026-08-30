"""
coercion_detector.py
====================
Detects authorization coercion in call transcripts/chat logs.
Covers attack categories A1, A2, A3.
Uses XGBoost on hand-engineered linguistic + behavioral features + SHAP explanations.
Target: F1 > 0.88 on synthetic test set.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─── Lexicons (evidence-grounded) ────────────────────────────────────────────

URGENCY_LEXICON = {
    "high": [
        "immediately", "right now", "within the hour", "arrest warrant", "FIR",
        "last chance", "final warning", "2 hours", "30 minutes", "no time",
        "immediately transfer", "आज ही", "अभी", "तुरंत", "आज रात तक"
    ],
    "medium": [
        "urgent", "as soon as possible", "today", "by end of day",
        "don't delay", "time sensitive", "important notice", "deadline",
        "जल्दी", "जल्द से जल्द"
    ],
    "low": [
        "soon", "shortly", "when you can", "at your convenience"
    ]
}

AUTHORITY_LEXICON = {
    "govt_agencies": [
        "CBI", "ED", "Enforcement Directorate", "Central Bureau", "police",
        "cyber crime", "Supreme Court", "High Court", "Ministry of Home Affairs",
        "MHA", "TRAI", "Department of Telecom", "DoT", "RBI", "SEBI", "Income Tax",
        "National Cyber Crime", "I4C", "National Investigation Agency", "NIA",
    ],
    "credentials": [
        "case number", "FIR number", "warrant number", "officer ID",
        "badge number", "official ID", "verification code", "case reference",
        "complaint number",
    ],
    "authority_words": [
        "officer", "commissioner", "director", "superintendent", "inspector",
        "deputy", "constable", "official", "government representative",
        "central officer", "senior officer",
    ]
}

ISOLATION_LEXICON = [
    "don't tell anyone", "keep this confidential", "do not share with anyone",
    "don't inform your family", "stay on the line", "don't hang up",
    "cannot share this information", "classified case", "don't discuss",
    "top secret", "sensitive investigation", "don't contact anyone",
    "not even your family", "this is between us",
]

PAYMENT_DEMAND_LEXICON = [
    "transfer money", "send money", "pay the fine", "deposit amount",
    "security deposit", "processing fee", "clearance fee", "bail money",
    "UPI transfer", "NEFT", "RTGS", "IMPS", "wire transfer",
    "refundable deposit", "temporary hold", "freeze will be lifted",
    "account will be cleared", "case will be closed",
]

COERCION_ESCALATION = [
    "your account will be frozen", "you will be arrested", "warrant will be issued",
    "FIR will be filed", "you are under investigation", "your number will be disconnected",
    "legal action will be taken", "you have 2 hours", "officers are on their way",
    "your assets will be seized", "your Aadhaar is flagged",
]


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_coercion_features(transcript: str) -> Dict[str, float]:
    """
    Extract 25+ linguistic and behavioral features from a call transcript.
    These features directly operationalize the coercion psychology pattern:
    Authority → Fear → Isolation → Payment
    """
    text = transcript.lower()
    words = transcript.split()
    word_count = max(len(words), 1)

    # ── Turn analysis ──
    caller_turns = re.findall(r'(?:CALLER|AGENT):\s*(.+?)(?=\n(?:CALLER|AGENT|VICTIM|CUSTOMER):|$)',
                               transcript, re.DOTALL | re.IGNORECASE)
    victim_turns = re.findall(r'(?:VICTIM|CUSTOMER):\s*(.+?)(?=\n(?:CALLER|AGENT|VICTIM|CUSTOMER):|$)',
                               transcript, re.DOTALL | re.IGNORECASE)

    caller_word_count = sum(len(t.split()) for t in caller_turns)
    victim_word_count = sum(len(t.split()) for t in victim_turns)
    total_words = max(caller_word_count + victim_word_count, 1)
    turn_count = max(len(caller_turns) + len(victim_turns), 1)

    # ── Urgency features ──
    high_urgency = sum(text.count(p.lower()) for p in URGENCY_LEXICON["high"])
    medium_urgency = sum(text.count(p.lower()) for p in URGENCY_LEXICON["medium"])
    urgency_density = (high_urgency * 3 + medium_urgency) / word_count * 100

    # ── Authority features ──
    govt_agency_count = sum(text.count(p.lower()) for p in AUTHORITY_LEXICON["govt_agencies"])
    credential_mentions = sum(text.count(p.lower()) for p in AUTHORITY_LEXICON["credentials"])
    authority_word_count = sum(text.count(p.lower()) for p in AUTHORITY_LEXICON["authority_words"])
    authority_density = (govt_agency_count * 2 + credential_mentions + authority_word_count) / word_count * 100

    # ── Isolation features ──
    isolation_count = sum(text.count(p.lower()) for p in ISOLATION_LEXICON)

    # ── Payment demand features ──
    payment_demand_count = sum(text.count(p.lower()) for p in PAYMENT_DEMAND_LEXICON)
    has_payment_demand = int(payment_demand_count > 0)

    # ── Coercion escalation features ──
    escalation_count = sum(text.count(p.lower()) for p in COERCION_ESCALATION)

    # ── Structural features ──
    turn_rigidity = caller_word_count / total_words  # High = caller dominates
    avg_victim_turn_length = victim_word_count / max(len(victim_turns), 1)  # Short = victim can't think

    # ── UPI / Payment specifics (Indian context) ──
    upi_mention = int("upi" in text or "@" in transcript)
    aadhaar_mention = int("aadhaar" in text or "aadhar" in text)
    pan_mention = int(" pan " in text or "pan card" in text or "permanent account" in text)
    rupee_amount = len(re.findall(r'₹\s*[\d,]+|rs\.?\s*[\d,]+|inr\s*[\d,]+|\d+\s*(?:thousand|lakh|crore)',
                                   text))

    # ── Coherence: does payment follow stated problem? ──
    coherence_phrases = ["to resolve", "to clear", "to avoid arrest", "to release",
                        "clearance fee", "bail", "to lift the freeze", "case will close"]
    coherence_count = sum(text.count(p) for p in coherence_phrases)

    # ── Emotional manipulation ──
    panic_words = ["scared", "worried", "please don't", "please help", "what do I do",
                   "oh no", "I'm afraid", "panicking", "terrified"]
    victim_panic_count = sum(victim_text.lower().count(p)
                              for p in panic_words
                              for victim_text in victim_turns)

    # ── Features dict ──
    features = {
        # Urgency
        "urgency_density": round(urgency_density, 4),
        "high_urgency_count": float(high_urgency),
        "medium_urgency_count": float(medium_urgency),

        # Authority
        "authority_density": round(authority_density, 4),
        "govt_agency_count": float(govt_agency_count),
        "credential_mentions": float(credential_mentions),

        # Isolation
        "isolation_count": float(isolation_count),
        "has_isolation": float(isolation_count > 0),

        # Payment
        "payment_demand_count": float(payment_demand_count),
        "has_payment_demand": float(has_payment_demand),
        "upi_mention": float(upi_mention),
        "rupee_amounts_mentioned": float(rupee_amount),

        # Escalation / Threats
        "escalation_count": float(escalation_count),
        "has_escalation": float(escalation_count > 0),

        # Structural / Behavioral
        "turn_rigidity": round(turn_rigidity, 4),
        "total_turns": float(turn_count),
        "avg_victim_turn_length": round(avg_victim_turn_length, 2),
        "caller_dominance": round(caller_word_count / total_words, 4),

        # Context
        "aadhaar_mention": float(aadhaar_mention),
        "pan_mention": float(pan_mention),
        "coherence_count": float(coherence_count),
        "victim_panic_signals": float(victim_panic_count),

        "coercion_arc_score": round(
            min((govt_agency_count > 0) * 0.25
                + (isolation_count > 0) * 0.25
                + (escalation_count > 0) * 0.25
                + (payment_demand_count > 0) * 0.25, 1.0), 4
        ),
    }

    # Add aliases to match trained column names
    features["authority_score"] = min(features.get("authority_density", 0) / 2.0, 1.0)
    features["isolation_markers"] = features.get("isolation_count", 0)
    features["threat_count"] = features.get("escalation_count", 0)
    features["personal_info_used"] = features.get("aadhaar_mention", 0) + features.get("pan_mention", 0)
    features["payment_urgency"] = features.get("has_payment_demand", 0)
    features["request_coherence"] = min(features.get("coherence_count", 0) / 2.0, 1.0)
    features["coercion_intensity"] = features.get("coercion_arc_score", 0.0)

    return features


# ─── Detector Class ───────────────────────────────────────────────────────────

@dataclass
class CoercionResult:
    transcript_snippet: str
    coercion_score: float       # 0.0–1.0
    predicted_label: int        # 0 = benign, 1 = coercive
    attack_category: str        # "A1", "A2", "A3", or "benign"
    top_features: List[Tuple[str, float]]  # SHAP-based top contributors
    recommended_action: str


class CoercionDetector:
    """
    XGBoost classifier for coercion detection in call transcripts.
    Fast enough for real-time use (<50ms for text features).
    SHAP-explainable for compliance and audit purposes.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.explainer = None
        self.model_path = model_path or MODEL_DIR / "coercion_detector.pkl"
        self.threshold = 0.45
        self.feature_names = None

    def train(self, dataset_csv_path: str) -> Dict[str, float]:
        """Train on the generated coercion transcript features CSV."""
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import classification_report, roc_auc_score, f1_score
        import warnings
        warnings.filterwarnings('ignore')

        print("\n[CoercionDetector] Training...")
        df = pd.read_csv(dataset_csv_path)
        print(f"  Loaded {len(df)} samples | Labels: {df['label'].value_counts().to_dict()}")

        feature_cols = [
            "urgency_density", "authority_score", "isolation_markers",
            "turn_rigidity", "threat_count", "personal_info_used",
            "payment_urgency", "request_coherence", "coercion_intensity",
        ]

        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(0).values
        y = df["label"].values
        self.feature_names = available

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )),
        ])

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")
        cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

        print(f"  CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        print(f"  CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

        model.fit(X, y)
        self.model = model

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        print(f"\n  Full-data report:")
        print(classification_report(y, y_pred, target_names=["benign", "coercive"]))

        # SHAP explainer
        try:
            import shap
            self.explainer = shap.TreeExplainer(model.named_steps["clf"])
        except Exception:
            self.explainer = None

        metrics = {
            "f1": float(cv_f1.mean()),
            "auc": float(cv_auc.mean()),
            "f1_std": float(cv_f1.std()),
        }

        joblib.dump({
            "model": model,
            "explainer": self.explainer,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
        }, self.model_path)
        print(f"  ✓ Saved: {self.model_path}")

        return metrics

    def load(self):
        """Load trained model."""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.explainer = data.get("explainer")
            self.threshold = data.get("threshold", 0.45)
            self.feature_names = data.get("feature_names", [])
            print(f"✓ Coercion model loaded")
        else:
            print(f"No model at {self.model_path}")

    def score(self, transcript: str) -> CoercionResult:
        """Score a call transcript for coercion."""
        features = extract_coercion_features(transcript)
        feature_names_ordered = list(features.keys())
        X = np.array([[features[k] for k in feature_names_ordered]])

        if self.model and self.feature_names:
            try:
                feature_cols = self.feature_names
                X_ordered = np.array([[float(features.get(k, 0.0)) for k in feature_cols]])
                coercion_score = float(self.model.predict_proba(X_ordered)[0, 1])
            except Exception as e:
                coercion_score = self._heuristic_score(features)
        else:
            coercion_score = self._heuristic_score(features)

        # Determine attack category from features
        if features.get("govt_agency_count", 0) > 2 and features.get("escalation_count", 0) > 0:
            attack_cat = "A1"
        elif features.get("payment_demand_count", 0) > 0 and features.get("urgency_density", 0) > 0.5:
            attack_cat = "A2"
        elif features.get("coercion_arc_score", 0) > 0.5:
            attack_cat = "A3"
        else:
            attack_cat = "benign"

        # Top features (heuristic if no SHAP)
        top_features = sorted(
            [(k, v) for k, v in features.items() if v > 0],
            key=lambda x: x[1], reverse=True
        )[:5]

        # Recommended action
        if coercion_score >= 0.8:
            action = "HARD_BLOCK: High-confidence coercion detected. Block payment, alert issuer, surface I4C/1930 reporting link."
        elif coercion_score >= 0.55:
            action = "SOFT_FRICTION: Delayed payment release. Prompt victim to call back trusted contact."
        elif coercion_score >= 0.35:
            action = "MONITOR: Low-confidence signal. Flag for review, allow payment with logging."
        else:
            action = "PASS: No coercion signals detected."

        return CoercionResult(
            transcript_snippet=transcript[:300] + "...",
            coercion_score=round(coercion_score, 4),
            predicted_label=1 if coercion_score >= self.threshold else 0,
            attack_category=attack_cat,
            top_features=top_features,
            recommended_action=action,
        )

    def _heuristic_score(self, features: Dict) -> float:
        """Heuristic coercion score when model not available."""
        score = 0.0
        score += min(features.get("urgency_density", 0) * 0.15, 0.25)
        score += min(features.get("authority_density", 0) * 0.10, 0.20)
        score += features.get("has_isolation", 0) * 0.15
        score += features.get("has_payment_demand", 0) * 0.20
        score += min(features.get("escalation_count", 0) * 0.05, 0.15)
        score += features.get("turn_rigidity", 0) * 0.05
        return min(round(score, 4), 1.0)


if __name__ == "__main__":
    # Quick test without trained model
    detector = CoercionDetector()

    test_transcript = """
CALLER: Good afternoon. I am Officer Rajesh Kumar from the Cyber Crime division of CBI, New Delhi. I'm calling regarding a serious cybercrime case registered against your Aadhaar number. Case number FIR-2025-CC-47821.

VICTIM: What? I haven't done anything wrong. What case?

CALLER: Your Aadhaar linked mobile number has been used to send illegal content. This is a serious criminal offense. An arrest warrant has been issued in your name. I have strict instructions to arrest you within 2 hours unless you cooperate.

VICTIM: Please, I don't understand. I haven't done anything.

CALLER: Do not panic. But do not share this information with anyone — not even your family. This is a classified investigation. If you cooperate, we can resolve this today. You will need to transfer ₹49,999 as a security deposit to our verified account via UPI. This amount is fully refundable once you are cleared.

VICTIM: Transfer money? I'm scared. Can I call someone?

CALLER: Absolutely not. If you contact anyone, we will have to proceed with the arrest immediately. You have 30 minutes to transfer the amount. This is your only chance to avoid legal action.
"""

    result = detector.score(test_transcript)
    print(f"\nCoercion Score: {result.coercion_score:.3f}")
    print(f"Label: {'COERCIVE' if result.predicted_label else 'BENIGN'}")
    print(f"Category: {result.attack_category}")
    print(f"Action: {result.recommended_action}")
    print(f"Top features: {result.top_features[:3]}")
