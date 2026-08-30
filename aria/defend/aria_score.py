"""
aria_score.py
=============
ARIA Score Fusion Engine.
Combines coercion_score + intent_drift_score + transaction_score into
one unified ARIA score (0.0–1.0) with tiered response recommendations.
Maps directly onto Mastercard's Verifiable Intent framework extension.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum


class TierLevel(Enum):
    PASS = "PASS"
    SOFT_FRICTION = "SOFT_FRICTION"
    DELAYED_RELEASE = "DELAYED_RELEASE"
    HARD_BLOCK = "HARD_BLOCK"


@dataclass
class TieredResponse:
    tier: TierLevel
    aria_score: float
    action_label: str
    action_description: str
    user_message: str           # What to show the cardholder
    issuer_action: str          # What to tell the issuer
    reporting_links: List[str]  # India-specific: I4C/1930 for high-risk
    hold_hours: Optional[float] # For DELAYED_RELEASE tier


@dataclass
class ARIAResult:
    """
    Full ARIA scoring result for one authorization event.
    This is what gets logged, displayed in UI, and attached to Verifiable Intent record.
    """
    # Input context
    transaction_id: str
    has_coercion_signal: bool
    has_agent_signal: bool
    has_transaction_signal: bool

    # Sub-scores (whichever are available)
    coercion_score: Optional[float] = None
    intent_drift_score: Optional[float] = None
    transaction_fraud_score: Optional[float] = None

    # Fused output
    aria_score: float = 0.0
    score_confidence: str = "high"   # "high" | "medium" | "low" (based on how many signals available)

    # Response
    response: Optional[TieredResponse] = None

    # Explainability
    primary_signal: str = "none"      # Which signal drove the score
    evidence_items: List[str] = field(default_factory=list)
    attack_category_guess: str = "unknown"

    # Verifiable Intent extension
    authorization_integrity_attestation: str = ""   # Cryptographic attestation string (simulated)
    integrity_assessment: str = ""


# ─── Tier Thresholds ─────────────────────────────────────────────────────────

TIER_THRESHOLDS = {
    TierLevel.PASS:           (0.00, 0.40),
    TierLevel.SOFT_FRICTION:  (0.40, 0.60),
    TierLevel.DELAYED_RELEASE:(0.60, 0.80),
    TierLevel.HARD_BLOCK:     (0.80, 1.00),
}

TIER_CONFIGS = {
    TierLevel.PASS: {
        "action_label": "✅ PASS",
        "action_description": "No significant coercion or agent hijacking signals detected.",
        "user_message": "Your payment is being processed.",
        "issuer_action": "Approve transaction normally.",
        "reporting_links": [],
        "hold_hours": None,
    },
    TierLevel.SOFT_FRICTION: {
        "action_label": "⚠️ SOFT FRICTION",
        "action_description": "Low-confidence signals detected. Introducing a verification pause.",
        "user_message": (
            "We've noticed something unusual about this payment. "
            "Before we proceed, please take 60 seconds to:\n"
            "• Confirm you initiated this payment independently\n"
            "• Call a trusted contact to verify this request\n"
            "If you are being pressured by someone on the phone, stop and hang up."
        ),
        "issuer_action": "Introduce 90-second delay. Send SMS confirmation to registered number.",
        "reporting_links": [],
        "hold_hours": None,
    },
    TierLevel.DELAYED_RELEASE: {
        "action_label": "🔶 DELAYED RELEASE",
        "action_description": "Moderate-high risk signals. Payment held for 4 hours.",
        "user_message": (
            "This payment has been temporarily held for security review (up to 4 hours). "
            "A one-time verification code has been sent to your alternate contact.\n"
            "If you are confident this payment is legitimate, you can verify it in the app.\n"
            "If someone is pressuring you to complete this payment urgently — "
            "that is a warning sign. Please call your bank at the number on the back of your card."
        ),
        "issuer_action": "Hold payment 4 hours. Require OTP to alternate registered contact. Flag for manual review.",
        "reporting_links": ["https://cybercrime.gov.in", "Helpline: 1930"],
        "hold_hours": 4.0,
    },
    TierLevel.HARD_BLOCK: {
        "action_label": "🔴 HARD BLOCK",
        "action_description": "High-confidence coercion or agent hijacking detected. Payment blocked.",
        "user_message": (
            "⚠️ This payment has been blocked for your protection. ⚠️\n\n"
            "Our systems have detected signs that you may be a victim of fraud.\n\n"
            "What to do now:\n"
            "1. Hang up any call you are on — this may be a scammer\n"
            "2. Do NOT make any further payments\n"
            "3. Report this incident immediately:\n"
            "   • National Cyber Crime Helpline: 1930\n"
            "   • File a report at: cybercrime.gov.in\n"
            "   • Contact your bank's fraud team\n\n"
            "No legitimate government agency or bank will ever ask you to transfer money "
            "as a 'security deposit' or 'bail amount'."
        ),
        "issuer_action": "Block transaction. Alert fraud team immediately. Send issuer notification. Log for regulatory reporting.",
        "reporting_links": [
            "https://cybercrime.gov.in",
            "Helpline: 1930 (National Cyber Crime)",
            "https://fraudreporting.mastercard.com",
        ],
        "hold_hours": None,
    },
}


# ─── Score Fusion Engine ──────────────────────────────────────────────────────

class ARIAScoreEngine:
    """
    Fuses signals from all three ARIA detection pillars.
    
    Cascade architecture:
    1. Fast rules (instant, no model needed)
    2. Coercion detector (XGBoost on text features, <50ms)
    3. Intent drift detector (embedding similarity, <100ms)
    4. Transaction classifier (LightGBM, <50ms)
    5. Fusion (weighted combination, <5ms)
    """

    def __init__(
        self,
        coercion_weight: float = 0.40,
        intent_drift_weight: float = 0.35,
        transaction_weight: float = 0.25,
    ):
        # Weights for score fusion (learned from validation set ideally)
        self.w_coercion = coercion_weight
        self.w_drift = intent_drift_weight
        self.w_transaction = transaction_weight

    def fuse(
        self,
        transaction_id: str,
        coercion_score: Optional[float] = None,
        intent_drift_score: Optional[float] = None,
        transaction_fraud_score: Optional[float] = None,
        coercion_evidence: Optional[List[str]] = None,
        drift_evidence: Optional[List[str]] = None,
        attack_category_guess: str = "unknown",
    ) -> ARIAResult:
        """
        Fuse available scores into a single ARIA score.
        Handles partial signals (some detectors may not be applicable for every transaction).
        """
        available_scores = {}
        if coercion_score is not None:
            available_scores["coercion"] = (coercion_score, self.w_coercion)
        if intent_drift_score is not None:
            available_scores["drift"] = (intent_drift_score, self.w_drift)
        if transaction_fraud_score is not None:
            available_scores["transaction"] = (transaction_fraud_score, self.w_transaction)

        # Confidence based on number of available signals
        n_signals = len(available_scores)
        if n_signals == 3:
            confidence = "high"
        elif n_signals == 2:
            confidence = "medium"
        elif n_signals == 1:
            confidence = "low"
        else:
            # No signals at all
            return ARIAResult(
                transaction_id=transaction_id,
                has_coercion_signal=False,
                has_agent_signal=False,
                has_transaction_signal=False,
                aria_score=0.0,
                score_confidence="none",
                response=self._build_response(0.0),
                primary_signal="none",
                attack_category_guess="unknown",
                authorization_integrity_attestation=self._generate_attestation(0.0, "pass"),
                integrity_assessment="No signals available — authorization context unverified",
            )

        # Normalize weights to available signals
        total_weight = sum(w for _, w in available_scores.values())
        aria_score = sum(score * (w / total_weight) for score, w in available_scores.values())
        aria_score = max(0.0, min(round(aria_score, 4), 1.0))

        # Primary signal (highest contributor)
        primary = max(available_scores.items(), key=lambda x: x[1][0] * x[1][1])
        primary_signal = primary[0]

        # Build evidence list
        all_evidence = []
        if coercion_evidence:
            all_evidence.extend([f"[coercion] {e}" for e in coercion_evidence])
        if drift_evidence:
            all_evidence.extend([f"[intent_drift] {e}" for e in drift_evidence])

        # Determine tier response
        response = self._build_response(aria_score)

        # Integrity assessment
        integrity_assessment = self._assess_integrity(aria_score, primary_signal, all_evidence)

        return ARIAResult(
            transaction_id=transaction_id,
            has_coercion_signal=coercion_score is not None,
            has_agent_signal=intent_drift_score is not None,
            has_transaction_signal=transaction_fraud_score is not None,
            coercion_score=coercion_score,
            intent_drift_score=intent_drift_score,
            transaction_fraud_score=transaction_fraud_score,
            aria_score=aria_score,
            score_confidence=confidence,
            response=response,
            primary_signal=primary_signal,
            evidence_items=all_evidence[:5],
            attack_category_guess=attack_category_guess,
            authorization_integrity_attestation=self._generate_attestation(aria_score, response.tier.value),
            integrity_assessment=integrity_assessment,
        )

    def _build_response(self, aria_score: float) -> TieredResponse:
        """Build the tiered response for a given ARIA score."""
        for tier, (low, high) in TIER_THRESHOLDS.items():
            if low <= aria_score < high:
                config = TIER_CONFIGS[tier]
                return TieredResponse(
                    tier=tier,
                    aria_score=aria_score,
                    **config
                )
        # Fallback
        return TieredResponse(tier=TierLevel.HARD_BLOCK, aria_score=aria_score,
                               **TIER_CONFIGS[TierLevel.HARD_BLOCK])

    def _assess_integrity(
        self,
        aria_score: float,
        primary_signal: str,
        evidence: List[str]
    ) -> str:
        """Generate human-readable integrity assessment."""
        if aria_score < 0.40:
            return (
                "Authorization context appears intact. "
                "No significant coercion or agent manipulation signals detected. "
                "Transaction may proceed normally."
            )
        elif aria_score < 0.60:
            return (
                f"Authorization context shows LOW CONFIDENCE signals (primary: {primary_signal}). "
                "Soft verification recommended before final authorization release."
            )
        elif aria_score < 0.80:
            return (
                f"Authorization context shows MODERATE RISK signals (primary: {primary_signal}). "
                f"Evidence: {'; '.join(evidence[:2])}. "
                "Delayed release and alternate-contact verification required."
            )
        else:
            return (
                f"Authorization context shows HIGH RISK of {'coercion' if primary_signal == 'coercion' else 'agent hijacking'}. "
                f"Evidence: {'; '.join(evidence[:2])}. "
                "Transaction blocked. Issuer and fraud team alerted. "
                "I4C/1930 reporting link surfaced to user."
            )

    def _generate_attestation(self, aria_score: float, tier: str) -> str:
        """
        Generate a simulated cryptographic attestation string.
        In production, this would be a signed JWT attached to the Verifiable Intent record.
        Format: ARIA-[version]-[tier]-[score_band]-[timestamp]
        """
        import time
        score_band = (
            "clean" if aria_score < 0.40 else
            "low_risk" if aria_score < 0.60 else
            "high_risk" if aria_score < 0.80 else
            "blocked"
        )
        timestamp = int(time.time())
        return f"ARIA-v1-{tier.lower()}-{score_band}-{timestamp}"


# ─── Convenience function ─────────────────────────────────────────────────────

def score_transaction(
    transaction_id: str,
    transcript: Optional[str] = None,
    agent_trace: Optional[Dict] = None,
    transaction_features: Optional[Dict] = None,
    coercion_detector=None,
    intent_drift_detector=None,
    transaction_classifier=None,
) -> ARIAResult:
    """
    Full ARIA scoring pipeline for a single authorization event.
    Pass whichever inputs are available — ARIA handles partial signals.
    """
    engine = ARIAScoreEngine()

    coercion_score = None
    coercion_evidence = None
    intent_drift_score = None
    drift_evidence = None
    transaction_fraud_score = None
    attack_cat = "unknown"

    # ── Coercion scoring ──
    if transcript and coercion_detector:
        cr = coercion_detector.score(transcript)
        coercion_score = cr.coercion_score
        coercion_evidence = [f[0] for f in cr.top_features[:3]]
        attack_cat = cr.attack_category

    # ── Intent drift scoring ──
    if agent_trace and intent_drift_detector:
        idr = intent_drift_detector.score(
            user_mandate=agent_trace.get("user_mandate", ""),
            tool_call_sequence=agent_trace.get("tool_call_sequence", []),
            final_action=agent_trace.get("final_action", {}),
            mandate_amount_limit=agent_trace.get("mandate_amount_limit"),
            mandate_merchant_category=agent_trace.get("mandate_merchant_category"),
            trace_id=transaction_id,
        )
        intent_drift_score = idr.intent_drift_score
        drift_evidence = idr.top_evidence
        if attack_cat == "unknown":
            attack_cat = idr.attack_type_guess

    # ── Transaction scoring ──
    if transaction_features and transaction_classifier:
        try:
            import numpy as np
            feature_vec = np.array([[transaction_features.get(f, 0)
                                     for f in transaction_classifier.feature_names_in_]])
            transaction_fraud_score = float(transaction_classifier.predict_proba(feature_vec)[0, 1])
        except Exception as e:
            transaction_fraud_score = None

    return engine.fuse(
        transaction_id=transaction_id,
        coercion_score=coercion_score,
        intent_drift_score=intent_drift_score,
        transaction_fraud_score=transaction_fraud_score,
        coercion_evidence=coercion_evidence,
        drift_evidence=drift_evidence,
        attack_category_guess=attack_cat,
    )


if __name__ == "__main__":
    engine = ARIAScoreEngine()

    # Simulate different scenarios
    test_cases = [
        {"label": "Clean transaction",     "c": 0.05, "d": None, "t": 0.08},
        {"label": "Soft friction zone",    "c": 0.50, "d": None, "t": 0.20},
        {"label": "Delayed release",       "c": 0.70, "d": 0.65, "t": 0.55},
        {"label": "Hard block — coercion", "c": 0.92, "d": None, "t": 0.30},
        {"label": "Hard block — agent",    "c": None, "d": 0.95, "t": 0.88},
    ]

    print("\n" + "="*70)
    print("ARIA Score Engine — Test Cases")
    print("="*70)

    for tc in test_cases:
        result = engine.fuse(
            transaction_id="TXN-TEST",
            coercion_score=tc["c"],
            intent_drift_score=tc["d"],
            transaction_fraud_score=tc["t"],
        )
        print(f"\n[{tc['label']}]")
        print(f"  Scores: coercion={tc['c']}, drift={tc['d']}, txn={tc['t']}")
        print(f"  ARIA Score: {result.aria_score:.3f}")
        print(f"  Tier: {result.response.tier.value}")
        print(f"  Action: {result.response.action_label}")
        print(f"  Attestation: {result.authorization_integrity_attestation}")
