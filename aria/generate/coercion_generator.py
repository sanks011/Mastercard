"""
coercion_generator.py
=====================
Generates synthetic coercion call transcripts using Gemini 2.5 Flash.
Covers 5 attack sub-types across Category A (Authorization Coercion).
Outputs labeled transcripts with extracted linguistic features.
"""

import os
import json
import time
import random
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "coercion_transcripts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_SUBTYPES = {
    "A1_digital_arrest": {
        "persona": "CBI/ED/Police officer",
        "hook": "Your Aadhaar/PAN/mobile number is linked to a criminal case",
        "demand": "Transfer funds as 'security deposit' or 'clearance fee'",
        "urgency_type": "arrest warrant will be issued in 2 hours",
    },
    "A1_trai_scam": {
        "persona": "TRAI/DoT official",
        "hook": "Your phone number will be disconnected for illegal activity",
        "demand": "Pay fine to avoid disconnection and legal action",
        "urgency_type": "disconnection in 30 minutes",
    },
    "A2_bec": {
        "persona": "Company CFO / CEO / vendor",
        "hook": "Urgent payment required for a business deal / vendor invoice",
        "demand": "Wire transfer to a specific account immediately",
        "urgency_type": "deal will fall through / penalties if not paid today",
    },
    "A2_insurance": {
        "persona": "Insurance company representative",
        "hook": "Your policy has lapsed / you have unclaimed benefits",
        "demand": "Pay processing fee to release ₹X lakhs in insurance payout",
        "urgency_type": "offer expires in 24 hours",
    },
    "A3_romance": {
        "persona": "Romantic partner / new friend met online",
        "hook": "Emergency — stuck abroad, medical crisis, investment opportunity",
        "demand": "Send money urgently to help / invest in guaranteed returns",
        "urgency_type": "life-threatening emergency / opportunity closes in hours",
    },
}

VICTIM_PROFILES = [
    {"age": 65, "occupation": "retired", "location": "Delhi", "tech_savvy": "low"},
    {"age": 45, "occupation": "small business owner", "location": "Mumbai", "tech_savvy": "medium"},
    {"age": 30, "occupation": "IT professional", "location": "Bangalore", "tech_savvy": "high"},
    {"age": 55, "occupation": "homemaker", "location": "Kolkata", "tech_savvy": "low"},
    {"age": 28, "occupation": "student", "location": "Pune", "tech_savvy": "high"},
    {"age": 50, "occupation": "teacher", "location": "Chennai", "tech_savvy": "medium"},
    {"age": 40, "occupation": "doctor", "location": "Hyderabad", "tech_savvy": "medium"},
]

# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class CoercionTranscript:
    id: str
    attack_subtype: str
    attack_category: str  # A1, A2, or A3
    victim_profile: dict
    transcript: str
    label: int            # 1 = coercive (attack), 0 = benign
    coercion_intensity: float  # 0.0–1.0

    # Linguistic features (extracted post-generation)
    urgency_density: float = 0.0          # urgency phrases per 100 words
    authority_score: float = 0.0          # authority/official language score
    isolation_markers: int = 0            # count of "don't tell anyone" type phrases
    turn_rigidity: float = 0.0            # how much caller dominates conversation
    threat_count: int = 0                 # explicit threat statements
    personal_info_used: int = 0           # how many personal details mentioned
    payment_urgency: bool = False         # explicit payment demand present
    request_coherence: float = 0.0        # does payment request logically follow context?


# ─── Prompt Templates ─────────────────────────────────────────────────────────

COERCION_PROMPT = """You are generating a SYNTHETIC, LABELED training dataset for an AI fraud detection research system.

Generate a realistic call transcript between a FRAUDSTER and a VICTIM.

Attack Type: {attack_subtype}
Fraudster Persona: {persona}
Initial Hook: {hook}
Payment Demand: {demand}
Urgency Mechanism: {urgency_type}

Victim Profile:
- Age: {age}
- Occupation: {occupation}  
- Location: {location}
- Tech savvy: {tech_savvy}

Requirements for the transcript:
1. Write 20–35 conversational turns (alternating CALLER and VICTIM)
2. Include the classic coercion arc: Establish credibility → Create fear → Isolation → Payment demand
3. Include realistic Indian payment context (UPI, Aadhaar, PAN references)
4. The victim should show realistic resistance/confusion before eventually being coerced
5. Include specific linguistic coercion markers: urgency language, authority claims, isolation instructions
6. Make it realistic — NOT cartoonishly obvious. Good fraudsters sound professional.

After the transcript, provide a JSON analysis block:
```json
{{
  "coercion_intensity": <float 0.0-1.0>,
  "key_coercion_phrases": ["phrase1", "phrase2", "phrase3"],
  "attack_phases": ["credibility", "fear", "isolation", "payment"],
  "payment_amount_mentioned": <true/false>,
  "isolation_instruction": <true/false>,
  "authority_entities_mentioned": ["entity1", "entity2"]
}}
```

Format the transcript as:
CALLER: [text]
VICTIM: [text]
CALLER: [text]
...

IMPORTANT: This is for defensive AI research to detect and prevent such scams. The data will be used to train a model that protects victims.
"""

BENIGN_PROMPT = """Generate a LEGITIMATE customer service call transcript for an AI fraud detection training dataset (benign class).

Scenario: {scenario}

Requirements:
1. Write 10–20 conversational turns
2. Realistic, helpful customer service interaction
3. May involve payment discussions but NO coercion or pressure
4. Customer service agent is helpful and gives customer time to think
5. Include Indian context (UPI, bank, insurance, etc.)

Format:
AGENT: [text]
CUSTOMER: [text]

After the transcript, provide:
```json
{{
  "coercion_intensity": 0.0,
  "key_coercion_phrases": [],
  "is_benign": true
}}
```
"""

BENIGN_SCENARIOS = [
    "Customer calling bank to dispute a legitimate charge on their credit card",
    "Customer calling insurance company to check claim status",
    "Customer calling telecom support to fix a billing error",
    "Bank calling customer to verify an unusual but legitimate transaction they made",
    "Customer calling mutual fund company to redeem units",
    "UPI support helping customer recover access to their account after phone change",
]


# ─── Feature Extraction ───────────────────────────────────────────────────────

URGENCY_PHRASES = [
    "immediately", "urgent", "right now", "within the hour", "don't delay",
    "last chance", "final warning", "arrest warrant", "FIR will be filed",
    "2 hours", "30 minutes", "today only", "no time", "immediately transfer",
    "अभी", "तुरंत", "जल्दी"  # Hindi urgency words
]

AUTHORITY_PHRASES = [
    "CBI", "ED", "police", "supreme court", "RBI", "TRAI", "Enforcement Directorate",
    "Ministry", "government", "official", "case number", "FIR", "warrant",
    "cybercrime", "investigation", "national security"
]

ISOLATION_PHRASES = [
    "don't tell anyone", "keep this confidential", "do not share",
    "don't inform your family", "stay on the line", "don't hang up",
    "this is secret", "classified information", "cannot share with others",
    "don't discuss with anyone"
]


def extract_features(transcript: str, analysis_json: dict) -> dict:
    """Extract linguistic features from transcript text."""
    text_lower = transcript.lower()
    words = transcript.split()
    word_count = max(len(words), 1)

    # Count caller turns vs victim turns
    caller_turns = len(re.findall(r'^CALLER:', transcript, re.MULTILINE))
    victim_turns = len(re.findall(r'^VICTIM:', transcript, re.MULTILINE))
    total_turns = max(caller_turns + victim_turns, 1)

    # Urgency density
    urgency_count = sum(text_lower.count(p.lower()) for p in URGENCY_PHRASES)
    urgency_density = (urgency_count / word_count) * 100

    # Authority score
    authority_count = sum(text_lower.count(p.lower()) for p in AUTHORITY_PHRASES)
    authority_score = min(authority_count / 5.0, 1.0)

    # Isolation markers
    isolation_count = sum(text_lower.count(p.lower()) for p in ISOLATION_PHRASES)

    # Turn rigidity (caller dominance)
    caller_text = " ".join(re.findall(r'CALLER: (.+)', transcript))
    victim_text = " ".join(re.findall(r'VICTIM: (.+)', transcript))
    caller_word_count = len(caller_text.split())
    total_text_words = max(caller_word_count + len(victim_text.split()), 1)
    turn_rigidity = caller_word_count / total_text_words

    # Threat count (simple heuristic)
    threat_phrases = ["arrest", "jail", "prison", "legal action", "FIR", "warrant",
                      "seized", "frozen", "blocked", "cancelled"]
    threat_count = sum(text_lower.count(p) for p in threat_phrases)

    # Personal info used
    personal_markers = ["aadhaar", "pan", "account number", "your mobile", "your address", "your name"]
    personal_count = sum(text_lower.count(p) for p in personal_markers)

    # Payment demand present
    payment_phrases = ["transfer", "upi", "rtgs", "neft", "imps", "send money", "pay now", "deposit"]
    payment_urgency = any(p in text_lower for p in payment_phrases)

    # Request coherence (simple: does payment follow from stated problem?)
    # High if payment demand is directly tied to the stated authority issue
    coherence_signals = ["to resolve", "to clear", "to avoid", "to release", "clearance fee", "processing fee"]
    coherence_count = sum(text_lower.count(p) for p in coherence_signals)
    request_coherence = min(coherence_count / 3.0, 1.0)

    return {
        "urgency_density": round(urgency_density, 3),
        "authority_score": round(authority_score, 3),
        "isolation_markers": isolation_count,
        "turn_rigidity": round(turn_rigidity, 3),
        "threat_count": threat_count,
        "personal_info_used": personal_count,
        "payment_urgency": payment_urgency,
        "request_coherence": round(request_coherence, 3),
        "coercion_intensity": analysis_json.get("coercion_intensity", 0.0),
    }


# ─── Generator ────────────────────────────────────────────────────────────────

class CoercionDataGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.generated = []

    def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Call Gemini with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.85,
                        max_output_tokens=3000,
                    )
                )
                return response.text
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return ""

    def _extract_json_block(self, text: str) -> dict:
        """Extract JSON from ```json ... ``` block."""
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {"coercion_intensity": 0.5}

    def _extract_transcript(self, text: str) -> str:
        """Extract the transcript portion (before the JSON block)."""
        # Remove the JSON block
        clean = re.sub(r'```json.*?```', '', text, flags=re.DOTALL).strip()
        return clean

    def generate_coercive_transcript(
        self,
        attack_subtype: str,
        victim_profile: dict,
        idx: int
    ) -> CoercionTranscript:
        """Generate one coercive (attack) transcript."""
        subtype_config = ATTACK_SUBTYPES[attack_subtype]

        prompt = COERCION_PROMPT.format(
            attack_subtype=attack_subtype,
            persona=subtype_config["persona"],
            hook=subtype_config["hook"],
            demand=subtype_config["demand"],
            urgency_type=subtype_config["urgency_type"],
            **victim_profile
        )

        print(f"  Generating coercive transcript [{idx}]: {attack_subtype} | victim_age={victim_profile['age']}")
        raw_output = self._call_gemini(prompt)

        if not raw_output:
            return None

        analysis = self._extract_json_block(raw_output)
        transcript_text = self._extract_transcript(raw_output)
        features = extract_features(transcript_text, analysis)

        # Determine attack category (A1, A2, A3)
        if "digital_arrest" in attack_subtype or "trai" in attack_subtype:
            category = "A1"
        elif "bec" in attack_subtype or "insurance" in attack_subtype:
            category = "A2"
        else:
            category = "A3"

        return CoercionTranscript(
            id=f"coerce_{idx:04d}_{attack_subtype}",
            attack_subtype=attack_subtype,
            attack_category=category,
            victim_profile=victim_profile,
            transcript=transcript_text,
            label=1,
            **features
        )

    def generate_benign_transcript(self, scenario: str, idx: int) -> CoercionTranscript:
        """Generate one benign (legitimate) call transcript."""
        prompt = BENIGN_PROMPT.format(scenario=scenario)

        print(f"  Generating benign transcript [{idx}]: {scenario[:50]}...")
        raw_output = self._call_gemini(prompt)

        if not raw_output:
            return None

        analysis = self._extract_json_block(raw_output)
        transcript_text = self._extract_transcript(raw_output)

        return CoercionTranscript(
            id=f"benign_{idx:04d}",
            attack_subtype="benign",
            attack_category="benign",
            victim_profile={"scenario": scenario},
            transcript=transcript_text,
            label=0,
            coercion_intensity=0.0,
            urgency_density=random.uniform(0, 0.5),
            authority_score=random.uniform(0, 0.1),
            isolation_markers=0,
            turn_rigidity=random.uniform(0.4, 0.55),
            threat_count=0,
            personal_info_used=random.randint(0, 2),
            payment_urgency=random.choice([True, False]),
            request_coherence=random.uniform(0.7, 1.0)
        )

    def generate_dataset(
        self,
        n_coercive: int = 400,
        n_benign: int = 150,
        output_file: Optional[str] = None
    ) -> List[dict]:
        """
        Generate full coercion dataset.
        
        Args:
            n_coercive: Number of coercive transcripts to generate
            n_benign: Number of benign transcripts to generate
            output_file: Path to save JSON dataset (defaults to data/coercion_transcripts/)
        """
        print(f"\n{'='*60}")
        print(f"ARIA Coercion Dataset Generator")
        print(f"Generating {n_coercive} coercive + {n_benign} benign transcripts")
        print(f"{'='*60}\n")

        all_records = []
        idx = 0

        # ── Generate coercive transcripts ──
        subtypes = list(ATTACK_SUBTYPES.keys())
        per_subtype = n_coercive // len(subtypes)

        for subtype in subtypes:
            print(f"\n[Category: {subtype}] Generating {per_subtype} transcripts...")
            for i in range(per_subtype):
                victim = random.choice(VICTIM_PROFILES)
                record = self.generate_coercive_transcript(subtype, victim, idx)
                if record:
                    all_records.append(asdict(record))
                    idx += 1
                time.sleep(0.5)  # Rate limit

        # ── Generate benign transcripts ──
        print(f"\n[Benign] Generating {n_benign} legitimate call transcripts...")
        for i in range(n_benign):
            scenario = BENIGN_SCENARIOS[i % len(BENIGN_SCENARIOS)]
            record = self.generate_benign_transcript(scenario, idx)
            if record:
                all_records.append(asdict(record))
                idx += 1
            time.sleep(0.5)

        # ── Save ──
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"coercion_dataset_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)

        # Also save as feature CSV for easy model training
        csv_path = str(output_file).replace(".json", "_features.csv")
        self._save_feature_csv(all_records, csv_path)

        print(f"\n✓ Dataset saved: {output_file}")
        print(f"✓ Features CSV saved: {csv_path}")
        print(f"✓ Total records: {len(all_records)}")
        print(f"  - Coercive: {sum(1 for r in all_records if r['label'] == 1)}")
        print(f"  - Benign: {sum(1 for r in all_records if r['label'] == 0)}")

        self.generated = all_records
        return all_records

    def _save_feature_csv(self, records: List[dict], csv_path: str):
        """Save extracted features as CSV for model training."""
        import pandas as pd

        feature_cols = [
            "id", "attack_subtype", "attack_category", "label",
            "coercion_intensity", "urgency_density", "authority_score",
            "isolation_markers", "turn_rigidity", "threat_count",
            "personal_info_used", "payment_urgency", "request_coherence"
        ]

        rows = []
        for r in records:
            row = {col: r.get(col, None) for col in feature_cols}
            # victim profile flattening
            if isinstance(r.get("victim_profile"), dict):
                row["victim_age"] = r["victim_profile"].get("age", 0)
                row["victim_tech_savvy"] = r["victim_profile"].get("tech_savvy", "unknown")
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ARIA coercion transcript dataset")
    parser.add_argument("--coercive", type=int, default=100, help="Number of coercive transcripts (default: 100 for quick test)")
    parser.add_argument("--benign", type=int, default=40, help="Number of benign transcripts")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in environment. Set it in .env file.")
        exit(1)

    generator = CoercionDataGenerator()
    dataset = generator.generate_dataset(
        n_coercive=args.coercive,
        n_benign=args.benign,
        output_file=args.output
    )
