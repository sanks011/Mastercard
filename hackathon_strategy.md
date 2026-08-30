# 🔴 Mastercard AI Defense Lab 2026 — Strategy Brief
### *Independent Research-Backed Analysis | Aug 15, 2026*

---

## 0. What the Judges Are Actually Looking For (Reading Between the Lines)

From the rubric, in rough order of importance:
1. **Novelty of solution** — "wow, we've never seen that framing"
2. **Diversity + fidelity of attacks** — breadth + realism, not just one polished classifier
3. **Detection efficacy** — actual metrics (F1, AUC), not vibes
4. **Real-world feasibility** — could this live inside Mastercard's stack TODAY?
5. **Quality of presentation** — the working web prototype requirement is a differentiator

The GFF 2026 theme is **"Agentic AI | Tokenisation | Quantum"**. Any submission that directly echoes that theme gets unconscious credit.

> [!IMPORTANT]
> The judges **work at Mastercard**. They know their own products. Submissions that
> build *on top of* existing Mastercard infrastructure (Decision Intelligence, Agent Pay,
> Verifiable Intent, FIDO-standard AP2) score dramatically higher than ones that pretend
> those systems don't exist.

---

## 1. The Research Gap Nobody Will Fill (Your Moat)

After independent web research across fraud literature, Mastercard's own published architecture, OWASP 2026, and Indian cyber-crime data, here's the real whitespace:

### The Generic Answer (What 90% of teams will submit)
- "Build a deepfake/GAN classifier on transaction tabular data"
- "Detect synthetic identities using a CNN"
- Basically: **treat fraud as a CONTENT problem** → detect the fake artifact

### The Actual Problem (What the research says)
**GenAI fraud in 2026 has moved past content authenticity.** The real frontier is:**Authorization Integrity** — the question isn't "is this face real?" it's "was the DECISION to pay made under legitimate conditions?"

Three converging attack types, all exploiting the same blind spot:

| Attack Type | Current Detection Coverage | Why It Wins |
|---|---|---|
| **Coercion/Vishing + Deepfake** (Digital Arrest scams, ₹22,495 Cr/yr in India) | Transaction layer only | Legitimate card, legitimate user, coerced authorization |
| **Prompt Injection on AI Shopping Agents** (Zscaler/Unit 42 documented live attacks) | Almost none | Clean, fast, successful-looking transaction |
| **Agentic Collusion** (MultiAgentFinancialFraudBench, 2026) | Nothing deployed | Multiple LLM agents coordinate to execute multi-step fraud |

All three share: **the transaction looks completely normal to every existing fraud model. The corruption happened BEFORE the payment was authorized.**

That's the insight. That's your thesis.

---

## 2. The Solution: ARIA — Authorization Risk & Integrity Analyzer

*A closed-loop red-team/blue-team system that attacks and defends the **authorization decision**, not just the transaction.*

### Rename from "AIS" → "ARIA" — it's catchier, demo-friendly, and memorable for judges.

```
         ┌──────────────────────────────────────────────────┐
         │                    ARIA SYSTEM                   │
         │                                                  │
         │  ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
         │  │  IDENTIFY │──▶│ GENERATE │──▶│   DEFEND    │  │
         │  │          │   │          │   │             │  │
         │  │ Attack   │   │ Synthetic│   │ Detection   │  │
         │  │ Taxonomy │   │ Dataset  │   │ + Response  │  │
         │  │ (5 types)│   │ Generator│   │ Engine      │  │
         │  └──────────┘   └──────────┘   └─────────────┘  │
         │        │               │               │          │
         │        └───────────────┴───────────────┘          │
         │                  Feedback Loop                    │
         │         (detected gaps → new attack vectors)      │
         └──────────────────────────────────────────────────┘
```

---

## 3. IDENTIFY — The Attack Taxonomy (Breadth = Scoring Points)

Don't pitch 1-2 attacks. The rubric explicitly rewards **"diversity and breadth."** Here's a 5-category taxonomy that covers every major GenAI fraud surface in 2026, each with evidence:

### Category A: Authorization Coercion Attacks
*Exploiting the human in the loop through AI-powered social engineering*

**A1 — LLM-Powered Digital Arrest / Vishing**
- Real evidence: 58,239 Indian complaints in 2025, ₹22,495 Cr total fraud losses
- Attack: LLM generates personalized authority-figure scripts (CBI/ED impersonation) + real-time deepfake voice replication, coercing victims to self-transfer funds
- Why novel: The authorization is technically legitimate — victim pressed "pay"

**A2 — Predictive Invoice Impersonation ("Business Email Compromise 2.0")**
- Real evidence: Sardine.ai 2026 report — AI times invoices to match vendor patterns
- Attack: LLM scrapes public data (LinkedIn, company websites) to generate hyper-personalized BEC at machine speed, matching known vendor payment rhythms exactly

**A3 — Emotional Manipulation via Synthetic Relationships**
- Pig-butchering scams now fully automated via LLM agents playing "romantic partner" for months before the payment request

### Category B: Agentic Commerce Attacks
*Compromising AI agents with delegated payment authority*

**B1 — Indirect Prompt Injection on Shopping Agents**
- Real evidence: Zscaler caught live attacks in 2025; Unit 42 documented the vector
- Attack: Hidden HTML/metadata on product pages instructs AI shopping agent to "resolve an account error" by sending payment to attacker wallet
- Why it's undetected: Looks like a normal, fast, successful transaction to Decision Intelligence

**B2 — Tool Poisoning / Malicious API Description Injection**
- Real evidence: arXiv 2026 — attackers embed malicious logic in tool definitions that LLM agents treat as authoritative
- Attack: Compromised payment tool SDK returns modified instructions telling agent to add 1% "processing fee" to attacker account

**B3 — Multi-Agent Collusion Fraud**
- Real evidence: *MultiAgentFinancialFraudBench* (OpenReview 2026)
- Attack: Network of coordinated LLM agents — one does account research, one does social engineering, one executes the transfer — no single agent has enough context to trigger individual flags

### Category C: Synthetic Identity & KYC Bypass
*GenAI creating entirely new fraudulent personas*

**C1 — "Frankenstein" Synthetic Identity Generation**
- GenAI generates fake but consistent documents (Aadhaar-style), deepfake face that passes liveness checks via camera injection attacks
- Builds "slow burn" credit history over months before bust-out

**C2 — RAG Poisoning of Fraud Detection Systems**
- Novel attack: poison the vector database of RAG-based fraud monitoring systems by injecting semantically similar but fraudulent transaction patterns as "legitimate" embeddings
- Causes detector to classify fraud as safe

### Category D: Adversarial ML Evasion
*Making fraud invisible to trained models*

**D1 — Gradient-Guided Transaction Morphing**
- Real evidence: FRAUD-RLA model (arXiv 2025) uses RL to iteratively craft transactions that score as "legitimate" to target fraud model
- Attacker uses feedback from fraud score (e.g., declined = too risky) to hill-climb toward undetectable fraud

**D2 — GAN-Synthesized "Ghost Transactions"**
- Train a GAN on legitimate transaction distributions; generate synthetic fraud that is statistically indistinguishable from real behavior

### Category E: Infrastructure & Agentic Token Attacks
*Targeting the new Agent Pay / Verifiable Intent layer*

**E1 — Scope Creep via Mandate Ambiguity**
- Real evidence: Mastercard's own Verifiable Intent docs acknowledge "selective disclosure" complexity
- Attack: Exploit ambiguously-worded mandate scopes (e.g., "household expenses") to authorize clearly out-of-scope transactions
- Why relevant: Mastercard themselves are building the defense here, so attacking it shows you understand their stack

**E2 — Token Replay with Stale Consent**
- Attack: Steal or replay an agentic token after the human's consent window has expired but before the token has been invalidated

---

## 4. GENERATE — Building the Synthetic Dataset (The Technical Core)

This is where most teams fail — they use existing datasets. You generate your own, and that's a genuine contribution.

### Sub-system 1: Coercion Conversation Generator

**Input:** Attack category + victim profile (demographics, account type)
**Process:**
1. LLM (GPT-4o or Gemini) generates vishing script using a structured prompt framework:
   - Authority phase (impersonation + credential fabrication)
   - Urgency phase (time pressure, account freeze threats)
   - Isolation phase ("don't tell anyone, stay on the line")
   - Payment phase (instruction to transfer)
2. Extract linguistic features: coercion markers, urgency lexicon, turn-rigidity score
3. Label with coercion intensity score (0-1)

**Output:** 1000+ synthetic call transcripts with feature vectors + labels

### Sub-system 2: Agent Trace Generator

**Architecture:**
```python
# Toy agentic commerce harness
User Intent → Shopping Agent (LangChain/LangGraph) → Tool Calls → Payment API

# Inject attack payloads at:
# - Product page metadata (Indirect Prompt Injection)
# - Tool API response (Tool Poisoning)  
# - Multi-agent message passing (Collusion)
```

**Output:** JSON tool-call traces with fields:
- `user_mandate`: original stated intent (e.g., "buy running shoes under ₹3000")
- `tool_call_sequence`: list of API calls made
- `final_action`: actual payment (merchant, amount, recipient)
- `label`: 0 = benign, 1 = hijacked

### Sub-system 3: Synthetic Transaction Generator

**Method:** Conditional Tabular GAN (CTGAN from SDV library — open source, well-cited)
- Train on public datasets: IEEE-CIS Fraud Detection, PaySim
- Condition on attack type to generate category-specific fraud distributions
- Use Wasserstein distance to measure fidelity vs. real distributions (the "fidelity" score judges want)

**Output:** Synthetic transaction table with realistic distributions, labeled by attack type

---

## 5. DEFEND — The Detection Architecture

### Tier 1: Fast Coercion Detector (sub-100ms, on text)
**Input:** Call transcript / chat log
**Features:**
- Coercion marker score (authority words, urgency phrases, isolation tactics)
- Turn-taking rigidity (caller dominates, victim gets no time to think)
- Request-to-context coherence (does the payment request make sense given stated reason?)
- Time pressure density (urgency words per minute)

**Model:** XGBoost classifier — fast, interpretable, explainable output
**Target metrics:** F1 > 0.90 on synthetic test set

### Tier 2: Agent Intent-Drift Detector
*The most novel component — nothing like this is deployed anywhere*

**Input:** Agent tool-call trace
**Algorithm:**
```
mandate_vector = embed(user_stated_intent)
action_vector = embed(final_payment_action: merchant + amount + recipient)
intent_drift_score = cosine_distance(mandate_vector, action_vector)

# Enhanced with:
scope_violation_score = check_explicit_constraints(mandate_limits, actual_action)
injection_signature_score = detect_prompt_injection_patterns(intermediate_tool_calls)

final_score = weighted_fusion(intent_drift_score, scope_violation_score, injection_signature_score)
```

**Why this works:** Legitimate agents show LOW drift (you said "buy shoes" → agent bought shoes). Compromised agents show HIGH drift (you said "buy shoes" → agent transferred ₹50,000 to unknown wallet).

### Tier 3: Ensemble Fraud Classifier (Tabular)
**Model:** LightGBM on synthetic transaction data
**Features:** Standard tabular + agent signals (agent_id present, mandate_freshness, scope_match_score)
**Builds on:** Mastercard's own Decision Intelligence architecture (explicitly reference this)

### Score Fusion
```
ARIA_Score = α × coercion_score + β × intent_drift_score + γ × transaction_score

# Where α, β, γ learned by gradient boosting on labeled validation set
# Cascade: fast models first, expensive LLM explainer only for borderline cases
```

### Tiered Response Engine
| ARIA Score | Action | Rationale |
|---|---|---|
| 0.0 - 0.4 | Pass | Clean transaction |
| 0.4 - 0.6 | Soft friction | "Confirm this payment — take 60 seconds to call a trusted contact" |
| 0.6 - 0.8 | Delayed release | Hold 4 hours, send OTP to alternate contact |
| 0.8 - 1.0 | Hard block + Alert | Block, notify issuer, surface I4C/1930 reporting link (India-specific) |

> [!TIP]
> The soft-friction tier is *critical* for judges. Binary block/pass shows you don't
> understand real-world fraud operations. Tiered response shows product thinking.
> Coercion victims PANIC and retry if hard-blocked — soft friction breaks the attacker's
> script. This is novel and defensible.

---

## 6. The Verifiable Intent Extension (Judges Will Love This)

This is the move that signals you've done real homework.

Mastercard's Verifiable Intent (published March 2026, contributed to FIDO Alliance) creates a cryptographic audit trail of:
- Consumer identity ↔ Authorized instructions ↔ Transaction outcome

**Your extension:** Add a 4th element to the chain:
- Consumer identity ↔ Authorized instructions ↔ **Authorization context integrity** ↔ Transaction outcome

Where "authorization context integrity" is ARIA's assessment of whether the authorization event itself was coercion-free and agent-hijack-free.

In practice: ARIA generates a signed integrity attestation alongside each ARIA score. The Verifiable Intent record can include this attestation — issuers know not just "this agent was authorized" but "this authorization was made under verified uncoerced conditions."

This is a concrete, implementable extension that builds on Mastercard's own 2026 roadmap. No other team will pitch this.

---

## 7. What Claude Got Right (and Where to Push Harder)

Claude's AIS concept is essentially correct, but here's where to push further:

✅ **Validate:** The "authorization event" framing is the right insight — nobody's scoring it
✅ **Validate:** Agent intent-drift detector is genuinely novel and buildable
✅ **Validate:** Verifiable Intent extension is the right credibility signal to judges
✅ **Validate:** Tiered response > binary block

⬆️ **Push harder here:**
1. **Attack taxonomy breadth** — Claude pitched 2-3 vectors. The rubric says "exhaustive." You need 10+ named, evidenced attack types. The 5-category, 10+ attack taxonomy above is what scores on "diversity."
2. **RAG poisoning** (Category C2) — Claude missed this entirely. It's a 2026-specific GenAI attack vector that's technically sophisticated and shows research depth.
3. **Multi-agent collusion** (Category B3) — Claude mentioned it vaguely. The *MultiAgentFinancialFraudBench* paper (2026) gives you a real citation to ground this.
4. **GFF theme alignment** — GFF 2026 theme is "Agentic AI | Tokenisation." Explicitly frame ARIA as the security layer for the Agentic AI + Tokenisation economy. This is not just cosmetic — the judges care.
5. **Graph neural network for mule detection** — Graph-based approaches detect 40% more fraud rings than tabular ML (Fluxforce.ai research). Add a GNN layer for your tabular fraud classifier to cover mule network detection.

---

## 8. Realistic Build Plan (Aug 15 → Aug 31 = 16 days)

### If Solo:
| Days | Task | Output |
|---|---|---|
| Day 1-3 | Build coercion conversation generator + feature extractor | 500+ labeled transcripts |
| Day 4-6 | Build agent trace harness (LangGraph + synthetic product pages with injection) | 300+ labeled traces |
| Day 7-9 | Train intent-drift detector (the novel component) | Working model, metrics |
| Day 10-11 | CTGAN synthetic transaction generator + LightGBM classifier | Tabular fraud model |
| Day 12-13 | Score fusion + tiered response engine | ARIA score pipeline |
| Day 14-15 | Streamlit web prototype (required per rules) | Working UI |
| Day 16 | Deck + writeup + GitHub cleanup | Submission ready |

### If Team (2-5 people):
- **Person 1:** Attack taxonomy research + coercion dataset generator
- **Person 2:** Agent trace harness + intent-drift detector
- **Person 3:** CTGAN + tabular fraud model + GNN layer
- **Person 4 (if exists):** Web prototype UI + score visualization dashboard
- **Person 5 (if exists):** Deck + writeup + Verifiable Intent extension design

---

## 9. The Prototype (Web UI) — What to Build

**Required per rules.** Most teams will build a Streamlit form. You should build something that visually demonstrates the closed loop and WOWs a judge who's seen 50 demos.

**Recommended stack:** React (Vite) + Python FastAPI backend

**Three demo modes:**
1. **Red Team Mode** — "Generate Attack": select attack type (A1-E2), configure victim profile, watch the system generate a synthetic attack trace in real time
2. **Blue Team Mode** — "Analyze Trace": paste a call transcript or agent trace, watch ARIA score it in real time with feature contribution breakdown (SHAP values)
3. **Live Loop Mode** — run a simulated attack, watch the defender catch it, see the tiered response fire. This is your wow moment.

**Key UI elements:**
- ARIA score gauge (0-1) with color gradient (green → red)
- Feature breakdown bar chart (which signals drove the score)
- Tiered response card (what action fires and why)
- Attack taxonomy browser (all 10+ vectors with descriptions)

---

## 10. Writeup Structure (Kaggle Report)

### Title: **"ARIA: Authorization Risk & Integrity Analyzer — Defending the Decision, Not Just the Transaction"**

### Subtitle: **"A closed-loop red-team/blue-team system targeting the undefended attack surface in GenAI payment fraud"**

### Sections:
1. **The Reframe** (200 words) — "Everyone is defending the transaction. We're defending the authorization event." Use the India ₹22,495 Cr stat + Zscaler agentic prompt injection as your dual evidence anchor.
2. **Attack Taxonomy** (breadth scoring) — the 5-category, 10+ attack table. Be exhaustive. Show citations.
3. **Generate** — explain the three sub-systems, show sample outputs (transcript snippet, trace JSON, synthetic transaction distribution plot)
4. **Defend** — architecture diagram, model metrics (precision, recall, F1, AUC on each detector), the tiered response table
5. **Verifiable Intent Extension** — the Mastercard-stack integration. Show you know their product.
6. **Feasibility Analysis** — latency analysis (fast models <50ms for Tier 1/2, consistent with Mastercard's <50ms DI requirement), integration points with Decision Intelligence, realistic deployment path
7. **Limitations & Future Work** — audio/prosody pipeline for live call analysis, real-time SHAP for explainability at scale. Showing you know your limits builds trust.

---

## 11. Confidence Assessment

| Claim | Evidence Quality | Risk if Wrong |
|---|---|---|
| Authorization-event framing is novel | HIGH — web search found no existing commercial product doing this | Low — "novel" is inherently somewhat subjective |
| India ₹22,495 Cr fraud stat | HIGH — I4C / MHA official data | Low |
| Zscaler prompt injection on agents | HIGH — Unit 42 documented | Low |
| Verifiable Intent is real and correct | HIGH — Mastercard + Google + FIDO published | Low |
| Decision Intelligence <50ms requirement | HIGH — Mastercard published | Low |
| MultiAgentFinancialFraudBench paper | HIGH — cited in search results | Medium — verify paper exists before citing |
| GNN detects 40% more fraud rings | MEDIUM — one source (Fluxforce.ai) | Medium — use as directional, cite source |
| CTGAN/SDV for synthetic data | HIGH — well-established open source library | Low |

> [!WARNING]
> Before citing *MultiAgentFinancialFraudBench*, find the actual paper on OpenReview/arXiv
> and verify its title, authors, and findings. Citing papers that don't exist is an
> automatic credibility killer.

---

## 12. The Single Sentence Pitch

> "ARIA catches what every existing fraud system misses: not the fake transaction, but the coerced decision behind it — using synthetic attack data we generated ourselves, across human coercion and AI agent hijacking, fused into one explainable score that fits inside Mastercard's own Verifiable Intent framework."

---

*Research completed: Aug 15, 2026. Sources: Mastercard.com, I4C/MHA India, Zscaler/Unit 42, OWASP 2026, arXiv, OpenReview, sardine.ai, fluxforce.ai, FIDO Alliance announcements.*
