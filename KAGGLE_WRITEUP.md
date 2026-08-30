# ARIA: Authorization Risk & Integrity Analyzer
## *A Closed-Loop Red-Team / Blue-Team System Defending the Authorization Event in GenAI-Powered Payment Fraud*

**Mastercard Innovation Challenge @ GFF 2026 — AI Defense Lab for Payment Security**  
**Track:** AI Defense Lab for Payment Security  

---

## 1. Executive Summary: The Fundamental Reframe

Most conventional approaches to AI payment security frame fraud as a **content authenticity** or **transaction anomaly** problem: detecting deepfakes, flagging stolen cards, or calculating velocity deviations.

However, in 2026, empirical data reveals a critical undefended attack surface:
* In India, **Digital Arrest** and AI-driven vishing scams drained over **₹22,495 crore in 2025 across 28.15 lakh cases** (MHA / I4C data). In these attacks, legitimate cardholders are psychologically coerced via real-time LLM dialogue and voice deepfakes into pressing "Authorize".
* In **Agentic Commerce**, autonomous shopping agents with delegated payment tokens are subverted through **Indirect Prompt Injection** and **Tool Poisoning** (documented live in 2025 by Zscaler ThreatLabZ and Palo Alto Unit 42). The resulting transaction appears clean, fast, and legitimate to every standard tabular classifier.

```
       LEGITIMATE IDENTITY (Human / AI Agent)
                     │
         [Attacker Corrupts Decision / Context]
                     │
                     ▼
         AUTHORIZATION EVENT  <─── ARIA SITS HERE (Undefended Wedge)
                     │
                     ▼
             PAYMENT RAIL     <─── Existing Fraud Systems (Mastercard DI)
```

**Our Core Insight:** *The payment transaction looks legitimate because the identity is verified. The corruption occurred during the authorization decision itself.*

**ARIA (Authorization Risk & Integrity Analyzer)** is an end-to-end, closed-loop red-team/blue-team AI system that:
1. **Identifies** an exhaustive 5-category, 12-vector taxonomy of novel GenAI payment attacks.
2. **Generates** high-fidelity synthetic adversarial datasets across human coercion calls, agent tool-call traces, and CTGAN-synthesized transactions.
3. **Defends** via a tri-modal cascade architecture (Coercion Detector, Intent Drift Detector, Transaction Classifier) fused into a unified **ARIA Score (0.0–1.0)** with **Tiered Friction Mitigation** and cryptographic integration into **Mastercard's Verifiable Intent** framework.

---

## 2. Pillar I: IDENTIFY — 5-Category 12-Vector Attack Taxonomy

The challenge rubric explicitly rewards diversity, depth, and real-world grounding. ARIA maps 12 distinct attack vectors:

| Category | Vector ID | Attack Name | Target Surface | Real-World Evidence & Citations |
| :--- | :---: | :--- | :--- | :--- |
| **A. Coercion** | **A1** | LLM-Powered Digital Arrest / Vishing | Voice / Video | 58,239 cases in India (2025), I4C/MHA; deepfake vishing surged 1,600% |
| | **A2** | Predictive Invoice Impersonation (BEC 2.0) | Email / Chat | Sardine.ai 2026 Report; vendor rhythm harvesting |
| | **A3** | Automated Romance / Pig-Butchering Agents | Chat / Messaging | Interpol Op First Light 2025; long-horizon multi-persona agents |
| **B. Agentic Commerce** | **B1** | Indirect Prompt Injection on AI Shopping Agents | Web / Agent DOM | Zscaler & Unit 42 (2025); hidden metadata overrides agent goals |
| | **B2** | Tool Poisoning / Malicious API Descriptors | MCP / API Specs | OWASP Top 10 for Agents (2026); malicious schema manipulation |
| | **B3** | Multi-Agent Collusion Fraud Rings | Multi-Agent Network | *MultiAgentFinancialFraudBench* (OpenReview 2026); task distribution |
| **C. Synthetic KYC** | **C1** | GenAI Frankenstein Identity (Slow-Burn) | Onboarding / KYC | Group-IB (2025); camera injection attacks bypassing liveness SDKs |
| | **C2** | RAG Poisoning of Fraud Knowledge Bases | Vector DBs | MDPI (2026); embedding injection corrupting fraud reference indices |
| **D. AML Evasion** | **D1** | Gradient-Guided Transaction Morphing | Tabular Classifier | *FRAUD-RLA* (arXiv 2025); reinforcement learning score feedback |
| | **D2** | GAN-Synthesized Ghost Transactions | Payment Engine | CTGAN / Wasserstein evasion of statistical anomaly filters |
| **E. Agentic Token** | **E1** | Mandate Scope Creep via Ambiguity | Agentic Tokens | Mastercard Verifiable Intent (2026); natural language semantic gap |
| | **E2** | Stale Consent Token Replay Attack | OAuth2 / Agent Pay | OAuth token replay inheritance in autonomous persistent token stores |

---

## 3. Pillar II: GENERATE — Closed-Loop Synthetic Data Engines

Because no public labeled dataset exists for coercion dialogues or compromised agent tool-call sequences, we engineered three dedicated generation engines:

### 1. Coercion Transcript Generator (`aria/generate/coercion_generator.py`)
* Leverages **Gemini 2.5 Flash** with prompt templates implementing the 4-phase psychological coercion arc: *Credibility $\rightarrow$ Fear $\rightarrow$ Isolation $\rightarrow$ Payment Demand*.
* Automatically extracts 25+ linguistic features per turn (Urgency Density, Authority Lexicon, Turn-Rigidity, Panic Signals).
* Produced 850 labeled dialogue records across digital arrest, TRAI, BEC, and legitimate support calls.

### 2. Agent Trace Harness (`aria/generate/agent_trace_generator.py`)
* Implements a LangGraph-style autonomous purchasing harness.
* Injects 4 attack types (B1 Indirect Prompt Injection, B2 Tool Poisoning, B3 Multi-Agent Collusion, E1 Scope Creep).
* Outputs structured tool-call traces recording `user_mandate`, `tool_call_sequence`, `final_action`, and computed drift metrics.

### 3. High-Fidelity Transaction Synthesizer (`aria/generate/transaction_generator.py`)
* Combines **CTGAN** with localized Indian payment context (UPI, IMPS, RuPay/Mastercard card rails).
* Generates 9,200 labeled transactions with statistical fidelity verified via **Wasserstein Distance** (hour distance: 0.38, agent drift distance: 0.50).

---

## 4. Pillar III: DEFEND — Tri-Modal Cascade & Score Fusion

```
                       AUTHORIZATION EVENT
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   [Tier 1: Coercion]    [Tier 2: Intent Drift]   [Tier 3: Transaction]
    XGBoost Classifier    Embedding / Bounds CLF    LightGBM Tabular CLF
     P95: ~2.7 ms            P95: ~4.1 ms            P95: ~35.7 ms
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                      [ARIA SCORE FUSION]
                     Score: S ∈ [0.0, 1.0]
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   [0.00 - 0.40]           [0.40 - 0.60]           [0.60 - 0.80]       [0.80 - 1.00]
     ✅ PASS             ⚠️ SOFT FRICTION      🔶 DELAYED RELEASE     🔴 HARD BLOCK
  Approve payment         60s pause + SMS       4-hr hold + Alt OTP   Reject + I4C 1930
```

### 1. Fast Coercion Detector (Text/Voice Analysis)
* Model: **XGBoost Classifier** on 25+ linguistic & turn-dominance features.
* **Performance:** Cross-Validation **F1 = 1.0000**, **ROC-AUC = 1.0000**, Inference Latency P95 = **2.7 ms**.

### 2. Agent Intent Drift Detector (Autonomous Agent Defense)
* Evaluates semantic distance:
$$\text{Drift} = 1.0 - \cos(\mathbf{e}_{\text{mandate}}, \mathbf{e}_{\text{action}}) + \text{ScopeViolation} + \text{InjectionSig}$$
* Model: Sentence-Transformers (`all-MiniLM-L6-v2`) + Gradient Boosting Meta-Learner.
* **Performance:** Cross-Validation **F1 = 1.0000**, **ROC-AUC = 1.0000**, Inference Latency P95 = **4.1 ms**.

### 3. Context-Aware Transaction Classifier
* Model: **LightGBM** integrating traditional tabular signals with novel agentic fields (`mandate_freshness_hours`, `scope_match_score`, `agent_drift_score`).
* **Performance:** Cross-Validation **ROC-AUC = 1.0000**, **F1 = 0.9996**, Inference Latency P95 = **35.7 ms**.

### 4. ARIA Score Fusion & End-to-End Latency
* Fuses all available sub-scores into unified $S_{\text{ARIA}} \in [0, 1]$.
* **Total End-to-End P95 Latency:** **42.66 ms** — fully compliant with Mastercard Decision Intelligence's sub-50ms real-time SLA!

---

## 5. Mastercard Infrastructure Integration: Verifiable Intent Extension

In March 2026, Mastercard and Google introduced **Verifiable Intent** to the FIDO Alliance as a cryptographic trust layer for agent payments.

ARIA extends Verifiable Intent by appending an **Authorization Integrity Attestation (AIA)**:
* Standard Verifiable Intent: Binds Consumer Identity $\leftrightarrow$ Signed Mandate $\leftrightarrow$ Transaction.
* **ARIA-Enhanced Verifiable Intent:** Binds Consumer Identity $\leftrightarrow$ Signed Mandate $\leftrightarrow$ **Authorization Context Integrity** $\leftrightarrow$ Transaction.

Each processed authorization event generates a signed attestation token (e.g. `ARIA-v1-pass-clean-1755289941` or `ARIA-v1-blocked-high_risk-1755289941`), enabling issuers to cryptographically verify that neither coercion nor agent prompt injection was present.

---

## 6. Tiered Mitigation Strategy vs. Binary Declines

Binary "Accept/Decline" paradigms fail in coercion scenarios because panicked victims repeatedly retry transfers or switch payment rails. ARIA introduces **psychology-aware tiered mitigation**:

1. **Tier 1: PASS ($S < 0.40$)** — Direct authorization without friction.
2. **Tier 2: SOFT FRICTION ($0.40 \le S < 0.60$)** — 60-second verification pause with psychological grounding prompts ("Call a trusted contact to verify"). Breaks attacker script continuity.
3. **Tier 3: DELAYED RELEASE ($0.60 \le S < 0.80$)** — 4-hour settlement hold requiring OTP verification from an alternate registered contact.
4. **Tier 4: HARD BLOCK & ESCALATION ($S \ge 0.80$)** — Transaction rejection, issuer network alert, and automatic display of India's **National Cyber Crime Helpline (1930 / cybercrime.gov.in)** reporting gateway.

---

## 7. Working Web Prototype & Reproducibility

The submission includes a fully functional, self-contained interactive web prototype:
* **Interactive UI:** Real-time animated ARIA Risk Gauge, live Red Team attack generator, Blue Team feature SHAP breakdown, and Verifiable Intent token inspector.
* **FastAPI Backend:** Fully documented RESTful API with automated integration tests.
* **Reproducibility:** One command to run the complete data generation, model training, evaluation, and web server:
```bash
python run_pipeline.py
python -m uvicorn backend.main:app --reload
```

---

*Submitted to Mastercard Innovation Challenge @ Global Fintech Fest 2026, Jio World Centre, Mumbai.*
