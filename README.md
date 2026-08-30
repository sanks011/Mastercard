# 🔴 ARIA — Authorization Risk & Integrity Analyzer
### *Mastercard Innovation Challenge @ GFF 2026 — AI Defense Lab for Payment Security*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-eb001b.svg)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5+-ff5f00.svg)](https://lightgbm.readthedocs.io)
[![Mastercard Verifiable Intent](https://img.shields.io/badge/Mastercard-Verifiable%20Intent%20Ready-f79e1b.svg)](https://www.mastercard.com)

---

## 🎯 The Core Thesis

Traditional payment fraud solutions evaluate the **transaction** (amount, merchant, velocity, card present).  
In 2026, GenAI fraud operates by manipulating the **authorization decision** before the payment is sent:
* **Human Coercion:** Digital Arrest & Deepfake Vishing (₹22,495 Cr lost in India in 2025 across 28.15 lakh cases).
* **Agentic Commerce Hijacking:** Prompt Injection & Tool Poisoning subverting autonomous purchasing agents.

**ARIA defends the authorization event itself, sitting as an ultra-fast trust layer (<50ms) before existing fraud engines.**

```
         ┌────────────────────────────────────────────────────────┐
         │                      ARIA SYSTEM                       │
         │                                                        │
         │  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
         │  │   IDENTIFY   │──▶│   GENERATE   │──▶│   DEFEND   │  │
         │  │  12 Vectors  │   │  3 Synthetic │   │ Tri-Modal  │  │
         │  │  5 Taxonomy  │   │ Data Engines │   │ + Cascade  │  │
         │  │  Categories  │   │  Transcripts │   │ ARIA Score │  │
         │  │              │   │ Traces & CSV │   │ & Friction │  │
         │  └──────────────┘   └──────────────┘   └────────────┘  │
         │         ▲                                     │        │
         │         └─────────────────────────────────────┘        │
         │                  Closed-Loop Feedback                  │
         └────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart (1-Minute Setup)

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/ARIA-Mastercard-Defense-Lab.git
cd ARIA-Mastercard-Defense-Lab

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
source .venv/bin/activate      # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Closed-Loop Pipeline (Generate $\rightarrow$ Train $\rightarrow$ Benchmark)
```bash
python run_pipeline.py
```
*Outputs: 850 coercion dialogues, 600 agent traces, 9,200 transactions, trained models in `models/`, and `benchmark_results.json`.*

### 3. Launch the Interactive Web Prototype Dashboard
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to experience:
* ⚡ **Live Closed-Loop Simulation** (Adversarial Red Team Generator $\leftrightarrow$ Blue Team Defender)
* 🔵 **Blue Team Analyzer** with real-time SHAP feature breakdown & gauge visualization
* 🔴 **Red Team Synthetic Payload Studio**
* 📚 **12-Vector Attack Taxonomy Browser** with evidence citations
* 📊 **Latency & Evaluation Benchmark Analytics**

---

## 📊 Benchmark Results

| Model Component | Architecture | Evaluation Metric | Inference Latency (P95) |
| :--- | :--- | :---: | :---: |
| **Coercion Detector** | XGBoost (25+ Linguistic Signals) | **F1 = 1.0000** | **2.7 ms** |
| **Intent Drift Detector** | Sentence-Transformers + GradBoost | **F1 = 1.0000** | **4.1 ms** |
| **Transaction Classifier** | LightGBM + Agent Context | **ROC-AUC = 1.0000** | **35.7 ms** |
| **ARIA Fused Engine** | Cascaded Meta-Fusion | **End-to-End** | **42.66 ms** (<50ms SLA) |

---

## 📁 Repository Structure

```
.
├── aria/
│   ├── identify/
│   │   └── attack_taxonomy.py         # 5-Category 12-Vector Attack Taxonomy
│   ├── generate/
│   │   ├── coercion_generator.py      # LLM Coercion Transcript Generator (Gemini Flash)
│   │   ├── agent_trace_generator.py   # LangGraph Agent Hijacking Trace Engine
│   │   └── transaction_generator.py   # CTGAN Tabular Payment Generator
│   ├── defend/
│   │   ├── coercion_detector.py       # XGBoost Coercion Detector
│   │   ├── intent_drift_detector.py   # Embedding-based Agent Intent Drift Detector
│   │   ├── transaction_classifier.py  # LightGBM Context-Aware Classifier
│   │   └── aria_score.py              # ARIA Score Fusion + Tiered Response Engine
│   └── utils/
│       └── evaluation.py              # Benchmark & Latency Profiler
├── backend/
│   ├── main.py                        # FastAPI Application & Endpoints
│   ├── schemas.py                     # Pydantic Schemas
│   └── static/
│       └── index.html                 # Interactive Glassmorphism Web Prototype UI
├── data/                              # Generated synthetic datasets
├── models/                            # Serialized trained defense models (.pkl)
├── run_pipeline.py                    # Master pipeline runner
├── test_backend.py                    # End-to-end API & model integration tests
├── KAGGLE_WRITEUP.md                  # Official competition submission writeup
├── hackathon_strategy.md              # Research & design brief
└── requirements.txt
```

---

## 🏆 Mastercard Innovation Challenge Submission

* **Code Repository:** Public runnable repository with reproducible pipelines.
* **Solution Walkthrough:** Available in [`KAGGLE_WRITEUP.md`](./KAGGLE_WRITEUP.md) and [`hackathon_strategy.md`](./hackathon_strategy.md).
* **Working Prototype:** High-performance web prototype accessible at `/`.
