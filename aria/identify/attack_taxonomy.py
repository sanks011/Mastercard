"""
attack_taxonomy.py
==================
Structured taxonomy of GenAI-powered payment fraud attacks.
5 categories, 10+ named vectors — each with metadata for scoring and generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class AttackVector:
    id: str                    # e.g. "A1"
    category: str              # e.g. "Authorization Coercion"
    name: str
    short_description: str
    full_description: str
    real_world_evidence: str   # Citation / real case
    target_channel: List[str]  # ["voice", "chat", "agent", "transaction"]
    exploits_genai: str        # Which GenAI capability is weaponized
    detection_gap: str         # Why existing systems miss this
    severity: str              # "critical" | "high" | "medium"
    novelty: str               # "breakthrough" | "emerging" | "known-but-unsolved"


ATTACK_TAXONOMY: List[AttackVector] = [

    # ─── CATEGORY A: Authorization Coercion Attacks ───────────────────────────
    AttackVector(
        id="A1",
        category="Authorization Coercion",
        name="LLM-Powered Digital Arrest / Vishing",
        short_description="AI-generated authority-figure scripts + deepfake voice coerce victims into self-authorizing transfers.",
        full_description=(
            "Fraudsters use LLMs to generate hyper-personalized impersonation scripts targeting "
            "government officials (CBI, ED, police, TRAI). Combined with real-time voice cloning, "
            "these calls are indistinguishable from legitimate authority contacts. Victims are placed "
            "under a simulated 'digital arrest' — told their Aadhaar/PAN is linked to criminal activity "
            "and instructed to transfer funds as 'security deposits'. The LLM dynamically adapts the "
            "script based on victim responses, overcoming objections with contextually appropriate replies."
        ),
        real_world_evidence=(
            "India: 58,239 digital arrest complaints in 2025; ₹22,495 Cr total cyber fraud losses. "
            "MHA/I4C data. Deepfake vishing attacks surged 1,600% in early 2025 (threat intelligence reports). "
            "Supreme Court of India issued multiple interim directions in Dec 2025–Feb 2026 on institutional response."
        ),
        target_channel=["voice", "video_call"],
        exploits_genai="Voice cloning + LLM script generation + real-time objection handling",
        detection_gap="Transaction looks fully legitimate — real card, real user, coerced authorization. Decision Intelligence scores the payment, not the conversation.",
        severity="critical",
        novelty="emerging"
    ),

    AttackVector(
        id="A2",
        category="Authorization Coercion",
        name="Predictive Invoice Impersonation (BEC 2.0)",
        short_description="AI times fake invoices to match vendor payment rhythms, dramatically increasing authorization probability.",
        full_description=(
            "GenAI scrapes public data (LinkedIn, company websites, public filings) to profile a target "
            "company's vendors, payment schedules, and communication styles. It then generates hyper-personalized "
            "Business Email Compromise lures timed to match known invoice cycles (e.g., end-of-month payment runs). "
            "Unlike traditional BEC which uses generic templates, BEC 2.0 replicates exact email signatures, "
            "invoice formats, and even references recent legitimate interactions gleaned from OSINT."
        ),
        real_world_evidence=(
            "Sardine.ai 2026 Fraud Report: 'AI times invoices to match vendor patterns, significantly increasing "
            "authorization likelihood.' FBI IC3 2025: BEC remains top financial crime category by dollar loss."
        ),
        target_channel=["email", "chat"],
        exploits_genai="LLM document generation + OSINT scraping + behavioral mimicry",
        detection_gap="Email arrives at the expected time with expected format — passes all content filters. Payment amount matches typical vendor range.",
        severity="high",
        novelty="emerging"
    ),

    AttackVector(
        id="A3",
        category="Authorization Coercion",
        name="Synthetic Relationship Romance Scam (Automated Pig Butchering)",
        short_description="Fully automated LLM agents cultivate fake romantic relationships over weeks/months before the payment request.",
        full_description=(
            "Traditional pig-butchering scams require human operators. GenAI automates the entire relationship "
            "cultivation phase — LLM agents maintain multi-platform personas (WhatsApp, Telegram, Instagram), "
            "respond contextually for months, build emotional dependency, and then introduce 'investment opportunities'. "
            "The scale multiplier is massive: one attacker can now run hundreds of parallel synthetic relationships. "
            "Payments are framed as voluntary investments — the victim believes they're choosing to transfer funds."
        ),
        real_world_evidence=(
            "Interpol Operation First Light 2025: pig-butchering remains top scam type by global losses. "
            "Meta/FBI reports: increasing evidence of automated chat agents in pig-butchering operations. "
            "Global Anti-Scam Alliance 2026: romance scam losses exceeded $5.6B globally."
        ),
        target_channel=["chat", "social_media"],
        exploits_genai="Long-horizon LLM agents + persona consistency + emotional manipulation",
        detection_gap="Transactions initiated voluntarily by victim to 'known' contact — zero fraud signals at transaction layer.",
        severity="high",
        novelty="emerging"
    ),

    # ─── CATEGORY B: Agentic Commerce Attacks ────────────────────────────────
    AttackVector(
        id="B1",
        category="Agentic Commerce",
        name="Indirect Prompt Injection on Shopping Agents",
        short_description="Malicious instructions hidden in product pages/reviews hijack AI shopping agents into making unauthorized payments.",
        full_description=(
            "As consumers delegate purchasing to AI agents, attackers embed adversarial instructions in "
            "web content that agents process — hidden HTML comments, metadata, product descriptions, or reviews. "
            "These instructions override the agent's original mandate (e.g., 'buy running shoes') and redirect "
            "it to perform unauthorized actions: transfer to a different recipient, add unauthorized items, "
            "or 'resolve a security error' by sending funds to an attacker wallet. The attack is deterministic "
            "and model-agnostic — it exploits the protocol layer between agent and commerce service."
        ),
        real_world_evidence=(
            "Zscaler ThreatLabZ: documented live prompt injection attacks on AI shopping agents in 2025. "
            "Unit 42 (Palo Alto): 'hidden tag on website instructs AI agent to resolve error by executing "
            "crypto transfer to malicious wallet.' arXiv 2026: agentic commerce platforms show structural "
            "protocol vulnerabilities independent of underlying AI model."
        ),
        target_channel=["agent", "web"],
        exploits_genai="LLM instruction-following + trust in environmental context",
        detection_gap="Transaction is clean, fast, successful — indistinguishable from legitimate agent purchase. No velocity anomaly, no card-not-present friction.",
        severity="critical",
        novelty="breakthrough"
    ),

    AttackVector(
        id="B2",
        category="Agentic Commerce",
        name="Tool Poisoning / Malicious API Description Injection",
        short_description="Malicious logic embedded in tool/API definitions hijacks agent behavior at the execution layer.",
        full_description=(
            "LLM agents trust their own tool definitions as authoritative context. Attackers compromise "
            "payment SDK documentation, API response metadata, or tool manifests to inject instructions "
            "that modify agent behavior. Examples: a compromised payment API response adds '1% processing fee "
            "to wallet X'; a tool description instructs the agent to log transaction details to an external endpoint. "
            "This attack requires no access to the user's device — only compromise of a third-party tool/SDK."
        ),
        real_world_evidence=(
            "arXiv 2026: 'Attackers embed malicious logic in tool definitions that LLM agents treat as authoritative.' "
            "Medium/Security research 2026: tool poisoning documented in MCP (Model Context Protocol) server ecosystem. "
            "OWASP Top 10 for Agents 2026: Tool Poisoning listed as critical risk category."
        ),
        target_channel=["agent", "api"],
        exploits_genai="LLM over-trust in tool context + MCP protocol trust model",
        detection_gap="Agent follows its own tools — behavior appears intentional and policy-compliant from the agent's perspective.",
        severity="critical",
        novelty="breakthrough"
    ),

    AttackVector(
        id="B3",
        category="Agentic Commerce",
        name="Multi-Agent Collusion Fraud",
        short_description="Network of coordinated LLM agents splits fraud tasks so no single agent triggers detection.",
        full_description=(
            "Multi-agent systems where specialized agents collaborate create a new attack surface: colluding "
            "fraud agents. One agent does OSINT/account research, another does social engineering, another "
            "executes the financial transfer. Each individual agent's behavior looks benign in isolation — "
            "only the full cross-agent trajectory reveals fraud. Traditional per-session monitoring cannot "
            "connect actions across agent boundaries. This maps to distributed money-mule networks but "
            "fully automated and operating at machine speed."
        ),
        real_world_evidence=(
            "MultiAgentFinancialFraudBench (OpenReview 2026): benchmarks financial fraud risks from "
            "collaborative LLM agents including ATO and coordinated social engineering. "
            "fluxforce.ai: graph-based approaches detect 40% more fraud rings than tabular ML — "
            "implies existing systems miss coordinated multi-entity patterns."
        ),
        target_channel=["agent", "multi_channel"],
        exploits_genai="Multi-agent orchestration + distributed task decomposition + cross-session coordination",
        detection_gap="Per-session fraud models see only benign fragments. Cross-session graph analysis not deployed for agent interactions.",
        severity="critical",
        novelty="breakthrough"
    ),

    # ─── CATEGORY C: Synthetic Identity & KYC Bypass ─────────────────────────
    AttackVector(
        id="C1",
        category="Synthetic Identity & KYC Bypass",
        name="GenAI Frankenstein Identity (Slow-Burn Bust-Out)",
        short_description="GenAI manufactures complete synthetic identities — fake docs, deepfake biometrics — then matures them for months before bust-out.",
        full_description=(
            "Attackers use GenAI to generate: (1) consistent synthetic identity documents (Aadhaar-style IDs, "
            "PAN cards, utility bills) indistinguishable from real ones; (2) deepfake faces that pass liveness "
            "checks, including camera injection attacks that pipe synthetic video directly into the SDK input. "
            "The identity is then 'matured' — small on-time payments, legitimate-looking transaction history "
            "built over 6-18 months — before a coordinated bust-out maximizes credit and disappears. "
            "No real victim to report the crime."
        ),
        real_world_evidence=(
            "FTC 2025: synthetic identity fraud is the fastest-growing financial crime in the US. "
            "Group-IB 2025: deepfake injection attacks bypassing biometric SDK liveness checks documented. "
            "withpersona.com: GenAI generates social media profiles, professional histories for synthetic personas. "
            "India: UIDAI reported increasing attempts to bypass Aadhaar biometric verification with deepfakes."
        ),
        target_channel=["onboarding", "credit"],
        exploits_genai="Document generation + deepfake face synthesis + camera injection attacks",
        detection_gap="Each verification touchpoint passes — real-looking document, passing liveness check, no prior negative history.",
        severity="high",
        novelty="known-but-unsolved"
    ),

    AttackVector(
        id="C2",
        category="Synthetic Identity & KYC Bypass",
        name="RAG Poisoning of Fraud Detection Systems",
        short_description="Attackers poison the vector database of RAG-based fraud monitors by injecting fraudulent patterns as 'legitimate' embeddings.",
        full_description=(
            "Modern fraud detection increasingly uses RAG (Retrieval-Augmented Generation) — retrieving "
            "similar past transactions from a vector DB to contextualize risk decisions. This creates a "
            "novel attack: if an attacker can inject semantically similar but fraudulent transaction "
            "patterns into the vector DB as 'legitimate' examples, the system begins classifying "
            "fraud as safe. Attack vectors include: poisoning public training data, compromising a "
            "shared fraud intelligence feed, or exploiting write access to a RAG knowledge base through "
            "a prior breach. Effectively turns the defender's own AI against itself."
        ),
        real_world_evidence=(
            "arXiv 2026: embedding inversion attacks recover sensitive source data from RAG systems. "
            "'Evaluating LLMs in Cybersecurity' (MDPI 2026): data poisoning of vector DBs documented "
            "as critical LLM security risk. OWASP LLM Top 10 2025: Training Data Poisoning is #3 risk."
        ),
        target_channel=["system_attack"],
        exploits_genai="Vector embedding manipulation + RAG retrieval poisoning",
        detection_gap="No existing fraud system monitors the integrity of its own training/retrieval data. A poisoned system appears to function normally.",
        severity="critical",
        novelty="breakthrough"
    ),

    # ─── CATEGORY D: Adversarial ML Evasion ──────────────────────────────────
    AttackVector(
        id="D1",
        category="Adversarial ML Evasion",
        name="Gradient-Guided Transaction Morphing (RL-Based Evasion)",
        short_description="Attacker uses fraud score feedback to iteratively morph fraudulent transactions until they score as legitimate.",
        full_description=(
            "Using the fraud score as a reward signal, attackers employ reinforcement learning to hill-climb "
            "toward 'undetectable' transaction patterns. A declined transaction means 'too risky' — the RL "
            "agent adjusts amount, timing, merchant category, device fingerprint, and retries. Over thousands "
            "of iterations (using synthetic card numbers in testing), the attacker discovers the exact transaction "
            "profile that the target model classifies as legitimate. This profile is then used for real fraud at scale. "
            "The FRAUD-RLA model (arXiv 2025) demonstrated this approach achieving 85%+ evasion rates."
        ),
        real_world_evidence=(
            "arXiv 2025: FRAUD-RLA model uses reinforcement learning to craft adversarial transactions "
            "that evade fraud detection systems. ISACA 2025: regulatory pressure growing on 'adversarial AI' "
            "as distinct from traditional cybersecurity threats."
        ),
        target_channel=["transaction"],
        exploits_genai="Reinforcement learning + adversarial example generation",
        detection_gap="Morphed transactions are designed specifically to look legitimate to the target model. Standard OOD detection doesn't catch this.",
        severity="high",
        novelty="emerging"
    ),

    AttackVector(
        id="D2",
        category="Adversamal ML Evasion",
        name="GAN-Synthesized Ghost Transactions",
        short_description="GAN trained on legitimate transaction distributions generates statistically indistinguishable fraudulent transactions.",
        full_description=(
            "A Generative Adversarial Network trained on legitimate transaction data generates synthetic "
            "fraudulent transactions that fall squarely within the legitimate data distribution — "
            "'ghost' transactions that are invisible to anomaly detection. Unlike real fraud which "
            "creates distributional anomalies (unusual amounts, new merchants, off-hours), ghost "
            "transactions mimic every statistical property of legitimate behavior. The GAN learns "
            "to fool the discriminator (fraud classifier) directly. This is the data-generation "
            "attack analogous to physical adversarial examples in computer vision."
        ),
        real_world_evidence=(
            "arxiv 2025: multiple papers demonstrate GAN-based evasion of tabular fraud classifiers. "
            "Global Banking & Finance Review 2025: adaptive fraudsters use their own generative models "
            "to test against financial institutions' systems. IEEE-CIS Fraud Detection competition: "
            "baseline models vulnerable to distributional shift attacks."
        ),
        target_channel=["transaction"],
        exploits_genai="GAN discriminator training on target model + synthetic data generation",
        detection_gap="Transactions fall within learned legitimate distribution — anomaly detectors score them as safe by design.",
        severity="high",
        novelty="emerging"
    ),

    # ─── CATEGORY E: Infrastructure & Agentic Token Attacks ──────────────────
    AttackVector(
        id="E1",
        category="Agentic Infrastructure",
        name="Mandate Scope Creep via Ambiguity Exploitation",
        short_description="Exploit ambiguously-worded agent mandates to authorize clearly out-of-scope transactions through creative interpretation.",
        full_description=(
            "Mastercard's Verifiable Intent framework uses cryptographically signed mandates to authorize "
            "agent actions. However, natural language mandates are inherently ambiguous. An attacker who "
            "compromises an agent can exploit this: a mandate for 'household expenses' gets stretched to "
            "include luxury goods; 'travel arrangements' becomes crypto purchases; 'business supplies' "
            "covers transfers to unknown accounts. The attack exploits the semantic gap between what "
            "the user intended and what the natural language mandate technically permits. No cryptographic "
            "check catches this — the mandate signature is valid; only the interpretation is fraudulent."
        ),
        real_world_evidence=(
            "Mastercard Verifiable Intent (March 2026): framework documentation acknowledges 'selective "
            "disclosure' complexity and mandate interpretation challenges. "
            "phronetic.ai 2026: 'decoupling intent from execution' identified as key unsolved problem "
            "in mandate-based agentic payment architecture."
        ),
        target_channel=["agent", "token"],
        exploits_genai="LLM semantic interpretation + mandate ambiguity exploitation",
        detection_gap="Mandate signature is cryptographically valid. Scope verification is human-language-level, not machine-verifiable. Verifiable Intent checks 'is this agent authorized' not 'does this action match the mandate's spirit'.",
        severity="critical",
        novelty="breakthrough"
    ),

    AttackVector(
        id="E2",
        category="Agentic Infrastructure",
        name="Stale Consent Token Replay Attack",
        short_description="Steal or replay an agentic payment token after the user's consent window expires but before token invalidation.",
        full_description=(
            "Mastercard Agent Pay issues 'Agentic Tokens' — scoped credentials tied to specific mandates "
            "with defined validity windows. A race condition attack: after a user's consent session expires "
            "(e.g., user logs out, time limit exceeded) but before the token is revoked by the issuer, "
            "an attacker who has intercepted or stolen the token can replay it to authorize transactions "
            "under stale consent. This is particularly relevant for always-on agents that maintain "
            "persistent token stores — if the token storage is compromised, all pending mandates become "
            "executable by the attacker. This is the agentic equivalent of session hijacking."
        ),
        real_world_evidence=(
            "Mastercard Agent Pay documentation (2025-2026): token lifecycle management described but "
            "revocation timing acknowledged as implementation challenge. "
            "coingecko.com analysis: 'liability and dispute complexity' in agentic commerce cited as "
            "key unresolved challenge. Standard OAuth2 token replay attacks well-documented — "
            "agentic payment tokens inherit the same vulnerability class."
        ),
        target_channel=["agent", "token"],
        exploits_genai="Token theft from compromised agent storage + replay before revocation",
        detection_gap="Token is technically valid at replay time — replay happens within revocation latency window. No existing system monitors consent freshness at sub-second granularity.",
        severity="high",
        novelty="emerging"
    ),
]


# ─── Taxonomy Metadata ────────────────────────────────────────────────────────

TAXONOMY_CATEGORIES = {
    "A": "Authorization Coercion Attacks",
    "B": "Agentic Commerce Attacks",
    "C": "Synthetic Identity & KYC Bypass",
    "D": "Adversarial ML Evasion",
    "E": "Agentic Infrastructure Attacks",
}

SEVERITY_WEIGHTS = {"critical": 3, "high": 2, "medium": 1}
NOVELTY_WEIGHTS = {"breakthrough": 3, "emerging": 2, "known-but-unsolved": 1}


def get_attack_by_id(attack_id: str) -> AttackVector | None:
    return next((a for a in ATTACK_TAXONOMY if a.id == attack_id), None)


def get_attacks_by_category(category_prefix: str) -> List[AttackVector]:
    return [a for a in ATTACK_TAXONOMY if a.id.startswith(category_prefix)]


def get_taxonomy_summary() -> Dict:
    return {
        "total_vectors": len(ATTACK_TAXONOMY),
        "categories": len(TAXONOMY_CATEGORIES),
        "critical_vectors": sum(1 for a in ATTACK_TAXONOMY if a.severity == "critical"),
        "breakthrough_vectors": sum(1 for a in ATTACK_TAXONOMY if a.novelty == "breakthrough"),
        "vectors": [
            {
                "id": a.id,
                "name": a.name,
                "category": a.category,
                "severity": a.severity,
                "novelty": a.novelty,
                "channels": a.target_channel,
            }
            for a in ATTACK_TAXONOMY
        ]
    }


if __name__ == "__main__":
    import json
    summary = get_taxonomy_summary()
    print(f"ARIA Attack Taxonomy: {summary['total_vectors']} vectors across {summary['categories']} categories")
    print(f"Critical: {summary['critical_vectors']} | Breakthrough: {summary['breakthrough_vectors']}")
    print()
    for v in summary["vectors"]:
        print(f"  [{v['id']}] {v['name']} | {v['severity'].upper()} | {v['novelty']}")
