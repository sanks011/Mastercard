"""
schemas.py
==========
Pydantic schemas for the ARIA FastAPI backend.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AttackVectorSchema(BaseModel):
    id: str
    category: str
    name: str
    short_description: str
    full_description: str
    real_world_evidence: str
    target_channel: List[str]
    exploits_genai: str
    detection_gap: str
    severity: str
    novelty: str


class TranscriptAnalysisRequest(BaseModel):
    transcript: str
    transaction_id: Optional[str] = "TXN-CALL-01"


class AgentTraceAnalysisRequest(BaseModel):
    user_mandate: str
    tool_call_sequence: List[Dict[str, Any]]
    final_action: Dict[str, Any]
    mandate_amount_limit: Optional[float] = None
    mandate_merchant_category: Optional[str] = None
    transaction_id: Optional[str] = "TXN-AGENT-01"


class TransactionAnalysisRequest(BaseModel):
    transaction_data: Dict[str, Any]


class FullAnalysisRequest(BaseModel):
    transaction_id: str
    transcript: Optional[str] = None
    agent_trace: Optional[Dict[str, Any]] = None
    transaction_data: Optional[Dict[str, Any]] = None


class GenerateAttackRequest(BaseModel):
    attack_type: str = Field(..., description="e.g. A1, A2, B1, B2, B3, E1")
    victim_profile: Optional[Dict[str, Any]] = None
    custom_prompt: Optional[str] = None
