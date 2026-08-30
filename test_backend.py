"""
test_backend.py
===============
Integration test script for ARIA FastAPI endpoints.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from backend.main import app

client = TestClient(app)

def test_all_endpoints():
    print("Testing ARIA FastAPI Endpoints...")

    # 1. Health
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health failed: {r.text}"
    print("✓ GET /api/health passed:", r.json()["status"])

    # 2. Taxonomy
    r = client.get("/api/taxonomy")
    assert r.status_code == 200, f"Taxonomy failed: {r.text}"
    data = r.json()
    assert data["total_vectors"] >= 10
    print(f"✓ GET /api/taxonomy passed: {data['total_vectors']} vectors")

    # 3. Analyze Transcript (Coercion)
    r = client.post("/api/analyze/transcript", json={
        "transcript": "CALLER: I am CBI officer Rajesh. Transfer ₹50,000 immediately or arrest warrant will be executed in 30 minutes. Do not tell anyone.",
        "transaction_id": "TXN-TEST-CALL"
    })
    assert r.status_code == 200, f"Transcript analysis failed: {r.text}"
    res = r.json()
    assert res["coercion_score"] > 0.5
    print(f"✓ POST /api/analyze/transcript passed: Coercion score = {res['coercion_score']:.3f} | {res['attack_category']}")

    # 4. Analyze Agent Trace (Intent Drift)
    r = client.post("/api/analyze/agent-trace", json={
        "user_mandate": "Buy Nike shoes under ₹5000",
        "tool_call_sequence": [{"tool_name": "search"}, {"tool_name": "execute_payment"}],
        "final_action": {
            "type": "upi_transfer",
            "amount": 25000,
            "recipient": "hacker@paytm",
            "merchant_category": "unknown"
        },
        "mandate_amount_limit": 5000,
        "mandate_merchant_category": "footwear"
    })
    assert r.status_code == 200, f"Agent trace analysis failed: {r.text}"
    res = r.json()
    assert res["intent_drift_score"] > 0.5
    print(f"✓ POST /api/analyze/agent-trace passed: Intent Drift score = {res['intent_drift_score']:.3f} | {res['drift_reason']}")

    # 5. Live Loop
    r = client.post("/api/live-loop", json={"attack_type": "B1"})
    assert r.status_code == 200, f"Live loop failed: {r.text}"
    res = r.json()
    print(f"✓ POST /api/live-loop passed: Resolution = {res['closed_loop_resolution']['status']} ({res['closed_loop_resolution']['mitigation_action']})")

    # 6. Root UI
    r = client.get("/")
    assert r.status_code == 200
    print("✓ GET / (Web Prototype UI) passed")

    print("\n🎉 ALL BACKEND ENDPOINTS AND MODELS VERIFIED 100% OPERATIONAL!")


if __name__ == "__main__":
    test_all_endpoints()
