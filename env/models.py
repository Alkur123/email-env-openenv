from pydantic import BaseModel
from typing import List, Optional


# =========================
# EMAIL MODEL
# =========================
class Email(BaseModel):
    id: int
    subject: str
    body: str
    true_label: str  # spam / urgent / normal
    priority: str    # high / medium / low

    # ✅ CRITICAL (YOU WERE MISSING THIS)
    expected_response: Optional[str] = None


# =========================
# OBSERVATION MODEL
# =========================
class Observation(BaseModel):
    emails: List[Email]
    last_action_result: str


# =========================
# ACTION MODEL
# =========================
class Action(BaseModel):
    action_type: str  # classify / prioritize / respond
    email_id: int

    # ✅ MUST stay optional
    label: Optional[str] = None
    priority: Optional[str] = None
    response: Optional[str] = None


# =========================
# REWARD MODEL (OK TO KEEP)
# =========================
class Reward(BaseModel):
    value: float


# =========================
# STATE MODEL
# =========================
class State(BaseModel):
    emails: List[Email]
    processed_ids: List[int]
    score: float
