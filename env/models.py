from pydantic import BaseModel
from typing import List, Optional


class Email(BaseModel):
    id: int
    subject: str
    body: str
    true_label: str
    priority: str
    expected_response: str

    # ✅ ADD THESE FIELDS (CRITICAL)
    predicted_label: Optional[str] = None
    predicted_priority: Optional[str] = None
    predicted_response: Optional[str] = None


class State(BaseModel):
    emails: List[Email]
    processed_ids: List[int]
    score: float


class Observation(BaseModel):
    emails: List[Email]
    last_action_result: str


class Action(BaseModel):
    action_type: str
    email_id: int
    label: Optional[str] = None
    priority: Optional[str] = None
    response: Optional[str] = None
