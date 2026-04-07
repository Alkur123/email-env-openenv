from pydantic import BaseModel
from typing import List, Optional

class Email(BaseModel):
    id: int
    subject: str
    body: str
    true_label: str  # spam / urgent / normal
    priority: str    # high / medium / low

class Observation(BaseModel):
    emails: List[Email]
    last_action_result: str

class Action(BaseModel):
    action_type: str  # classify / prioritize / respond
    email_id: int
    label: Optional[str] = None
    priority: Optional[str] = None
    response: Optional[str] = None

class Reward(BaseModel):
    value: float

class State(BaseModel):
    emails: List[Email]
    processed_ids: List[int]
    score: float