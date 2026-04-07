import json
from typing import Tuple
from .models import Observation, Action, State, Email


class EmailEnv:

    def __init__(self):
        self._state = None
        self.done = False

    def reset(self) -> Observation:
        with open("data/emails.json", "r") as f:
            raw_emails = json.load(f)

        # ✅ Convert dicts → Email objects
        emails = [Email(**e) for e in raw_emails]

        self._state = State(
            emails=emails,
            processed_ids=[],
            score=0.0
        )
        self.done = False

        return Observation(
            emails=self._state.emails,
            last_action_result="Environment reset"
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:

        reward = 0.0
        info = {}

        email = next(
            (e for e in self._state.emails if e.id == action.email_id),
            None
        )

        if not email:
            return (
                Observation(
                    emails=self._state.emails,
                    last_action_result="Invalid ID"
                ),
                -0.2,
                False,
                {}
            )

        # =========================
        # ✅ TASK 1: CLASSIFICATION GRADER
        # =========================
        if action.action_type == "classify":
            if action.label == email.true_label:
                reward = 0.3
                info = {"task": "classification", "result": "correct"}
            else:
                reward = -0.2
                info = {"task": "classification", "result": "wrong"}

        # =========================
        # ✅ TASK 2: PRIORITY GRADER
        # =========================
        elif action.action_type == "prioritize":
            if action.priority == email.priority:
                reward = 0.2
                info = {"task": "prioritization", "result": "correct"}
            else:
                reward = -0.1
                info = {"task": "prioritization", "result": "wrong"}

        # =========================
        # ✅ TASK 3: RESPONSE GRADER
        # =========================
        elif action.action_type == "respond":
            if (
                isinstance(action.response, str)
                and len(action.response) > 5
            ):
                reward = 0.1
                info = {"task": "response", "result": "valid"}
            else:
                reward = -0.05
                info = {"task": "response", "result": "invalid"}

        else:
            reward = -0.1
            info = {"task": "invalid", "result": "error"}

        # =========================
        # STATE UPDATE
        # =========================
        self._state.score += reward

        if email.id not in self._state.processed_ids:
            self._state.processed_ids.append(email.id)

        if len(self._state.processed_ids) >= len(self._state.emails):
            self.done = True

        return (
            Observation(
                emails=self._state.emails,
                last_action_result="Action processed"
            ),
            reward,
            self.done,
            info
        )

    # ✅ STATE FUNCTION (FIXED)
    def state(self):
        return self._state
