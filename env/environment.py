import json
from typing import Tuple
from .models import Observation, Action, State, Email

# ✅ ADD THIS
from .graders import grade_easy, grade_medium, grade_hard


class EmailEnv:

    def __init__(self):
        self._state = None
        self.done = False

    def reset(self) -> Observation:
        with open("data/emails.json", "r") as f:
            raw_emails = json.load(f)

        emails = [Email(**e) for e in raw_emails]

        self._state = State(
            emails=emails,
            processed_ids=[],
            score=0.0
        )

        # ✅ REQUIRED: 3 TASKS DECLARED
        self.tasks = [
            {"task_id": "task_classification", "type": "classification"},
            {"task_id": "task_prioritization", "type": "prioritization"},
            {"task_id": "task_response", "type": "response"}
        ]

        # ✅ ADD THIS (CONNECT GRADERS)
        self.graders = {
            "easy": grade_easy,
            "medium": grade_medium,
            "hard": grade_hard
        }

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
                0.0,
                False,
                {"error": "invalid_id"}
            )

        # =========================
        # TASK 1: CLASSIFICATION
        # =========================
        if action.action_type == "classify":
            # ✅ STORE PREDICTION
            email.predicted_label = action.label

            if action.label == email.true_label:
                reward = 1.0
                info = {"task": "classification", "result": "correct"}
            else:
                reward = 0.0
                info = {"task": "classification", "result": "wrong"}

        # =========================
        # TASK 2: PRIORITIZATION
        # =========================
        elif action.action_type == "prioritize":
            # ✅ STORE PREDICTION
            email.predicted_priority = action.priority

            if action.priority == email.priority:
                reward = 1.0
                info = {"task": "prioritization", "result": "correct"}
            else:
                reward = 0.0
                info = {"task": "prioritization", "result": "wrong"}

        # =========================
        # TASK 3: RESPONSE
        # =========================
        elif action.action_type == "respond":
            # ✅ STORE PREDICTION
            email.predicted_response = action.response

            if isinstance(action.response, str) and len(action.response) > 5:
                reward = 1.0
                info = {"task": "response", "result": "valid"}
            else:
                reward = 0.0
                info = {"task": "response", "result": "invalid"}

        else:
            reward = 0.0
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

    def state(self):
        return self._state
