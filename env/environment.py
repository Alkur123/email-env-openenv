import json
from typing import Tuple
from .models import Observation, Action, State, Email


class EmailEnv:

    def __init__(self):
        self._state = None   # renamed to avoid conflict with state() method
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

        # ✅ Safe lookup using object attributes
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

        # ✅ Classification reward
        if action.action_type == "classify":
            if action.label == email.true_label:
                reward += 0.3
            else:
                reward -= 0.2

        # ✅ Priority reward
        elif action.action_type == "prioritize":
            if action.priority == email.priority:
                reward += 0.2
            else:
                reward -= 0.1

        # ✅ Response reward
        elif action.action_type == "respond":
            if action.response:
                reward += 0.1

        else:
            reward -= 0.1  # penalty for invalid action

        # ✅ Update state safely
        self._state.score += reward

        if email.id not in self._state.processed_ids:
            self._state.processed_ids.append(email.id)

        # ✅ Done condition
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