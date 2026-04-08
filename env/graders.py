def clamp_score(value: float) -> float:
    """
    Ensure score is strictly between (0,1)
    """
    if value <= 0.0:
        return 0.01
    elif value >= 1.0:
        return 0.99
    return value


# =========================
# ✅ EASY TASK (Classification)
# =========================
def grade_easy(state):
    correct = 0
    total = len(state.emails)

    for e in state.emails:
        if hasattr(e, "predicted_label") and e.predicted_label == e.true_label:
            correct += 1

    score = correct / total if total > 0 else 0.0
    return clamp_score(score)


# =========================
# ✅ MEDIUM TASK (Classification + Priority)
# =========================
def grade_medium(state):
    score = 0
    total = len(state.emails)

    for e in state.emails:
        if hasattr(e, "predicted_label") and e.predicted_label == e.true_label:
            score += 1

        if hasattr(e, "predicted_priority") and e.predicted_priority == e.priority:
            score += 1

    final_score = score / (2 * total) if total > 0 else 0.0
    return clamp_score(final_score)


# =========================
# ✅ HARD TASK (Classification + Priority + Response)
# =========================
def grade_hard(state):
    score = 0
    total = len(state.emails)

    for e in state.emails:
        if hasattr(e, "predicted_label") and e.predicted_label == e.true_label:
            score += 1

        if hasattr(e, "predicted_priority") and e.predicted_priority == e.priority:
            score += 1

        if hasattr(e, "predicted_response") and isinstance(e.predicted_response, str) and len(e.predicted_response) > 5:
            score += 1

    final_score = score / (3 * total) if total > 0 else 0.0
    return clamp_score(final_score)
