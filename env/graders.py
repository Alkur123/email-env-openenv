def grade_easy(state):
    correct = 0
    total = len(state.emails)

    for e in state.emails:
        if hasattr(e, "predicted_label") and e.predicted_label == e.true_label:
            correct += 1

    return correct / total if total > 0 else 0.0


def grade_medium(state):
    score = 0
    total = len(state.emails)

    for e in state.emails:
        if hasattr(e, "predicted_label") and e.predicted_label == e.true_label:
            score += 1
        if hasattr(e, "predicted_priority") and e.predicted_priority == e.priority:
            score += 1

    return score / (2 * total) if total > 0 else 0.0


def grade_hard(state):
    score = 0
    total = len(state.emails)

    for e in state.emails:
        if hasattr(e, "predicted_label") and e.predicted_label == e.true_label:
            score += 1
        if hasattr(e, "predicted_priority") and e.predicted_priority == e.priority:
            score += 1
        if hasattr(e, "predicted_response") and len(e.predicted_response) > 5:
            score += 1

    return score / (3 * total) if total > 0 else 0.0
