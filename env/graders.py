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
            score += 0.5
        if hasattr(e, "predicted_priority") and e.predicted_priority == e.priority:
            score += 0.5

    return score / total


def grade_hard(state):
    score = 0
    total = len(state.emails)

    for e in state.emails:
        if hasattr(e, "predicted_label") and e.predicted_label == e.true_label:
            score += 0.4
        if hasattr(e, "predicted_priority") and e.predicted_priority == e.priority:
            score += 0.3
        if hasattr(e, "response"):
            score += 0.3

    return score / total