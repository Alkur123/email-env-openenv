def safe_div(num, denom):
    if denom == 0:
        return 0.5
    return float(num) / float(denom)


def clamp_score(x):
    try:
        x = float(x)
    except Exception:
        return 0.01

    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99
    return x


def grade_easy(state):
    emails = getattr(state, "emails", [])
    total = len(emails)
    correct = 0

    for e in emails:
        if getattr(e, "predicted_label", None) == getattr(e, "true_label", None):
            correct += 1

    score = safe_div(correct, total)
    return clamp_score(score)


def grade_medium(state):
    emails = getattr(state, "emails", [])
    total = len(emails)
    score = 0

    for e in emails:
        if getattr(e, "predicted_label", None) == getattr(e, "true_label", None):
            score += 1
        if getattr(e, "predicted_priority", None) == getattr(e, "priority", None):
            score += 1

    score = safe_div(score, 2 * total)
    return clamp_score(score)


def grade_hard(state):
    emails = getattr(state, "emails", [])
    total = len(emails)
    score = 0

    for e in emails:
        if getattr(e, "predicted_label", None) == getattr(e, "true_label", None):
            score += 1
        if getattr(e, "predicted_priority", None) == getattr(e, "priority", None):
            score += 1

        response = getattr(e, "predicted_response", None)
        if isinstance(response, str) and len(response) > 5:
            score += 1

    score = safe_div(score, 3 * total)
    return clamp_score(score)
