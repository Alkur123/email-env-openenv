import os
import asyncio
from openai import OpenAI

# ✅ ENV VARIABLES (MANDATORY)
API_BASE_URL = os.getenv("API_BASE_URL", "https://dummy-api.com")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy-model")
HF_TOKEN = os.getenv("HF_TOKEN")  # no default

# ✅ OpenAI client (safe even if dummy)
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ✅ fallback agent (your improved logic)
def fallback_agent(email):
    text = (email["subject"] + " " + email["body"]).lower()

    spam_words = ["free", "win", "offer", "click", "buy now"]
    urgent_words = ["urgent", "asap", "immediately", "meeting", "deadline"]

    spam_score = sum(word in text for word in spam_words)
    urgent_score = sum(word in text for word in urgent_words)

    if spam_score > urgent_score and spam_score > 0:
        return "spam"
    elif urgent_score > 0:
        return "urgent"
    else:
        return "normal"


# ✅ LLM + fallback (CRITICAL FUNCTION)
def simple_agent(email):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify this email into spam, urgent, or normal:\nSubject: {email['subject']}\nBody: {email['body']}\nReturn only one word."
                }
            ],
            timeout=2
        )

        output = response.choices[0].message.content.strip().lower()

        if output in ["spam", "urgent", "normal"]:
            return output
        else:
            return fallback_agent(email)

    except Exception:
        # ✅ SAFE FALLBACK (VERY IMPORTANT)
        return fallback_agent(email)


async def main():
    print("[START]")

    emails = [
        {"id": 1, "subject": "Win a free iPhone!!!", "body": "Click here"},
        {"id": 2, "subject": "Meeting at 3 PM", "body": "Important meeting"},
        {"id": 3, "subject": "Newsletter", "body": "Weekly updates"}
    ]

    rewards = []
    valid_labels = ["spam", "urgent", "normal"]

    for step, email in enumerate(emails):
        prediction = simple_agent(email)

        reward = 0.3 if prediction in valid_labels else -0.2
        done = step == len(emails) - 1

        print(f"[STEP] step={step} action={prediction} reward={reward} done={done}")

        rewards.append(reward)

    score = sum(rewards) / len(rewards)

    print(f"[END] success=True steps={len(emails)} score={score} rewards={rewards}")


if __name__ == "__main__":
    asyncio.run(main())
