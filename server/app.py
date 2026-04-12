from fastapi import FastAPI
from env.environment import EmailEnv
from env.models import Action
import uvicorn

app = FastAPI()
env = EmailEnv()


# ✅ RESET (KEEP AS IS)
@app.get("/reset")
@app.post("/reset")
def reset():
    obs = env.reset()

    return {
        "observation": obs.dict(),
        "done": False,
        "tasks": [
            {
                "id": "classification",
                "grader": "env.graders:grade_easy"
            },
            {
                "id": "prioritization",
                "grader": "env.graders:grade_medium"
            },
            {
                "id": "response",
                "grader": "env.graders:grade_hard"
            }
        ]
    }


# ✅ STEP (UNCHANGED)
@app.post("/step")
def step(action: Action):
    if env._state is None:
        env.reset()

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }


# ✅ STATE (FINAL FIX — DIRECT RETURN)
@app.get("/state")
def state():
    return env.state()   # ✅ DO NOT TOUCH STRUCTURE


# ✅ REQUIRED MAIN FUNCTION
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


# ✅ REQUIRED ENTRY POINT
if __name__ == "__main__":
    main()
