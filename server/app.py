from fastapi import FastAPI
from env.environment import EmailEnv
from env.models import Action
import uvicorn

app = FastAPI()
env = EmailEnv()


# ✅ RESET (WITH TASKS)
@app.get("/reset")
@app.post("/reset")
def reset():
    obs = env.reset()

    return {
        "observation": obs.dict(),
        "done": False,

        # ✅ expose tasks with graders
        "tasks": [
            {
                "id": "classification",
                "type": "classification",
                "grader": "env.graders:grade_easy"
            },
            {
                "id": "prioritization",
                "type": "prioritization",
                "grader": "env.graders:grade_medium"
            },
            {
                "id": "response",
                "type": "response",
                "grader": "env.graders:grade_hard"
            }
        ]
    }


# ✅ STEP
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


# ✅ STATE (🔥 FIXED — NO MANUAL GRADER CALLS)
@app.get("/state")
def state():
    return env.state()   # ✅ JUST RETURN THIS


# ✅ REQUIRED MAIN FUNCTION
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


# ✅ REQUIRED ENTRY POINT
if __name__ == "__main__":
    main()
