from fastapi import FastAPI
from env.environment import EmailEnv
from env.models import Action
import uvicorn

app = FastAPI()
env = EmailEnv()


# ✅ FIX: support BOTH GET and POST
@app.get("/reset")
@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.dict(), "done": False}


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


@app.get("/state")
def state():
    return env.state().dict()


# ✅ REQUIRED MAIN FUNCTION
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


# ✅ REQUIRED ENTRY POINT
if __name__ == "__main__":
    main()
