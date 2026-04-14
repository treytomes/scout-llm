from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config as config
from routes import datasets


app = FastAPI()
app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")
app.include_router(datasets.router)


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html")


@app.get("/api/hello")
def hello():
    return {
        "message": "Hello from Scout server",
        "status": "ok"
    }


@app.get("/api/status")
def status():
    return {
        "training": False,
        "model": "scout-dev",
        "uptime": "unknown"
    }
