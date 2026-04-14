from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from server.routes import datasets


app = FastAPI()

WEB_DIR = Path(__file__).parent.parent / "web"

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.include_router(datasets.router)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


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
