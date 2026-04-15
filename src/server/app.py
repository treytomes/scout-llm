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


# @app.get("/test")
# def test():
#     from corpus.dataset_repository import DatasetRepository
#     repo = DatasetRepository()

#     data = repo.list_datasets()
#     ds_name = "WildChat-50M"
#     data = repo.get_dataset(ds_name)
#     if not data.is_normalized():
#         repo.normalize_dataset(ds_name)
#         data = repo.get_dataset(ds_name)
#     data = data.get_rows(False, "train", 20, 0)
#     return {
#         "status": "ok",
#         "data": data,
#     }


@app.get("/api/status")
def status():
    return {
        "training": False,
        "model": "scout-dev",
        "uptime": "unknown"
    }
