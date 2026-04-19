from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

import config as config
from .models.dataset_preview import DatasetPreview
from .models.training_plan_request import TrainingPlanRequest
from ai_clients.tokenizer import load_tokenizer
from corpus.dataset_repository import DatasetRepository
from workers.dataset_download_job_manager import DatasetDownloadJobManager


api_router = APIRouter(prefix="/api/datasets")
view_router = APIRouter(prefix="/datasets")
job_manager = DatasetDownloadJobManager()
repo = DatasetRepository();


@view_router.get("/preview")
def index():
    return FileResponse(config.WEB_DIR / "preview.html")


@api_router.get("")
def list_datasets():
    return repo.list_datasets()


@api_router.get("/{name}/status")
def dataset_status(name: str):
    return job_manager.job_status(name)


@api_router.post("/{name}/download")
def start_download(name: str):
    job_manager.start_download(name)
    return {"status": "started"}


@api_router.get("/{name}/progress")
def dataset_progress(name: str):
    return job_manager.job_status(name)


@api_router.delete("/{name}")
def delete_dataset(name: str):
    job_manager.delete(name)
    return {"status": "ok"}


@api_router.post("/{name}/normalize")
def normalize_dataset(name: str):
    import threading
    def _run():
        try:
            repo.normalize_dataset(name)
        except Exception as e:
            print(f"Normalization error for {name}: {e}")
    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@api_router.get("/{name}/preview")
def preview_dataset(
    name: str,
    limit: int = Query(20, ge=1, le=200),
    page: int = Query(0, ge=0),
    is_raw: bool = Query(False),
    split_name: str = Query("train"),
):
    try:
        model = repo.get_dataset(name)
        if not model.exists():
            raise Exception("Dataset not downloaded.")
        total_rows = model.get_total_rows(is_raw, split_name)
        rows = model.get_rows(is_raw, split_name, limit, page)

        return DatasetPreview(name, split_name, page, limit, total_rows, rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/{name}/training_plan")
def training_plan(name: str, req: TrainingPlanRequest):
    try:
        model = repo.get_dataset(name)
        if not model.exists():
            raise Exception("Dataset not downloaded.")

        tokenizer = load_tokenizer()
        dataset = model.get_normalized(req.split_name)

        BATCH_SIZE = 100_000  # number of chunks per tokenizer call

        batch = []
        total_tokens = 0
        n = 0
        total = len(dataset)

        print("Beginning token counting.")

        for row in dataset:
            chunk = row.get("chunk", "")
            if not chunk:
                continue

            batch.append(chunk)

            if len(batch) > BATCH_SIZE:
                enc = tokenizer(
                    batch,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )
                total_tokens += sum(len(ids) for ids in enc["input_ids"])
                batch = []
                print(f"Processed {n}/{total}")

            n += 1

        if batch:
            enc = tokenizer(
                batch,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            total_tokens += sum(len(ids) for ids in enc["input_ids"])

        print("Done.")

        vocab_size = tokenizer.vocab_size
        seq_len = req.block_size

        samples = max(0, total_tokens // seq_len)
        tokens_per_step = seq_len * req.batch_size
        steps_per_epoch = total_tokens / tokens_per_step

        return {
            "dataset": name,
            "split": req.split_name,
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "sequence_length": seq_len,
            "batch_size": req.batch_size,
            "tokens_per_step": tokens_per_step,
            "training_samples": samples,
            "steps_per_epoch": steps_per_epoch,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))