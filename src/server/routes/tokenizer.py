# routes/tokenizer.py

from fastapi import APIRouter
from fastapi.responses import FileResponse

import config
from .models.tokenizer_request import TokenizeRequest
from ai_clients.tokenizer import load_tokenizer


api_router = APIRouter(prefix="/api/tokenizer")
view_router = APIRouter(prefix="/tokenizer")
tokenizer = load_tokenizer()


@view_router.get("/")
def index():
    return FileResponse(config.WEB_DIR / "tokenizer.html")


@api_router.get("/info")
def tokenizer_info():
    return {
        "name": config.TOKENIZER_NAME,
        "vocab_size": tokenizer.vocab_size,
    }


@api_router.post("/tokenize")
def tokenize(req: TokenizeRequest):
    encoding = tokenizer(
        req.text,
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    token_strings = tokenizer.convert_ids_to_tokens(tokens)

    decoded_text = tokenizer.decode(tokens)

    return {
        "token_count": len(tokens),
        "tokens": tokens,
        "token_strings": token_strings,
        "offsets": offsets,
        "decoded_text": decoded_text
    }