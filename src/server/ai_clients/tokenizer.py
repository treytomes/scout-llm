# ai_clients/tokenizer.py

import logging
from transformers import (
    AutoTokenizer,
    TokenizersBackend, 
    SentencePieceBackend,
)

import config


logger = logging.getLogger(config.LOGGER_NAME)
_tokenizer: TokenizersBackend | SentencePieceBackend | None = None


def load_tokenizer() -> TokenizersBackend | SentencePieceBackend:
    global _tokenizer
    if _tokenizer is None:
        logger.info("Loading tokenizer: %s", config.TOKENIZER_NAME)
        try:
            # Try loading strictly from local cache
            _tokenizer = AutoTokenizer.from_pretrained(
                config.TOKENIZER_NAME,
                cache_dir=config.HUGGINGFACE_CACHE_PATH,
                local_files_only=True,
            )
        except Exception:
            # If not cached yet, download and store in cache
            _tokenizer = AutoTokenizer.from_pretrained(
                config.TOKENIZER_NAME,
                cache_dir=config.HUGGINGFACE_CACHE_PATH,
            )
    return _tokenizer