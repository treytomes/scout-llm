"""
Rewrites character self-references in dialogue using a local LLM.

When a speaker refers to themselves by their character name, that name is
replaced with the corpus tag name (USER_NAME or MODEL_NAME). References to
other characters are left unchanged.

Only utterances where the speaker's name actually appears (case-insensitive)
are sent to the LLM — the rest are returned unchanged.
"""

import json
import logging
import re
import requests

_OLLAMA_URL = "http://localhost:11434/api/chat"
_MODEL = "gemma4:e2b"
_BATCH_SIZE = 20
_TIMEOUT = 60

log = logging.getLogger("speaker_rewriter")


def _ollama_rewrite_batch(items: list[dict]) -> list[str]:
    """
    Send a batch of {speaker, utterance, tag_name} items to Ollama.
    Returns a list of rewritten utterances in the same order.
    Falls back to original utterances on any failure.
    """
    numbered = "\n".join(
        f'{i+1}. Speaker: {it["speaker"]} | Tag: {it["tag_name"]} | Utterance: {it["utterance"]}'
        for i, it in enumerate(items)
    )

    prompt = f"""You are editing dialogue utterances. For each numbered item:
- If the speaker refers to themselves by their own name, replace that name with their Tag name.
- If the name belongs to someone else in the conversation, leave it unchanged.
- Return ONLY a JSON array of the rewritten utterances, in the same order, no commentary.

{numbered}

Return format: ["rewritten utterance 1", "rewritten utterance 2", ...]"""

    try:
        resp = requests.post(
            _OLLAMA_URL,
            json={"model": _MODEL, "stream": False, "messages": [{"role": "user", "content": prompt}]},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()

        # Extract JSON array from response (may have markdown fences)
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            log.warning("No JSON array in Ollama response; falling back. Response: %s", content[:200])
            return [it["utterance"] for it in items]

        results = json.loads(match.group())
        if isinstance(results, list) and len(results) == len(items):
            return [str(r) for r in results]

        log.warning("Ollama returned %d results for %d items; falling back.", len(results), len(items))

    except Exception as e:
        log.error("Ollama batch failed: %s", e)

    return [it["utterance"] for it in items]


def rewrite_speaker_names(
    speakers: list[str],
    dialogue: list[str],
    tag_names: list[str],
) -> list[str]:
    """
    Given parallel lists of speaker names, utterances, and corpus tag names,
    return utterances with self-references replaced by the tag name.

    tag_names should match speakers positionally — USER_NAME for even turns,
    MODEL_NAME for odd turns (same alternating logic as the normalizer).
    """
    result = list(dialogue)

    # Short-circuit: only process utterances where speaker name appears in text
    pending = []
    for i, (speaker, utterance, tag_name) in enumerate(zip(speakers, dialogue, tag_names)):
        if speaker and speaker.lower() in utterance.lower():
            pending.append({"index": i, "speaker": speaker, "utterance": utterance, "tag_name": tag_name})

    if not pending:
        return result

    log.debug("  %d/%d utterances flagged for rewrite", len(pending), len(dialogue))

    # Process in batches
    for batch_start in range(0, len(pending), _BATCH_SIZE):
        batch = pending[batch_start: batch_start + _BATCH_SIZE]
        rewritten = _ollama_rewrite_batch(batch)
        for item, new_text in zip(batch, rewritten):
            if new_text != item["utterance"]:
                log.info(
                    "  REWRITE [%s→%s]: %r → %r",
                    item["speaker"], item["tag_name"],
                    item["utterance"][:80], new_text[:80],
                )
            result[item["index"]] = new_text

    return result