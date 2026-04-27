"""
model/lora.py — LoRA adapters for Scout's conversational memory layer.

Architecture
────────────
After Module 1 is frozen, dream cycles no longer update Module 1's weights
directly. Instead, a LoRAAdapter wraps the two attention projections in each
of Module 1's transformer blocks:

    attn.qkv  — [dim, dim*3]  — controls attention pattern + value extraction
    attn.out  — [dim, dim]    — controls the block's residual contribution

For each target Linear W, the forward pass becomes:

    y = x @ W.T  +  (x @ A.T) @ B.T  *  scale

Where A is [rank, in_features], B is [out_features, rank], and
scale = alpha / rank. W is frozen; only A and B train.

Per-conversation adapters
─────────────────────────
Each dream cycle produces one adapter. Adapters are saved independently
under data/lora_adapters/<conversation_id>.pt and accumulated until the
merge threshold is reached, at which point all accumulated adapters are
averaged and merged into Module 1's base weights. The adapter directory
is then cleared.

Manifest
────────
data/lora_adapters/manifest.json tracks adapter count and merge history
so the dream cycle can decide whether to trigger a merge.
"""

import json
import math
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# LoRA wrapper for a single nn.Linear
# ─────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a low-rank trainable delta.

    The original weight is never modified. The LoRA contribution is
    added at forward time and can be merged in-place when ready.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()

        self.base = base
        self.rank = rank
        self.scale = alpha / rank

        in_f  = base.weight.shape[1]
        out_f = base.weight.shape[0]

        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

        # Kaiming uniform on A, zeros on B → adapter starts as identity delta
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze the base weight
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return base_out + lora_out

    def merge_into_base(self):
        """Add the LoRA delta into base.weight in-place, then zero the adapter."""
        delta = (self.lora_B @ self.lora_A) * self.scale  # [out_f, in_f]
        with torch.no_grad():
            self.base.weight.add_(delta)
            self.lora_A.zero_()
            self.lora_B.zero_()

    def lora_state(self) -> dict:
        return {"lora_A": self.lora_A.data.clone(), "lora_B": self.lora_B.data.clone()}


# ─────────────────────────────────────────────────────────────
# Attach / detach LoRA to a TransformerModule
# ─────────────────────────────────────────────────────────────

_LORA_TARGETS = ("attn.qkv", "attn.out")


def attach_lora(module, rank: int, alpha: float) -> list[str]:
    """
    Wrap the qkv and out projections in every block of a TransformerModule
    with LoRALinear. The base weights remain frozen.

    Returns list of wrapped attribute paths for bookkeeping.
    """
    wrapped = []
    for i, block in enumerate(module.blocks):
        for target in _LORA_TARGETS:
            # Traverse: block.attn.qkv etc.
            parts = target.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            base_linear = getattr(parent, attr)

            if isinstance(base_linear, LoRALinear):
                continue  # already wrapped

            lora_linear = LoRALinear(base_linear, rank=rank, alpha=alpha)
            setattr(parent, attr, lora_linear)
            wrapped.append(f"blocks.{i}.{target}")

    return wrapped


def detach_lora(module):
    """
    Replace LoRALinear wrappers with their base nn.Linear (without merging).
    Used to cleanly remove an adapter without applying its delta.
    """
    for block in module.blocks:
        for target in _LORA_TARGETS:
            parts = target.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            linear = getattr(parent, attr)
            if isinstance(linear, LoRALinear):
                setattr(parent, attr, linear.base)


def merge_lora(module):
    """
    Merge all LoRALinear deltas into their base weights, then detach.
    After this call, the module behaves identically but without the wrapper.
    """
    for block in module.blocks:
        for target in _LORA_TARGETS:
            parts = target.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            linear = getattr(parent, attr)
            if isinstance(linear, LoRALinear):
                linear.merge_into_base()
                setattr(parent, attr, linear.base)


def lora_state_dict(module) -> dict:
    """Extract only the LoRA A/B parameters from a module with attached adapters."""
    state = {}
    for block_idx, block in enumerate(module.blocks):
        for target in _LORA_TARGETS:
            parts = target.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            linear = getattr(parent, attr)
            if isinstance(linear, LoRALinear):
                prefix = f"blocks.{block_idx}.{target}"
                state[f"{prefix}.lora_A"] = linear.lora_A.data.clone()
                state[f"{prefix}.lora_B"] = linear.lora_B.data.clone()
    return state


def load_lora_state_dict(module, state: dict, rank: int, alpha: float):
    """
    Attach LoRA to a module and load saved A/B weights into it.
    Safe to call on an already-wrapped module — re-uses existing wrappers.
    """
    attach_lora(module, rank=rank, alpha=alpha)

    for block_idx, block in enumerate(module.blocks):
        for target in _LORA_TARGETS:
            parts = target.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            linear = getattr(parent, attr)
            if isinstance(linear, LoRALinear):
                prefix = f"blocks.{block_idx}.{target}"
                a_key = f"{prefix}.lora_A"
                b_key = f"{prefix}.lora_B"
                if a_key in state:
                    linear.lora_A.data.copy_(state[a_key])
                if b_key in state:
                    linear.lora_B.data.copy_(state[b_key])


# ─────────────────────────────────────────────────────────────
# Adapter manifest — tracks saved adapters and merge history
# ─────────────────────────────────────────────────────────────

class LoRAManifest:
    """
    Tracks per-conversation LoRA adapters in data/lora_adapters/.

    Layout:
        data/lora_adapters/
            manifest.json          — adapter list and merge log
            <conversation_id>.pt   — saved lora_state_dict for one conversation
    """

    def __init__(self, adapters_dir: Path):
        self.dir = adapters_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / "manifest.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {"adapters": [], "merge_log": [], "total_merged": 0}

    def _save(self):
        self._path.write_text(json.dumps(self._data, indent=2))

    def record_adapter(self, conversation_id: str, adapter_path: str):
        self._data["adapters"].append({
            "conversation_id": conversation_id,
            "path": adapter_path,
        })
        self._save()

    def pending_count(self) -> int:
        return len(self._data["adapters"])

    def pending_adapters(self) -> list[dict]:
        return list(self._data["adapters"])

    def clear_pending(self, merge_note: str = ""):
        from datetime import datetime, timezone
        self._data["merge_log"].append({
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self._data["adapters"]),
            "note": merge_note,
        })
        self._data["total_merged"] += len(self._data["adapters"])
        self._data["adapters"] = []
        self._save()

    def total_merged(self) -> int:
        return self._data.get("total_merged", 0)


# ─────────────────────────────────────────────────────────────
# High-level: save adapter and trigger merge if threshold reached
# ─────────────────────────────────────────────────────────────

def save_adapter_and_maybe_merge(
    model,
    module_index: int,
    conversation_id: str,
    adapters_dir: Path,
    merge_every: int,
    rank: int,
    alpha: float,
) -> bool:
    """
    Save the current LoRA state for one dream cycle, then merge all pending
    adapters into the base weights if the merge threshold is reached.

    Returns True if a merge was performed.
    """
    module = model.expert_modules[module_index]
    manifest = LoRAManifest(adapters_dir)

    # Save this conversation's adapter
    adapter_path = adapters_dir / f"{conversation_id}.pt"
    state = lora_state_dict(module)
    torch.save({"rank": rank, "alpha": alpha, "state": state}, adapter_path)
    manifest.record_adapter(conversation_id, str(adapter_path))

    if manifest.pending_count() < merge_every:
        return False

    # ── Merge ────────────────────────────────────────────────────────────────
    # Average all pending adapter deltas, then merge into base weights.
    # Averaging prevents any single conversation from dominating.
    pending = manifest.pending_adapters()

    # Accumulate sum of all adapter states
    sum_state: Optional[dict] = None
    loaded = 0
    for entry in pending:
        p = Path(entry["path"])
        if not p.exists():
            continue
        saved = torch.load(p, map_location="cpu", weights_only=False)
        s = saved["state"]
        if sum_state is None:
            sum_state = {k: v.clone().float() for k, v in s.items()}
        else:
            for k in sum_state:
                sum_state[k].add_(s[k].float())
        loaded += 1

    if sum_state is None or loaded == 0:
        manifest.clear_pending("no valid adapters found")
        return False

    # Average
    avg_state = {k: v / loaded for k, v in sum_state.items()}

    # Load averaged adapter into the module and merge
    load_lora_state_dict(module, avg_state, rank=rank, alpha=alpha)
    merge_lora(module)

    # Clean up adapter files
    for entry in pending:
        try:
            Path(entry["path"]).unlink(missing_ok=True)
        except Exception:
            pass

    manifest.clear_pending(f"merged {loaded} adapters (averaged)")

    return True
