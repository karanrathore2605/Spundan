import ctypes
import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_API_URL,
    EMBEDDING_FALLBACK_MODEL,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_embedding_model: Optional[SentenceTransformer] = None
_active_model_name: Optional[str] = None
_is_fallback_active: bool = False
_fallback_reason: Optional[str] = None
_model_dimension: Optional[int] = None


def _get_optimal_device() -> str:
    """Detect whether CUDA is available or default to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _is_qwen_embedding_model(model_name: str) -> bool:
    """Check if model belongs to Qwen/GTE embedding family."""
    lower = model_name.lower()
    return "qwen" in lower or "gte" in lower


def _get_system_ram_gb() -> float:
    """Get system RAM in gigabytes using ctypes or fallback."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        return 16.0


def _is_model_cached(model_name: str) -> bool:
    """Check if the Hugging Face model config is already cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache
        res = try_to_load_from_cache(model_name, "config.json")
        return res is not None and not isinstance(res, type(None))
    except Exception:
        return False


def _call_remote_embedding_api(texts: List[str]) -> Optional[np.ndarray]:
    """Call an external OpenAI-compatible or Ollama embedding API if configured."""
    if not EMBEDDING_API_URL:
        return None

    import httpx

    headers = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(EMBEDDING_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            if "data" in data:
                # OpenAI-compatible response format
                embeddings = [item["embedding"] for item in data["data"]]
                arr = np.asarray(embeddings, dtype="float32")
                # Normalize
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return arr / norms
    except Exception as exc:
        logger.warning("Remote embedding API call failed: %s. Falling back to local loader.", exc)

    return None


def get_embedding_model() -> SentenceTransformer:
    """
    Load or retrieve the cached SentenceTransformer model.
    Targets Qwen 3-Embedding - 8B (Qwen/Qwen3-Embedding-8B) with automatic
    hardware detection, remote code trust, and graceful fallback if memory is constrained.
    """
    global _embedding_model, _active_model_name, _is_fallback_active, _fallback_reason, _model_dimension

    if _embedding_model is not None:
        return _embedding_model

    device = _get_optimal_device()
    target_model = EMBEDDING_MODEL
    force_download = os.getenv("FORCE_DOWNLOAD_8B", "false").lower() in ("true", "1")
    system_ram_gb = _get_system_ram_gb()
    is_large_8b = any(k in target_model.lower() for k in ["8b", "7b", "14b"])
    cached = _is_model_cached(target_model)

    # Prevent out-of-memory crash if local system has < 16GB RAM and model is not yet cached
    if is_large_8b and not cached and not force_download and system_ram_gb < 14.0:
        reason = (
            f"Primary model '{target_model}' is an 8B parameter model (~16 GB weights) not yet cached locally, "
            f"and host memory ({system_ram_gb:.1f} GB) is below the recommended 16 GB threshold. "
            f"Using '{EMBEDDING_FALLBACK_MODEL}' to avoid memory exhaustion. "
            "To connect Qwen 3-Embedding - 8B, provide an external endpoint via EMBEDDING_API_URL (e.g. Ollama or vLLM) "
            "or set FORCE_DOWNLOAD_8B=true."
        )
        logger.warning(reason)
        _is_fallback_active = True
        _fallback_reason = reason
        _embedding_model = SentenceTransformer(EMBEDDING_FALLBACK_MODEL, device=device)
        _active_model_name = EMBEDDING_FALLBACK_MODEL
        try:
            sample_emb = _embedding_model.encode(["test"], convert_to_numpy=True)
            _model_dimension = int(sample_emb.shape[1])
        except Exception:
            _model_dimension = 384
        return _embedding_model

    logger.info("Loading primary embedding model: %s (device: %s)", target_model, device)

    # Configuration kwargs for Qwen / large transformer models
    model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    try:
        _embedding_model = SentenceTransformer(
            target_model,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        _active_model_name = target_model
        _is_fallback_active = False
        _fallback_reason = None
        logger.info("Successfully loaded embedding model: %s", target_model)
    except Exception as exc:
        reason = f"Could not load primary embedding model '{target_model}': {exc}"
        logger.warning("%s. Falling back to '%s'.", reason, EMBEDDING_FALLBACK_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_FALLBACK_MODEL, device=device)
        _active_model_name = EMBEDDING_FALLBACK_MODEL
        _is_fallback_active = True
        _fallback_reason = reason

    # Determine embedding dimension
    try:
        sample_emb = _embedding_model.encode(["test"], convert_to_numpy=True)
        _model_dimension = int(sample_emb.shape[1])
    except Exception:
        _model_dimension = 384

    return _embedding_model


def get_embedding_info() -> Dict[str, Any]:
    """Retrieve diagnostic information regarding the active embedding model."""
    global _active_model_name, _is_fallback_active, _fallback_reason, _model_dimension
    return {
        "requested_model": EMBEDDING_MODEL,
        "active_model": _active_model_name or EMBEDDING_MODEL,
        "is_fallback": _is_fallback_active,
        "fallback_model": EMBEDDING_FALLBACK_MODEL,
        "fallback_reason": _fallback_reason,
        "device": _get_optimal_device(),
        "dimension": _model_dimension or (4096 if "8b" in EMBEDDING_MODEL.lower() else 384),
        "remote_api_configured": bool(EMBEDDING_API_URL),
    }


def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    """
    Generate normalized float32 embeddings for a batch of texts.
    Supports Qwen3-Embedding query prompting when is_query=True.
    """
    if not texts:
        dim = _model_dimension or (4096 if "8b" in EMBEDDING_MODEL.lower() else 384)
        return np.empty((0, dim), dtype="float32")

    # Check for external API first
    remote_embeddings = _call_remote_embedding_api(texts)
    if remote_embeddings is not None:
        return remote_embeddings

    model = get_embedding_model()
    is_qwen = _is_qwen_embedding_model(_active_model_name or EMBEDDING_MODEL)

    # Qwen embedding models utilize prompt_name='query' for search queries
    if is_query and is_qwen:
        try:
            embeddings = model.encode(
                texts,
                prompt_name="query",
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype="float32")
        except Exception:
            pass

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Generate normalized float32 embedding for a single query."""
    return embed_texts([query], is_query=True)
