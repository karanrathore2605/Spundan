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
    HF_TOKEN,
)

logger = logging.getLogger(__name__)

_embedding_model: Optional[SentenceTransformer] = None
_active_model_name: Optional[str] = None
_is_fallback_active: bool = False
_fallback_reason: Optional[str] = None
_model_dimension: Optional[int] = None


def get_active_model_name() -> str:
    """Return currently active embedding model name."""
    return _active_model_name or EMBEDDING_MODEL


def get_model_dimension() -> int:
    """Return embedding vector dimension of currently active model."""
    if _model_dimension is not None:
        return _model_dimension
    return 384 if _is_fallback_active else (4096 if "8b" in EMBEDDING_MODEL.lower() else 384)


def is_fallback_active() -> bool:
    """Return True if fallback model is active."""
    return _is_fallback_active


def get_fallback_reason() -> Optional[str]:
    """Return reason for fallback activation, if any."""
    return _fallback_reason


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
    """
    Accurately detect available system/container RAM in GB across Windows, Linux, and Cloud containers.
    """
    # 1. Check Linux cgroups v2 / v1 (Streamlit Cloud, Docker, Kubernetes)
    for cgroup_path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            if os.path.exists(cgroup_path):
                with open(cgroup_path, "r") as f:
                    val = f.read().strip()
                    if val and val != "max":
                        limit_bytes = int(val)
                        if 0 < limit_bytes < (1024 ** 5):  # Valid finite limit
                            return limit_bytes / (1024 ** 3)
        except Exception:
            pass

    # 2. Check Linux /proc/meminfo
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
    except Exception:
        pass

    # 3. Check os.sysconf (POSIX)
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024 ** 3)
    except Exception:
        pass

    # 4. Check Windows ctypes (GlobalMemoryStatusEx)
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
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass

    # 5. Check psutil if available
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    # Safe conservative default for cloud containers if undetermined
    return 2.0


def _is_model_cached(model_name: str) -> bool:
    """Check if the Hugging Face model config is already cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache
        res = try_to_load_from_cache(model_name, "config.json", token=HF_TOKEN or None)
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
                embeddings = [item["embedding"] for item in data["data"]]
                arr = np.asarray(embeddings, dtype="float32")
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return arr / norms
    except Exception as exc:
        logger.warning("Remote embedding API call failed: %s. Falling back to local loader.", exc)

    return None


def get_embedding_model() -> SentenceTransformer:
    """
    Load or retrieve the cached SentenceTransformer model.
    Primary: Qwen/Qwen3-Embedding-8B
    Fallback: all-MiniLM-L6-v2
    Gracefully detects memory constraints, download timeouts, and HF rate limits.
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

    # Set HF_TOKEN in environment if provided so HuggingFace hub authenticates
    if HF_TOKEN and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = HF_TOKEN

    # [6] Embedding model loading log
    logger.info("[6] Embedding model loading: Target='%s' (device='%s', RAM=%.1f GB)", target_model, device, system_ram_gb)

    # Prevent out-of-memory crash if local/cloud container has < 14 GB RAM and model is not cached
    if is_large_8b and not cached and not force_download and system_ram_gb < 14.0:
        reason = (
            f"Primary model '{target_model}' (~16 GB weights) requires at least 16 GB RAM/GPU memory. "
            f"Host environment provides {system_ram_gb:.1f} GB RAM (device: {device}). "
            f"To prevent container termination / OOM kill, automatically activating fallback '{EMBEDDING_FALLBACK_MODEL}'. "
            "To connect Qwen 3-Embedding-8B, provide an external endpoint via EMBEDDING_API_URL or set FORCE_DOWNLOAD_8B=true."
        )
        logger.warning("%s", reason)
        _is_fallback_active = True
        _fallback_reason = reason
        try:
            _embedding_model = SentenceTransformer(
                EMBEDDING_FALLBACK_MODEL,
                device=device,
                token=HF_TOKEN or None,
            )
            _active_model_name = EMBEDDING_FALLBACK_MODEL
            sample_emb = _embedding_model.encode(["test"], convert_to_numpy=True)
            _model_dimension = int(sample_emb.shape[1])
            logger.info("[7] Embedding model loaded: '%s' (fallback=True, dim=%d)", _active_model_name, _model_dimension)
            return _embedding_model
        except Exception as fb_exc:
            logger.error("Failed to load fallback embedding model: %s", fb_exc)
            raise

    # Attempt to load primary model
    model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    try:
        _embedding_model = SentenceTransformer(
            target_model,
            device=device,
            trust_remote_code=True,
            token=HF_TOKEN or None,
            model_kwargs=model_kwargs,
        )
        _active_model_name = target_model
        _is_fallback_active = False
        _fallback_reason = None
        sample_emb = _embedding_model.encode(["test"], convert_to_numpy=True)
        _model_dimension = int(sample_emb.shape[1])
        logger.info("[7] Embedding model loaded: '%s' (fallback=False, dim=%d)", _active_model_name, _model_dimension)
    except Exception as exc:
        reason = f"Primary model '{target_model}' initialization failed: {exc}"
        logger.warning("[6] Embedding model loading: %s. Attempting fallback embedding model '%s'...", reason, EMBEDDING_FALLBACK_MODEL)
        try:
            _embedding_model = SentenceTransformer(
                EMBEDDING_FALLBACK_MODEL,
                device=device,
                token=HF_TOKEN or None,
            )
            _active_model_name = EMBEDDING_FALLBACK_MODEL
            _is_fallback_active = True
            _fallback_reason = reason
            sample_emb = _embedding_model.encode(["test"], convert_to_numpy=True)
            _model_dimension = int(sample_emb.shape[1])
            logger.info("[7] Embedding model loaded: '%s' (fallback=True, dim=%d)", _active_model_name, _model_dimension)
        except Exception as fb_exc:
            logger.error("Both primary and fallback embedding models failed to load: %s", fb_exc)
            raise

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
