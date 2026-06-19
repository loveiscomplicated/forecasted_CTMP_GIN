import json
import os
import random
import time
import hashlib
import subprocess
import shutil
from datetime import datetime
from pathlib import Path


DATASET_ID = "tedsd_2022"
REMOTE_BASE = "gdrive:CTMP_GIN_mi_service"

LOCAL_CACHE_DIR = Path("/workspace/CTMP_GIN/cache/mi_dict")
# LOCAL_CACHE_DIR = Path(".") # for debugging
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Rate-limit constants
_RATE_LIMIT_MARKERS = ("rateLimitExceeded", "RATE_LIMIT_EXCEEDED", "403", "429")
_RCLONE_MAX_RETRIES = 8
_RCLONE_BACKOFF_BASE = 5.0   # seconds
_RCLONE_BACKOFF_CAP  = 120.0 # seconds

# Ensure remote dirs are created only once per worker process
_remote_dirs_ensured = False

# Startup jitter: applied once per process to spread initial API burst across workers
_startup_jitter_applied = False


def _apply_startup_jitter() -> None:
    """
    Sleep once per process to stagger parallel workers' first API calls.
    Uses CUDA_VISIBLE_DEVICES to derive worker index (GPU 0=0s, GPU 1=8s, …).
    """
    global _startup_jitter_applied
    if _startup_jitter_applied:
        return
    _startup_jitter_applied = True

    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    try:
        gpu_id = int(str(gpu_env).split(",")[0])
    except (ValueError, IndexError):
        gpu_id = 0

    # 8 seconds per GPU + up to 4s random jitter → GPU9 waits ~76s max
    jitter = gpu_id * 8 + random.uniform(0, 4)
    if jitter > 0:
        print(f"[request_mi] startup jitter {jitter:.1f}s (GPU={gpu_id})")
        time.sleep(jitter)


def _is_rate_limit_error(err: subprocess.CalledProcessError) -> bool:
    combined = (err.stdout or "") + (err.stderr or "")
    return any(marker in combined for marker in _RATE_LIMIT_MARKERS)


def _run(cmd: list[str], *, allow_rate_limit_retry: bool = True) -> str:
    """
    Run command and return stdout.
    Retries on Google Drive rate-limit errors with exponential backoff + jitter.
    Raises RuntimeError with stdout/stderr on non-retryable failure.
    """
    for attempt in range(1, _RCLONE_MAX_RETRIES + 1):
        try:
            p = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return p.stdout
        except FileNotFoundError as e:
            raise RuntimeError(
                f"[CMD NOT FOUND]\n"
                f"cmd: {' '.join(cmd)}\n"
                f"Is '{cmd[0]}' installed and in PATH?\n"
                f"error: {e}\n"
            ) from e
        except subprocess.CalledProcessError as e:
            if allow_rate_limit_retry and _is_rate_limit_error(e) and attempt < _RCLONE_MAX_RETRIES:
                wait = min(_RCLONE_BACKOFF_BASE * (2 ** (attempt - 1)), _RCLONE_BACKOFF_CAP)
                wait += random.uniform(0, wait * 0.3)  # ±30% jitter
                print(
                    f"[request_mi] rate limit on attempt {attempt}/{_RCLONE_MAX_RETRIES}, "
                    f"retrying in {wait:.1f}s  cmd={' '.join(cmd)}"
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"[CMD FAILED]\n"
                f"cmd: {' '.join(cmd)}\n"
                f"returncode: {e.returncode}\n"
                f"stdout:\n{e.stdout}\n"
                f"stderr:\n{e.stderr}\n"
            ) from e


def _artifact_key(mode: str, fold: int | None, seed: int, n_neighbors: int, remove_los: bool=True) -> str:
    if mode == "cv":
        if fold is None:
            raise ValueError("mode=cv requires fold")
        return f"mi__ds={DATASET_ID}__mode=cv__fold={fold}__seed={seed}__n_neighbors={n_neighbors}__remove_los={remove_los}"
    if mode == "single":
        return f"mi__ds={DATASET_ID}__mode=single__seed={seed}__n_neighbors={n_neighbors}__remove_los={remove_los}"
    raise ValueError("mode must be 'cv' or 'single'")


def _request_id_from_artifact(artifact_key: str) -> str:
    # short stable hash + timestamp (avoid overly long names, avoid collisions)
    h = hashlib.sha1(artifact_key.encode("utf-8")).hexdigest()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}__{h}"


def _ensure_remote_dirs() -> None:
    """Create remote dirs once per worker process; no-op on subsequent calls."""
    global _remote_dirs_ensured
    if _remote_dirs_ensured:
        return
    _run(["rclone", "mkdir", f"{REMOTE_BASE}/requests"])
    _run(["rclone", "mkdir", f"{REMOTE_BASE}/responses"])
    _remote_dirs_ensured = True


def _acquire_lock(lock_dir: Path, local_pkl: Path, use_cache: bool = True) -> bool:
    """
    Acquire local lock directory. Returns True if acquired.
    If lock exists, waits until either pkl exists (then returns False) or lock becomes available.
    Detects and clears stale locks left by crashed processes.
    """
    while True:
        if use_cache and local_pkl.exists():
            return False
        try:
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()))
            return True
        except FileExistsError:
            # Check for stale lock (owning process no longer alive)
            pid_file = lock_dir / "pid"
            try:
                pid = int(pid_file.read_text())
                os.kill(pid, 0)  # signal 0: existence check only
            except (FileNotFoundError, ValueError):
                pass  # cannot determine, wait
            except ProcessLookupError:
                # Stale lock — owning process is gone
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            except PermissionError:
                pass  # process exists but we can't signal it

            if use_cache and local_pkl.exists():
                return False
            time.sleep(1)


def _release_lock(lock_dir: Path) -> None:
    # robust cleanup
    shutil.rmtree(lock_dir, ignore_errors=True)

def request_mi(
    *,
    mode: str,  # "single" or "cv"
    fold: int | None,
    seed: int,
    cfg: dict,
    n_neighbors: int,
    poll_interval_sec: int = 30,
    timeout_sec: int | None = None,
    serialize_cfg_default_str: bool = True,
    verbose_poll: bool = False,
    use_cache: bool = True,
) -> str:
    """
    Request mi_dict from worker via rclone remote folder, blocking until ready.
    Returns local path to the cached mi_dict pickle.

    Remote protocol:
      - Upload:  {REMOTE_BASE}/requests/{request_id}.json
      - Worker writes: {REMOTE_BASE}/responses/{request_id}.pkl
      - Download to: {LOCAL_CACHE_DIR}/{artifact_key}.pkl

    Args:
        mode: "single" or "cv"
        fold: required if mode=="cv"
        seed: random seed
        cfg: config payload (should be JSON-serializable)
        n_neighbors: MI estimator neighbors
        poll_interval_sec: seconds between polls
        timeout_sec: None for no timeout
        serialize_cfg_default_str: if True, json.dumps(..., default=str) to avoid non-serializable objects
        verbose_poll: if True, prints progress every ~10 polls
        use_cache: if True, use local cache if available

    Returns:
        str: local pickle path
    """
    # Stagger workers on first call to avoid simultaneous API burst
    _apply_startup_jitter()

    remove_los = True
    model_name = cfg["model"].get("name", None)
    if model_name in ["gin", "a3tgcn_2_points", "gin_gru_2_points"]:
        remove_los = False

    artifact_key = _artifact_key(mode, fold, seed, n_neighbors, remove_los)
    local_pkl = LOCAL_CACHE_DIR / f"{artifact_key}.pkl"

    # 1) local cache hit
    if use_cache and local_pkl.exists():
        return str(local_pkl)

    # 2) local lock (avoid duplicate request on same node)
    lock_dir = LOCAL_CACHE_DIR / f"{artifact_key}.lock"
    acquired = _acquire_lock(lock_dir, local_pkl, use_cache)
    if not acquired:
        # someone else produced it while we waited
        return str(local_pkl)

    tmp_json = None
    try:
        # re-check after lock
        if use_cache and local_pkl.exists():
            return str(local_pkl)

        request_id = _request_id_from_artifact(artifact_key)
        tmp_json = Path("/tmp") / f"{request_id}.json"

        payload = {
            "request_id": request_id,
            "artifact_key": artifact_key,
            "mode": mode,
            "fold": fold,
            "seed": seed,
            "cfg": cfg,
            "n_neighbors": n_neighbors,
            "use_cache": use_cache,
        }

        dumps_kwargs = {"ensure_ascii": False}
        if serialize_cfg_default_str:
            dumps_kwargs["default"] = str

        tmp_json.write_text(json.dumps(payload, **dumps_kwargs), encoding="utf-8")

        # 3) ensure remote dirs exist
        _ensure_remote_dirs()

        # 4) upload request
        remote_json = f"{REMOTE_BASE}/requests/{request_id}.json"
        _run(["rclone", "copyto", str(tmp_json), remote_json])

        # 5) poll for response
        start = time.time()
        remote_pkl = f"{REMOTE_BASE}/responses/{request_id}.pkl"

        polls = 0
        while True:
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                raise TimeoutError(f"Timed out waiting for {remote_pkl}")

            try:
                # Existence check: success(returncode 0) is enough
                _run(["rclone", "lsf", remote_pkl])
                break
            except RuntimeError:
                polls += 1
                if verbose_poll and polls % 10 == 0:
                    elapsed = int(time.time() - start)
                    print(f"[request_mi] waiting... elapsed={elapsed}s remote={remote_pkl}")
                time.sleep(poll_interval_sec)

        # 6) download to local cache (atomic-ish)
        tmp_local = local_pkl.with_suffix(".pkl.tmp")
        _run(["rclone", "copyto", remote_pkl, str(tmp_local)])
        tmp_local.replace(local_pkl)

        return str(local_pkl)

    finally:
        if tmp_json is not None:
            tmp_json.unlink(missing_ok=True)
        _release_lock(lock_dir)