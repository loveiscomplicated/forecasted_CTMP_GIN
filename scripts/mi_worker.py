import os
import sys
import time
import json
import pickle
import subprocess
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.data_utils import train_test_split_stratified
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.utils.device_set import device_set

REMOTE_BASE = "gdrive:CTMP_GIN_mi_service"
LOCAL_CACHE = os.path.expanduser("~/mi_cache")
REQUESTS_DIR = f"{REMOTE_BASE}/requests"
RESPONSES_DIR = f"{REMOTE_BASE}/responses"
DONE_DIR = f"{REMOTE_BASE}/done"

# [ADD] 실패 파일 보관 폴더
FAILED_DIR = f"{REMOTE_BASE}/failed"
# (선택) 처리중 폴더(동시 워커 대비)
PROCESSING_DIR = f"{REMOTE_BASE}/processing"

os.makedirs(LOCAL_CACHE, exist_ok=True)

# [ADD] retry 설정
MAX_RETRIES = 3
RETRY_SLEEP = 2  # seconds

def rclone_mkdir(remote_dir: str):
    subprocess.run(["rclone", "mkdir", remote_dir], check=True)

def rclone_copy(src, dst):
    subprocess.run(["rclone", "copyto", src, dst], check=True)


def rclone_move(src, dst):
    subprocess.run(["rclone", "moveto", src, dst], check=True)


def rclone_list(remote_path):
    result = subprocess.run(
        ["rclone", "lsf", remote_path],
        check=True,
        capture_output=True,
        text=True
    )
    return result.stdout.splitlines()


def _get_mi_helper(df: pd.DataFrame, seed: int, n_neighbors: int):
    mi_dict = {}
    for col in tqdm(df.columns):
        x = df.drop(col, axis=1)
        y = df[col]
        mi = mutual_info_classif(
            x,
            y,
            discrete_features=True,
            n_neighbors=n_neighbors,
            random_state=seed
        )
        mi_series = pd.Series(mi, index=x.columns)
        mi_dict[col] = mi_series
    return mi_dict


def load_train_df(mode, fold, seed, cfg, remove_los):
    cur_dir = os.path.dirname(__file__)
    root = os.path.join(cur_dir, '..', 'src', 'data')


    if mode == "single":
        dataset = TEDSTensorDataset(
            root=root,
            binary=cfg["train"].get("binary", True),
            ig_label=cfg["train"].get("ig_label", False),
            remove_los=remove_los,
        )

        cfg["model"]["params"]["col_info"] = dataset.col_info
        cfg["model"]["params"]["num_classes"] = dataset.num_classes
        device = device_set(cfg["device"])
        cfg["model"]["params"]["device"] = device

        num_nodes = len(dataset.col_info[2])
        if cfg["model"]["name"] == 'gin':
            num_nodes = len(dataset.col_info[0]) + 1
        print(f"num_nodes set to {num_nodes}")

        split_ratio = [cfg['train']['train_ratio'], cfg['train']['val_ratio'], cfg['train']['test_ratio']]
        train_loader, val_loader, test_loader, idx = train_test_split_stratified(
            dataset=dataset,  # type: ignore
            batch_size=cfg['train']['batch_size'],
            ratio=split_ratio,
            seed=seed,
            num_workers=cfg['train']['num_workers'],
        )
        train_df = dataset.processed_df.iloc[idx[0]]
        return train_df

    elif mode == "cv":
        raise KeyError("not implemented yet")


def process_one_request_file(fname: str):
    """
    한 request 파일을 처리.
    실패하면 예외를 던져서 상위 retry 로직이 처리하게 함.
    """
    remote_request = f"{REQUESTS_DIR}/{fname}"

    # (선택) 먼저 processing으로 옮겨서 "잡은 파일" 표시 (동시 워커 대비)
    remote_processing = f"{PROCESSING_DIR}/{fname}"
    try:
        rclone_move(remote_request, remote_processing)
        remote_request_in_hand = remote_processing
    except Exception:
        # processing 폴더를 안 쓰거나 move 실패하면 그냥 원래 위치에서 진행
        remote_request_in_hand = remote_request

    local_request = os.path.join("/tmp", fname)

    print(f"Processing {fname}")
    rclone_copy(remote_request_in_hand, local_request)

    with open(local_request) as f:
        req = json.load(f)

    artifact_key = req["artifact_key"]
    local_cache_path = os.path.join(LOCAL_CACHE, f"{artifact_key}.pkl")

    # 요청에서 캐시 사용 여부를 확인 (기본값 True)
    use_cache = req.get("use_cache", True)

    if use_cache and os.path.exists(local_cache_path):
        print(f"Cache hit for {artifact_key}.")
    else:
        if not use_cache:
            print(f"Cache bypass requested for {artifact_key}. Computing MI...")
        else:
            print(f"Cache miss for {artifact_key}. Computing MI...")
        
        remove_los = True
        model_name = req["cfg"]["model"].get("name", None)
        if model_name in ["gin", "gin_gru_2_points", "a3tgcn_2_points"]:
            remove_los = False
        
        train_df = load_train_df(
            req["mode"],
            req.get("fold"),
            req["seed"],
            req["cfg"],
            remove_los=remove_los
        )
        # Remove target columns (REASON/REASONb) before computing MI
        # to avoid including them as nodes in the graph edge index.
        for _label in ["REASON", "REASONb"]:
            if _label in train_df.columns:
                train_df = train_df.drop(_label, axis=1)
        mi_dict = _get_mi_helper(
            train_df,
            req["seed"],
            req["n_neighbors"],
        )

        # 계산 결과를 로컬 캐시에 저장 (다음 번에 use_cache=True인 요청이 오면 활용됨)
        with open(local_cache_path, "wb") as f:
            pickle.dump(mi_dict, f)

    remote_response = f"{RESPONSES_DIR}/{req['request_id']}.pkl"
    rclone_copy(local_cache_path, remote_response)

    # done 처리
    # (processing에 있었으면 processing -> done, 아니면 requests -> done)
    rclone_move(remote_request_in_hand, f"{DONE_DIR}/{fname}")
    print("Done.")


def main():
    print("MI worker started...")

    while True:
        request_files = []
        try:
            request_files = rclone_list(REQUESTS_DIR)
        except Exception as e:
            print("Error listing requests:", e)
            time.sleep(5)
            continue

        for fname in request_files:
            if not fname.endswith(".json"):
                continue

            ok = False
            last_err = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    process_one_request_file(fname)
                    ok = True
                    break
                except Exception as e:
                    last_err = e
                    print(f"[{fname}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
                    time.sleep(RETRY_SLEEP)

            if not ok:
                # 몇 번 시도해도 실패 -> failed로 이동시켜 큐에서 제거
                print(f"[{fname}] failed after {MAX_RETRIES} retries. Moving to failed.")
                try:
                    # requests에 남아있을 수도, processing에 있을 수도 있음
                    # 1) requests에 있으면 이동
                    rclone_move(f"{REQUESTS_DIR}/{fname}", f"{FAILED_DIR}/{fname}")
                except Exception:
                    try:
                        # 2) processing에 있으면 이동
                        rclone_move(f"{PROCESSING_DIR}/{fname}", f"{FAILED_DIR}/{fname}")
                    except Exception as e2:
                        print(f"[{fname}] could not move to failed: {e2} (original error: {last_err})")

        time.sleep(5)


if __name__ == "__main__":
    main()