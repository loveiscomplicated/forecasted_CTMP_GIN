import wandb
import json
import os


def upload_jsonl_to_wandb(
    file_path, project_name, run_name=None, group_name=None, config=None, job_type=None
):
    if not os.path.exists(file_path):
        print(f"Error: 파일을 찾을 수 없습니다. ({file_path})")
        return

    wandb.init(
        project=project_name,
        name=run_name,
        group=group_name,
        config=config,
        job_type=job_type,
    )

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # Case 1: 'split': 'test' 가 포함된 최종 결과 행
            if data.get("split") == "test":
                # 별도의 prefix 없이 원래 키 이름 그대로 summary에 강제 업데이트
                for k, v in data.items():
                    if k not in ["split", "epoch"]:
                        # wandb.log로 찍혔던 마지막 값보다 summary 값을 우선시함
                        wandb.run.summary[k] = v
                print(f"[{run_name}] Test metrics updated to summary.")

            # Case 2: 일반적인 학습/검증 지표 (epoch 기반)
            elif "epoch" in data:
                step = data.pop("epoch")
                wandb.log(data, step=step)

            # Case 3: 기타 데이터
            else:
                wandb.log(data)

    wandb.finish()


if __name__ == "__main__":
    import re

    cur_dir = os.path.dirname(__file__)
    runs_protected_path = os.path.join(cur_dir, "..", "..", "runs", "protected")
    model_name = "ctmp_gin"
    run_dir = (
        "(fuck)20260429-074321__ctmp_gin__bs=512__lr=6.10e-04__seed=3__cv=5__test=0.15"
    )
    common_path = os.path.join(runs_protected_path, "k_fold_CV", run_dir, "folds")
    # common_path = os.path.join(runs_protected_path, "ablation", run_dir, "folds")

    match = re.search(r"__seed=(\d+)__", run_dir)
    if match is None:
        raise ValueError(f"seed를 run_dir에서 찾을 수 없습니다: {run_dir}")
    seed = int(match.group(1))
    # 자동화 예시
    for i in range(5):
        fold_path = os.path.join(common_path, f"fold_{i}", "metrics.jsonl")
        run_name = f"{model_name}_seed{seed}_fold{i}"

        # config에 fold 번호를 넣어두면 나중에 W&B Table에서 필터링하기 매우 좋습니다.
        config = {"seed": seed, "fold": i, "model": model_name}

        upload_jsonl_to_wandb(
            file_path=fold_path,
            project_name="ctmp_gin_final",
            group_name=f"{model_name}_kfold",  # 시드별로 그룹을 묶는 것도 방법
            run_name=run_name,
            job_type="kfold",
            config=config,
        )
