import os
import optuna

def backup_to_sql(model_name):
    postgres_url = "postgresql+psycopg2://optuna:optuna_pw@127.0.0.1:5432/optuna_db"

    # 2. 저장 폴더 생성 (없으면 만들기)
    save_dir = "./runs"
    os.makedirs(save_dir, exist_ok=True)

    # 3. SQLite 경로 설정 (프로젝트 루트 기준 상대 경로)
    sqlite_path = os.path.join(save_dir, f"{model_name}_optuna.db")
    sqlite_url = f"sqlite:///{sqlite_path}"

    # 4. PostgreSQL의 모든 스터디 가져오기 및 복사
    summaries = optuna.get_all_study_summaries(storage=postgres_url)

    for summary in summaries:
        study_name = summary.study_name
        print(f"복사 중: {study_name} -> {sqlite_path}")

        try:
            # 이미 존재하는 study는 삭제 후 재복사 (덮어쓰기)
            existing = [s.study_name for s in optuna.get_all_study_summaries(storage=sqlite_url)]
            if study_name in existing:
                optuna.delete_study(study_name=study_name, storage=sqlite_url)

            optuna.copy_study(
                from_study_name=study_name,
                from_storage=postgres_url,
                to_storage=sqlite_url,
                to_study_name=study_name
            )
        except Exception as e:
            print(f"❌ {study_name} 복사 실패: {e}")

    print(f"\n✅ 완료! 파일 위치: {os.path.abspath(sqlite_path)}")