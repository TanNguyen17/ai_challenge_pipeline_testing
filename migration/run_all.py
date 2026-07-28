import subprocess
from app.core.logger import logger

def run_script(script_path: str):
    logger.info(f"Running migration script: {script_path}")
    try:
        res = subprocess.run(["python", script_path], check=True, capture_output=True, text=True)
        logger.info(f"SUCCESS: {script_path}\n{res.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ERROR: Failed running {script_path}\nStderr: {e.stderr}\nStdout: {e.stdout}")

def main():
    logger.info("Starting all migration tasks...")
    
    # 1. MongoDB migration (media-info data)
    run_script("d:/AI-HCMC/migration/mongo_migration.py")
    
    # 2. Milvus migration (numpy embedding data)
    run_script("d:/AI-HCMC/migration/milvus_migration.py")
    
    # 3. Elasticsearch migration (text indexing)
    run_script("d:/AI-HCMC/migration/es_migration.py")
    
    # 4. MinIO migration (keyframe images upload)
    run_script("d:/AI-HCMC/migration/minio_migration.py")
    
    logger.info("All migrations completed!")

if __name__ == "__main__":
    main()
