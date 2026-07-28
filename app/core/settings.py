from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Milvus settings
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    
    # Elasticsearch settings
    ES_HOST: str = "localhost"
    ES_PORT: int = 9200
    ES_SCHEME: str = "http"
    ES_USER: Optional[str] = None
    ES_PASSWORD: Optional[str] = None

    # MongoDB settings
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "ai_challenge_2026"

    # MinIO settings
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "keyframes"

    # LLM Settings
    ZHIPU_API_KEY: str = "mock_key"
    DEEPSEEK_API_KEY: str = "mock_key"
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com"

    # Encoder settings
    VISUAL_ENCODER_MODEL: str = "openai/clip-vit-base-patch32"  # or timm/PE-Core-bigG-14-448
    VN_TEXT_ENCODER_MODEL: str = "dangvantuan/vietnamese-embedding"

settings = Settings()
