from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from app.core.settings import settings
from app.core.logger import logger
from typing import List, Dict, Any

class MilvusRepository:
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.user = settings.MILVUS_USER
        self.password = settings.MILVUS_PASSWORD
        self.connect()

    def connect(self):
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            logger.info("Successfully connected to Milvus.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")

    def create_collection(self, collection_name: str, dim: int = 512):
        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists.")
            return Collection(collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="frame_idx", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, description=f"Collection for {collection_name}")
        collection = Collection(name=collection_name, schema=schema)

        # Create HNSW index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 30, "efConstruction": 360}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        logger.info(f"Created collection {collection_name} with HNSW index.")
        return collection

    def insert(self, collection_name: str, entities: List[Dict[str, Any]]):
        collection = Collection(collection_name)
        data = [
            [entity["video_id"] for entity in entities],
            [entity["frame_idx"] for entity in entities],
            [entity["embedding"] for entity in entities]
        ]
        res = collection.insert(data)
        collection.flush()
        return res

    def search(self, collection_name: str, query_vector: List[float], top_k: int = 100) -> List[Dict[str, Any]]:
        collection = Collection(collection_name)
        collection.load()
        search_params = {"metric_type": "COSINE", "params": {"ef": 256}}
        
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["video_id", "frame_idx"]
        )

        hits = []
        if results:
            for hit in results[0]:
                hits.append({
                    "video_id": hit.entity.get("video_id"),
                    "frame_idx": hit.entity.get("frame_idx"),
                    "score": hit.score
                })
        return hits
