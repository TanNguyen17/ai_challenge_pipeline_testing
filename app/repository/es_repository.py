from elasticsearch import Elasticsearch
from app.core.settings import settings
from app.core.logger import logger
from typing import List, Dict, Any, Optional

class ESRepository:
    def __init__(self):
        hosts = [f"{settings.ES_SCHEME}://{settings.ES_HOST}:{settings.ES_PORT}"]
        auth = None
        if settings.ES_USER and settings.ES_PASSWORD:
            auth = (settings.ES_USER, settings.ES_PASSWORD)
            
        try:
            self.client = Elasticsearch(hosts, basic_auth=auth)
            if self.client.ping():
                logger.info("Successfully connected to Elasticsearch.")
            else:
                logger.error("Elasticsearch ping failed.")
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {e}")

    def create_index(self, index_name: str):
        # We define settings with CocCoc Vietnamese tokenizer if available, otherwise fallback
        # Let's specify index mapping for OCR / ASR transcripts
        settings_body = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "vietnamese_analyzer": {
                            "type": "custom",
                            "tokenizer": "vi_tokenizer",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "video_id": {"type": "keyword"},
                    "frame_idx": {"type": "long"},
                    "text": {
                        "type": "text",
                        "analyzer": "vietnamese_analyzer"
                    }
                }
            }
        }
        
        # In case CocCoc analysis plugin vi_tokenizer is not installed, fallback to standard tokenizer
        fallback_body = {
            "mappings": {
                "properties": {
                    "video_id": {"type": "keyword"},
                    "frame_idx": {"type": "long"},
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    }
                }
            }
        }
        
        if self.client.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists.")
            return

        try:
            self.client.indices.create(index=index_name, body=settings_body)
            logger.info(f"Created index {index_name} with CocCoc tokenizer.")
        except Exception as e:
            logger.warn(f"Failed to create index with vi_tokenizer: {e}. Trying fallback standard analyzer.")
            try:
                self.client.indices.create(index=index_name, body=fallback_body)
                logger.info(f"Created fallback index {index_name} with standard analyzer.")
            except Exception as ex:
                logger.error(f"Failed to create fallback index {index_name}: {ex}")

    def index_document(self, index_name: str, doc_id: str, video_id: str, frame_idx: int, text: str):
        body = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "text": text
        }
        self.client.index(index=index_name, id=doc_id, document=body)

    def search(self, index_name: str, query_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        query = {
            "match": {
                "text": {
                    "query": query_text,
                    "operator": "or"
                }
            }
        }
        try:
            res = self.client.search(
                index=index_name,
                query=query,
                size=top_k
            )
            hits = []
            for hit in res["hits"]["hits"]:
                source = hit["_source"]
                hits.append({
                    "video_id": source["video_id"],
                    "frame_idx": source["frame_idx"],
                    "score": hit["_score"]
                })
            return hits
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []
