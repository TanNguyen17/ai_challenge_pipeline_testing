from fastapi import FastAPI
from app.core.lifespan import lifespan
from app.router.health import router as health_router

app = FastAPI(
    title="AI Challenge HCMC 2026 Engine",
    description="Multimodal Video Retrieval backend engine using FastAPI, Milvus, Elasticsearch, MongoDB, and MinIO",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(health_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to AI Challenge HCMC 2026 Retrieval Engine API.",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    from app.core.settings import settings
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
