from fastapi import FastAPI
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.core.settings import settings
from app.core.logger import logger
from app.models.media import VideoMetadata, KeyframeMetadata

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize MongoDB and Beanie
    logger.info("Initializing database connections on startup...")
    try:
        client = AsyncIOMotorClient(settings.MONGO_URI)
        await init_beanie(
            database=client[settings.MONGO_DB_NAME],
            document_models=[VideoMetadata, KeyframeMetadata]
        )
        logger.info("MongoDB initialized successfully with Beanie.")
    except Exception as e:
        logger.error(f"Error during startup lifespan initialization: {e}")

    yield

    # Shutdown: Clean up resources if necessary
    logger.info("Shutting down application...")
