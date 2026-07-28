import httpx
from app.core.settings import settings
from app.core.logger import logger
import base64

class LLMFactory:
    @staticmethod
    async def generate_text_deepseek(prompt: str, system_prompt: str = "You are a helpful assistant") -> str:
        if settings.DEEPSEEK_API_KEY == "mock_key":
            logger.info("Mocking DeepSeek text generation.")
            return "Mocked DeepSeek response."

        headers = {
            "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.DEEPSEEK_API_BASE}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60.0
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    logger.error(f"DeepSeek API error: {response.text}")
                    return ""
        except Exception as e:
            logger.error(f"DeepSeek request exception: {e}")
            return ""

    @staticmethod
    async def generate_vision_glm(prompt: str, image_bytes: bytes) -> str:
        if settings.ZHIPU_API_KEY == "mock_key":
            logger.info("Mocking Zhipu GLM vision generation.")
            return "Mocked GLM response."

        headers = {
            "Authorization": f"Bearer {settings.ZHIPU_API_KEY}",
            "Content-Type": "application/json"
        }
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data = {
            "model": "glm-4v",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60.0
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    logger.error(f"ZhipuAI API error: {response.text}")
                    return ""
        except Exception as e:
            logger.error(f"ZhipuAI request exception: {e}")
            return ""
