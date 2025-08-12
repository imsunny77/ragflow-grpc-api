"""RAGFlow REST API client wrapper."""

from typing import Any

import httpx
from pydantic import BaseModel


class RAGFlowConfig(BaseModel):
    """RAGFlow configuration."""

    base_url: str = "http://localhost:9380"
    api_token: str = ""


class RAGFlowClient:
    """Simple RAGFlow REST API client."""

    def __init__(self, config: RAGFlowConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_token}"},
            timeout=30.0,
        )

    async def create_knowledge_base(
        self, name: str, description: str = ""
    ) -> dict[str, Any]:
        """Create a knowledge base."""
        try:
            response = await self.client.post(
                "/v1/datasets", json={"name": name, "description": description}
            )
            data = response.json() if response.status_code == 200 else {}
            return {"status": response.status_code == 200, "data": data}
        except httpx.ConnectError:
            return {
                "status": False,
                "data": {},
                "error": "All connection attempts failed",
            }
        except Exception as e:
            return {"status": False, "data": {}, "error": str(e)}

    async def upload_document(
        self, kb_id: str, file_data: bytes, filename: str
    ) -> dict[str, Any]:
        """Upload document to knowledge base."""
        try:
            files = {"file": (filename, file_data)}
            response = await self.client.post(
                f"/v1/datasets/{kb_id}/documents", files=files
            )
            data = response.json() if response.status_code == 200 else {}
            return {"status": response.status_code == 200, "data": data}
        except httpx.ConnectError:
            return {
                "status": False,
                "data": {},
                "error": "All connection attempts failed",
            }
        except Exception as e:
            return {"status": False, "data": {}, "error": str(e)}

    async def chat(self, kb_id: str, question: str) -> dict[str, Any]:
        """Chat with knowledge base."""
        try:
            response = await self.client.post(
                f"/v1/datasets/{kb_id}/chat", json={"question": question}
            )
            data = response.json() if response.status_code == 200 else {}
            return {"status": response.status_code == 200, "data": data}
        except httpx.ConnectError:
            return {
                "status": False,
                "data": {},
                "error": "All connection attempts failed",
            }
        except Exception as e:
            return {"status": False, "data": {}, "error": str(e)}

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
