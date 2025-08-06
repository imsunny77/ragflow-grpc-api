"""RAGFlow REST API client wrapper."""
import os
from typing import Any, Dict, Optional

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
            headers={"Authorization": f"Bearer {config.api_token}"}
        )
    
    async def create_knowledge_base(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a knowledge base."""
        response = await self.client.post(
            "/v1/knowledge_bases",
            json={"name": name, "description": description}
        )
        return {"status": response.status_code == 200, "data": response.json()}
    
    async def upload_document(self, kb_id: str, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Upload document to knowledge base."""
        files = {"file": (filename, file_data)}
        response = await self.client.post(f"/v1/knowledge_bases/{kb_id}/documents", files=files)
        return {"status": response.status_code == 200, "data": response.json()}
    
    async def chat(self, kb_id: str, question: str) -> Dict[str, Any]:
        """Chat with knowledge base."""
        response = await self.client.post(
            f"/v1/knowledge_bases/{kb_id}/chat",
            json={"question": question}
        )
        return {"status": response.status_code == 200, "data": response.json()}
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()