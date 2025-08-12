"""RAGFlow REST API client wrapper."""

import os
from typing import Any, Dict, List, Optional

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
    ) -> Dict[str, Any]:
        """Create a knowledge base."""
        try:
            response = await self.client.post(
                "/api/v1/datasets", json={"name": name, "description": description}
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

    async def list_datasets(
        self,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List datasets with pagination and filtering."""
        try:
            params = {
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": str(desc).lower(),
            }
            if name:
                params["name"] = name
            if dataset_id:
                params["id"] = dataset_id

            response = await self.client.get("/api/v1/datasets", params=params)
            data = response.json() if response.status_code == 200 else {}

            # Handle both single dataset and list responses
            if response.status_code == 200 and "data" in data:
                datasets = (
                    data["data"] if isinstance(data["data"], list) else [data["data"]]
                )
                return {"status": True, "data": datasets}
            else:
                return {
                    "status": False,
                    "data": [],
                    "error": data.get("message", "Failed to list datasets"),
                }

        except httpx.ConnectError:
            return {
                "status": False,
                "data": [],
                "error": "All connection attempts failed",
            }
        except Exception as e:
            return {"status": False, "data": [], "error": str(e)}

    async def update_dataset(
        self, dataset_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update dataset configuration."""
        try:
            response = await self.client.put(
                f"/api/v1/datasets/{dataset_id}", json=update_data
            )
            data = response.json() if response.content else {}
            return {"status": response.status_code == 200, "data": data}
        except httpx.ConnectError:
            return {
                "status": False,
                "data": {},
                "error": "All connection attempts failed",
            }
        except Exception as e:
            return {"status": False, "data": {}, "error": str(e)}

    async def delete_datasets(
        self, dataset_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Delete datasets by IDs. If None, deletes all datasets."""
        try:
            payload = {"ids": dataset_ids} if dataset_ids else {"ids": None}

            response = await self.client.delete("/api/v1/datasets", json=payload)
            data = response.json() if response.content else {}
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
    ) -> Dict[str, Any]:
        """Upload document to knowledge base."""
        try:
            files = {"file": (filename, file_data)}
            response = await self.client.post(
                f"/api/v1/datasets/{kb_id}/documents", files=files
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

    async def chat(self, kb_id: str, question: str) -> Dict[str, Any]:
        """Chat with knowledge base."""
        try:
            response = await self.client.post(
                f"/api/v1/datasets/{kb_id}/chat", json={"question": question}
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
