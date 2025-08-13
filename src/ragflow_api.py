"""RAGFlow REST API client wrapper."""

import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel


class RAGFlowConfig(BaseModel):
    """RAGFlow configuration."""

    base_url: str = "http://localhost:9380"
    api_token: str = "ragflow-EzMDhhMWIyNzI5YTExZjA5ZTUwNDIwMT"


class RAGFlowClient:
    """Simple RAGFlow REST API client."""

    def __init__(self, config: RAGFlowConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_token}"},
            timeout=30.0,
        )

    # Dataset Management Methods
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

    # Document Management Methods
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

    async def list_documents(
        self,
        dataset_id: str,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        keywords: Optional[str] = None,
        document_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List documents in a dataset."""
        try:
            params = {
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": str(desc).lower(),
            }
            if keywords:
                params["keywords"] = keywords
            if document_id:
                params["id"] = document_id
            if name:
                params["name"] = name

            response = await self.client.get(
                f"/api/v1/datasets/{dataset_id}/documents", params=params
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

    async def update_document(
        self, dataset_id: str, document_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update document configuration."""
        try:
            response = await self.client.put(
                f"/api/v1/datasets/{dataset_id}/documents/{document_id}",
                json=update_data,
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

    async def download_document(
        self, dataset_id: str, document_id: str
    ) -> Dict[str, Any]:
        """Download document file."""
        try:
            response = await self.client.get(
                f"/api/v1/datasets/{dataset_id}/documents/{document_id}"
            )

            if response.status_code == 200:
                filename = "downloaded_file"
                content_disposition = response.headers.get("content-disposition", "")
                if "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[-1].strip('"')

                return {"status": True, "data": response.content, "filename": filename}
            else:
                data = response.json() if response.content else {}
                return {
                    "status": False,
                    "error": data.get("message", "Failed to download document"),
                }
        except httpx.ConnectError:
            return {"status": False, "error": "All connection attempts failed"}
        except Exception as e:
            return {"status": False, "error": str(e)}

    async def delete_documents(
        self, dataset_id: str, document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Delete documents from dataset."""
        try:
            payload = {"ids": document_ids} if document_ids else {"ids": None}

            response = await self.client.delete(
                f"/api/v1/datasets/{dataset_id}/documents", json=payload
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

    async def parse_documents(
        self, dataset_id: str, document_ids: List[str]
    ) -> Dict[str, Any]:
        """Start parsing documents into chunks."""
        try:
            payload = {"document_ids": document_ids}

            response = await self.client.post(
                f"/api/v1/datasets/{dataset_id}/chunks", json=payload
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

    # Chat Assistant Management Methods
    async def create_chat_assistant(
        self, assistant_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a chat assistant."""
        try:
            response = await self.client.post("/api/v1/chats", json=assistant_config)
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

    async def list_chat_assistants(
        self,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        name: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List chat assistants with pagination and filtering."""
        try:
            params = {
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": str(desc).lower(),
            }
            if name:
                params["name"] = name
            if chat_id:
                params["id"] = chat_id

            response = await self.client.get("/api/v1/chats", params=params)
            data = response.json() if response.status_code == 200 else {}

            if response.status_code == 200 and "data" in data:
                assistants = (
                    data["data"] if isinstance(data["data"], list) else [data["data"]]
                )
                return {"status": True, "data": assistants}
            else:
                return {
                    "status": False,
                    "data": [],
                    "error": data.get("message", "Failed to list chat assistants"),
                }

        except httpx.ConnectError:
            return {
                "status": False,
                "data": [],
                "error": "All connection attempts failed",
            }
        except Exception as e:
            return {"status": False, "data": [], "error": str(e)}

    async def update_chat_assistant(
        self, chat_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update chat assistant configuration."""
        try:
            response = await self.client.put(
                f"/api/v1/chats/{chat_id}", json=update_data
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

    async def delete_chat_assistants(
        self, assistant_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Delete chat assistants by IDs. If None, deletes all assistants."""
        try:
            payload = {"ids": assistant_ids} if assistant_ids else {"ids": None}

            response = await self.client.delete("/api/v1/chats", json=payload)
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

    # Session Management Methods
    async def create_session(
        self, chat_id: str, name: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a chat session."""
        try:
            session_data = {"name": name}
            if user_id:
                session_data["user_id"] = user_id

            response = await self.client.post(
                f"/api/v1/chats/{chat_id}/sessions", json=session_data
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

    async def list_sessions(
        self,
        chat_id: str,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List chat sessions with pagination and filtering."""
        try:
            params = {
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": str(desc).lower(),
            }
            if name:
                params["name"] = name
            if session_id:
                params["id"] = session_id
            if user_id:
                params["user_id"] = user_id

            response = await self.client.get(
                f"/api/v1/chats/{chat_id}/sessions", params=params
            )
            data = response.json() if response.status_code == 200 else {}

            if response.status_code == 200 and "data" in data:
                sessions = (
                    data["data"] if isinstance(data["data"], list) else [data["data"]]
                )
                return {"status": True, "data": sessions}
            else:
                return {
                    "status": False,
                    "data": [],
                    "error": data.get("message", "Failed to list sessions"),
                }

        except httpx.ConnectError:
            return {
                "status": False,
                "data": [],
                "error": "All connection attempts failed",
            }
        except Exception as e:
            return {"status": False, "data": [], "error": str(e)}

    async def update_session(
        self, chat_id: str, session_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update session configuration."""
        try:
            response = await self.client.put(
                f"/api/v1/chats/{chat_id}/sessions/{session_id}", json=update_data
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

    async def delete_sessions(
        self, chat_id: str, session_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Delete sessions from chat assistant."""
        try:
            payload = {"ids": session_ids} if session_ids else {"ids": None}

            response = await self.client.delete(
                f"/api/v1/chats/{chat_id}/sessions", json=payload
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

    # Chunk Management Methods
    async def create_chunk(
        self,
        dataset_id: str,
        document_id: str,
        content: str,
        metadata: Optional[str] = None,
        position: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a document chunk."""
        try:
            chunk_data = {
                "document_id": document_id,
                "content": content,
            }
            if metadata:
                chunk_data["metadata"] = metadata
            if position is not None:
                chunk_data["position"] = position

            response = await self.client.post(
                f"/api/v1/datasets/{dataset_id}/chunks", json=chunk_data
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

    async def list_chunks(
        self,
        dataset_id: str,
        document_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        keywords: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List chunks in a dataset."""
        try:
            params = {
                "page": page,
                "page_size": page_size,
                "orderby": orderby,
                "desc": str(desc).lower(),
            }
            if document_id:
                params["document_id"] = document_id
            if keywords:
                params["keywords"] = keywords
            if chunk_id:
                params["id"] = chunk_id

            response = await self.client.get(
                f"/api/v1/datasets/{dataset_id}/chunks", params=params
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

    async def update_chunk(
        self, dataset_id: str, chunk_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update chunk configuration."""
        try:
            response = await self.client.put(
                f"/api/v1/datasets/{dataset_id}/chunks/{chunk_id}",
                json=update_data,
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

    async def delete_chunks(
        self, dataset_id: str, chunk_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Delete chunks from dataset."""
        try:
            payload = {"ids": chunk_ids} if chunk_ids else {"ids": None}

            response = await self.client.delete(
                f"/api/v1/datasets/{dataset_id}/chunks", json=payload
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

    # OpenAI Compatible API Methods
    async def chat_completions(self, chat_params: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible chat completions."""
        try:
            # Extract RAGFlow-specific parameters
            dataset_id = chat_params.pop("dataset_id", None)

            # Build OpenAI-compatible request
            completion_data = {
                "messages": chat_params.get("messages", []),
                "model": chat_params.get("model", "ragflow-default"),
                "temperature": chat_params.get("temperature", 0.7),
                "max_tokens": chat_params.get("max_tokens", 1000),
                "top_p": chat_params.get("top_p", 1.0),
                "frequency_penalty": chat_params.get("frequency_penalty", 0.0),
                "presence_penalty": chat_params.get("presence_penalty", 0.0),
                "stream": chat_params.get("stream", False),
                "user": chat_params.get("user"),
            }

            # Remove None values
            completion_data = {
                k: v for k, v in completion_data.items() if v is not None
            }

            # Use OpenAI-compatible endpoint or RAG endpoint
            if dataset_id:
                # Use RAG-enhanced chat with dataset context
                endpoint = f"/api/v1/datasets/{dataset_id}/chat/completions"
            else:
                # Use standard OpenAI-compatible endpoint
                endpoint = "/api/v1/chat/completions"

            response = await self.client.post(endpoint, json=completion_data)
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

    async def create_embeddings(
        self, embedding_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate embeddings for text."""
        try:
            # Build embedding request
            embedding_data = {
                "input": embedding_params.get("input", []),
                "model": embedding_params.get("model", "ragflow-embedding"),
                "encoding_format": embedding_params.get("encoding_format", "float"),
                "user": embedding_params.get("user"),
            }

            # Remove None values
            embedding_data = {k: v for k, v in embedding_data.items() if v is not None}

            response = await self.client.post("/api/v1/embeddings", json=embedding_data)
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

    async def list_models(self) -> Dict[str, Any]:
        """List available models."""
        try:
            response = await self.client.get("/api/v1/models")
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

    # Retrieval/Search Methods
    async def search_documents(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filter_criteria: Optional[str] = None,
        include_content: bool = False,
    ) -> Dict[str, Any]:
        """Search documents using semantic search."""
        try:
            search_data = {
                "query": query,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "include_content": include_content,
            }
            if filter_criteria:
                search_data["filter"] = filter_criteria

            response = await self.client.post(
                f"/api/v1/datasets/{dataset_id}/search/documents", json=search_data
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

    async def retrieve_chunks(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.2,
        document_id: Optional[str] = None,
        rerank: bool = True,
    ) -> Dict[str, Any]:
        """Retrieve relevant chunks for RAG."""
        try:
            retrieve_data = {
                "query": query,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "rerank": rerank,
            }
            if document_id:
                retrieve_data["document_id"] = document_id

            response = await self.client.post(
                f"/api/v1/datasets/{dataset_id}/retrieve", json=retrieve_data
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

    async def similarity_search(
        self,
        dataset_id: str,
        text: Optional[str] = None,
        embedding: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        content_type: str = "both",
    ) -> Dict[str, Any]:
        """Perform similarity search using embeddings."""
        try:
            if not text and not embedding:
                return {
                    "status": False,
                    "data": {},
                    "error": "Either text or embedding must be provided",
                }

            search_data = {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "content_type": content_type,
            }
            if text:
                search_data["text"] = text
            if embedding:
                search_data["embedding"] = embedding

            response = await self.client.post(
                f"/api/v1/datasets/{dataset_id}/search/similarity", json=search_data
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

    # Chat Methods
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
