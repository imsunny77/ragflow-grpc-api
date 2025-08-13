"""Async gRPC server for RAGFlow."""

import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any


sys.path.append(os.path.join(os.path.dirname(__file__)))

import grpc
from dotenv import load_dotenv

import ragflow_pb2
import ragflow_pb2_grpc
from ragflow_api import RAGFlowClient, RAGFlowConfig

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RagServicesServicer(ragflow_pb2_grpc.RagServicesServicer):
    """gRPC servicer for RAGFlow operations."""

    def __init__(self) -> None:
        config = RAGFlowConfig(
            base_url=os.getenv("RAGFLOW_BASE_URL", "http://localhost:9380"),
            api_token=os.getenv("RAGFLOW_API_TOKEN", "demo_token"),
        )
        self.ragflow_client = RAGFlowClient(config)

    # Dataset Management Methods
    async def CreateKnowledgeBase(
        self,
        request: ragflow_pb2.CreateKnowledgeBaseRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.CreateKnowledgeBaseResponse:
        """Create knowledge base."""
        try:
            result = await self.ragflow_client.create_knowledge_base(
                request.name, request.description or ""
            )
            kb_id = ""
            if result["status"] and result.get("data"):
                kb_id = result["data"].get("data", {}).get("id", "") or result[
                    "data"
                ].get("id", "")

            return ragflow_pb2.CreateKnowledgeBaseResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                kb_id=kb_id,
            )
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            return ragflow_pb2.CreateKnowledgeBaseResponse(status=False, message=str(e))

    async def ListDatasets(
        self,
        request: ragflow_pb2.ListDatasetsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.ListDatasetsResponse:
        """List datasets."""
        try:
            result = await self.ragflow_client.list_datasets(
                page=request.page or 1,
                page_size=request.page_size or 30,
                orderby=request.orderby or "create_time",
                desc=request.desc,
                name=request.name or None,
                dataset_id=request.id or None,
            )

            datasets = []
            if result["status"] and result.get("data"):
                for dataset_data in result["data"]:
                    dataset = ragflow_pb2.Dataset(
                        id=dataset_data.get("id", ""),
                        name=dataset_data.get("name", ""),
                        description=dataset_data.get("description") or "",
                        avatar=dataset_data.get("avatar") or "",
                        embedding_model=dataset_data.get("embedding_model", ""),
                        permission=dataset_data.get("permission", ""),
                        chunk_method=dataset_data.get("chunk_method", ""),
                        chunk_count=dataset_data.get("chunk_count", 0),
                        document_count=dataset_data.get("document_count", 0),
                        create_date=dataset_data.get("create_date", ""),
                        update_date=dataset_data.get("update_date", ""),
                    )
                    datasets.append(dataset)

            return ragflow_pb2.ListDatasetsResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                datasets=datasets,
            )
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return ragflow_pb2.ListDatasetsResponse(
                status=False, message=str(e), datasets=[]
            )

    async def UpdateDataset(
        self,
        request: ragflow_pb2.UpdateDatasetRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Update dataset."""
        try:
            update_data = {}
            if request.name:
                update_data["name"] = request.name
            if request.description:
                update_data["description"] = request.description
            if request.embedding_model:
                update_data["embedding_model"] = request.embedding_model
            if request.permission:
                update_data["permission"] = request.permission
            if request.chunk_method:
                update_data["chunk_method"] = request.chunk_method

            result = await self.ragflow_client.update_dataset(
                request.dataset_id, update_data
            )

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error updating dataset: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    async def DeleteDatasets(
        self,
        request: ragflow_pb2.DeleteDatasetsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Delete datasets."""
        try:
            dataset_ids = list(request.ids) if request.ids else []
            result = await self.ragflow_client.delete_datasets(dataset_ids)

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error deleting datasets: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    # Document Management Methods
    async def UploadDocument(
        self,
        request: ragflow_pb2.UploadDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Upload document."""
        try:
            result = await self.ragflow_client.upload_document(
                request.kb_id, request.file_data, request.filename
            )
            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    async def ListDocuments(
        self,
        request: ragflow_pb2.ListDocumentsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.ListDocumentsResponse:
        """List documents in a dataset."""
        try:
            result = await self.ragflow_client.list_documents(
                dataset_id=request.dataset_id,
                page=request.page or 1,
                page_size=request.page_size or 30,
                orderby=request.orderby or "create_time",
                desc=request.desc,
                keywords=request.keywords or None,
                document_id=request.id or None,
                name=request.name or None,
            )

            documents = []
            total = 0
            if result["status"] and result.get("data"):
                doc_data = result["data"]
                total = doc_data.get("total", 0)
                docs_list = doc_data.get("docs", [])

                for doc_info in docs_list:
                    document = ragflow_pb2.Document(
                        id=doc_info.get("id", ""),
                        name=doc_info.get("name", ""),
                        dataset_id=doc_info.get("dataset_id", "")
                        or doc_info.get("knowledgebase_id", ""),
                        size=doc_info.get("size", 0),
                        type=doc_info.get("type", ""),
                        chunk_method=doc_info.get("chunk_method", ""),
                        chunk_count=doc_info.get("chunk_count", 0),
                        status=doc_info.get("status", ""),
                        create_date=doc_info.get("create_date", ""),
                        update_date=doc_info.get("update_date", ""),
                        thumbnail=doc_info.get("thumbnail") or "",
                    )
                    documents.append(document)

            return ragflow_pb2.ListDocumentsResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                documents=documents,
                total=total,
            )
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return ragflow_pb2.ListDocumentsResponse(
                status=False, message=str(e), documents=[], total=0
            )

    async def UpdateDocument(
        self,
        request: ragflow_pb2.UpdateDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Update document configuration."""
        try:
            update_data = {}
            if request.name:
                update_data["name"] = request.name
            if request.chunk_method:
                update_data["chunk_method"] = request.chunk_method
            if request.parser_config:
                import json

                try:
                    update_data["parser_config"] = json.loads(request.parser_config)
                except json.JSONDecodeError:
                    return ragflow_pb2.StatusResponse(
                        status=False, message="Invalid parser_config JSON"
                    )

            result = await self.ragflow_client.update_document(
                request.dataset_id, request.document_id, update_data
            )

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    async def DownloadDocument(
        self,
        request: ragflow_pb2.DownloadDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.DownloadDocumentResponse:
        """Download document file."""
        try:
            result = await self.ragflow_client.download_document(
                request.dataset_id, request.document_id
            )

            if result["status"]:
                return ragflow_pb2.DownloadDocumentResponse(
                    status=True,
                    message="Success",
                    file_data=result.get("data", b""),
                    filename=result.get("filename", "downloaded_file"),
                )
            else:
                return ragflow_pb2.DownloadDocumentResponse(
                    status=False, message=result.get("error", "Failed to download")
                )
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            return ragflow_pb2.DownloadDocumentResponse(status=False, message=str(e))

    async def DeleteDocuments(
        self,
        request: ragflow_pb2.DeleteDocumentsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Delete documents from dataset."""
        try:
            document_ids = list(request.ids) if request.ids else []

            result = await self.ragflow_client.delete_documents(
                request.dataset_id, document_ids
            )

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    async def ParseDocuments(
        self,
        request: ragflow_pb2.ParseDocumentsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Start parsing documents into chunks."""
        try:
            document_ids = list(request.document_ids)

            result = await self.ragflow_client.parse_documents(
                request.dataset_id, document_ids
            )

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error parsing documents: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    # Chat Assistant Management Methods
    async def CreateChatAssistant(
        self,
        request: ragflow_pb2.CreateChatAssistantRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.CreateChatAssistantResponse:
        """Create chat assistant."""
        try:
            # Build assistant configuration
            assistant_config = {
                "name": request.name,
                "description": request.description or "",
                "avatar": request.avatar or "",
                "dataset_ids": list(request.dataset_ids) if request.dataset_ids else [],
                "llm": {
                    "model_name": request.llm_model or "default",
                    "temperature": (
                        request.temperature
                    ),
                    "top_p": request.top_p,
                    "presence_penalty": (
                        request.presence_penalty
                        if request.presence_penalty != 0.0
                        else 0.4
                    ),
                    "frequency_penalty": (
                        request.frequency_penalty
                        if request.frequency_penalty != 0.0
                        else 0.7
                    ),
                },
                "prompt": {
                    "prompt": request.prompt or "You are a helpful assistant.",
                    "similarity_threshold": (
                        request.similarity_threshold
                        if request.similarity_threshold != 0.0
                        else 0.2
                    ),
                    "keywords_similarity_weight": (
                        request.keywords_similarity_weight
                        if request.keywords_similarity_weight != 0.0
                        else 0.7
                    ),
                    "top_n": request.top_n,
                },
            }

            result = await self.ragflow_client.create_chat_assistant(assistant_config)

            chat_id = ""
            if result["status"] and result.get("data"):
                chat_id = result["data"].get("data", {}).get("id", "") or result[
                    "data"
                ].get("id", "")

            return ragflow_pb2.CreateChatAssistantResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                chat_id=chat_id,
            )
        except Exception as e:
            logger.error(f"Error creating chat assistant: {e}")
            return ragflow_pb2.CreateChatAssistantResponse(status=False, message=str(e))

    async def ListChatAssistants(
        self,
        request: ragflow_pb2.ListChatAssistantsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.ListChatAssistantsResponse:
        """List chat assistants."""
        try:
            result = await self.ragflow_client.list_chat_assistants(
                page=request.page or 1,
                page_size=request.page_size or 30,
                orderby=request.orderby or "create_time",
                desc=request.desc,
                name=request.name or None,
                chat_id=request.id or None,
            )

            assistants = []
            if result["status"] and result.get("data"):
                for assistant_data in result["data"]:
                    llm_config = assistant_data.get("llm", {})
                    prompt_config = assistant_data.get("prompt", {})

                    assistant = ragflow_pb2.ChatAssistant(
                        id=assistant_data.get("id", ""),
                        name=assistant_data.get("name", ""),
                        description=assistant_data.get("description") or "",
                        avatar=assistant_data.get("avatar") or "",
                        llm_model=llm_config.get("model_name", ""),
                        temperature=llm_config.get("temperature", 0.1),
                        top_p=llm_config.get("top_p", 0.3),
                        presence_penalty=llm_config.get("presence_penalty", 0.4),
                        frequency_penalty=llm_config.get("frequency_penalty", 0.7),
                        prompt=prompt_config.get("prompt", ""),
                        similarity_threshold=prompt_config.get(
                            "similarity_threshold", 0.2
                        ),
                        keywords_similarity_weight=prompt_config.get(
                            "keywords_similarity_weight", 0.7
                        ),
                        top_n=prompt_config.get("top_n", 6),
                        create_date=assistant_data.get("create_date", ""),
                        update_date=assistant_data.get("update_date", ""),
                    )

                    # Add dataset IDs
                    dataset_ids = assistant_data.get("dataset_ids", [])
                    if dataset_ids:
                        assistant.dataset_ids.extend(dataset_ids)

                    assistants.append(assistant)

            return ragflow_pb2.ListChatAssistantsResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                assistants=assistants,
            )
        except Exception as e:
            logger.error(f"Error listing chat assistants: {e}")
            return ragflow_pb2.ListChatAssistantsResponse(
                status=False, message=str(e), assistants=[]
            )

    async def UpdateChatAssistant(
        self,
        request: ragflow_pb2.UpdateChatAssistantRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Update chat assistant configuration."""
        try:
            update_data = {}

            # Basic fields
            if request.name:
                update_data["name"] = request.name
            if request.description:
                update_data["description"] = request.description
            if request.avatar:
                update_data["avatar"] = request.avatar
            if request.dataset_ids:
                update_data["dataset_ids"] = list(request.dataset_ids)

            # LLM configuration
            llm_config = {}
            if request.llm_model:
                llm_config["model_name"] = request.llm_model
            if request.temperature != 0.0:
                llm_config["temperature"] = request.temperature
            if request.top_p != 0.0:
                llm_config["top_p"] = request.top_p
            if request.presence_penalty != 0.0:
                llm_config["presence_penalty"] = request.presence_penalty
            if request.frequency_penalty != 0.0:
                llm_config["frequency_penalty"] = request.frequency_penalty

            if llm_config:
                update_data["llm"] = llm_config

            # Prompt configuration
            prompt_config = {}
            if request.prompt:
                prompt_config["prompt"] = request.prompt
            if request.similarity_threshold != 0.0:
                prompt_config["similarity_threshold"] = request.similarity_threshold
            if request.keywords_similarity_weight != 0.0:
                prompt_config["keywords_similarity_weight"] = (
                    request.keywords_similarity_weight
                )
            if request.top_n != 0:
                prompt_config["top_n"] = request.top_n

            if prompt_config:
                update_data["prompt"] = prompt_config

            result = await self.ragflow_client.update_chat_assistant(
                request.chat_id, update_data
            )

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error updating chat assistant: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    async def DeleteChatAssistants(
        self,
        request: ragflow_pb2.DeleteChatAssistantsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Delete chat assistants."""
        try:
            assistant_ids = list(request.ids) if request.ids else []
            result = await self.ragflow_client.delete_chat_assistants(assistant_ids)

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error deleting chat assistants: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    # Session Management Methods
    async def CreateSession(
        self,
        request: ragflow_pb2.CreateSessionRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.CreateSessionResponse:
        """Create chat session."""
        try:
            result = await self.ragflow_client.create_session(
                request.chat_id, request.name, request.user_id or None
            )
            session_id = ""
            if result["status"] and result.get("data"):
                session_id = result["data"].get("data", {}).get("id", "") or result[
                    "data"
                ].get("id", "")

            return ragflow_pb2.CreateSessionResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return ragflow_pb2.CreateSessionResponse(status=False, message=str(e))

    async def ListSessions(
        self,
        request: ragflow_pb2.ListSessionsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.ListSessionsResponse:
        """List chat sessions."""
        try:
            result = await self.ragflow_client.list_sessions(
                chat_id=request.chat_id,
                page=request.page or 1,
                page_size=request.page_size or 30,
                orderby=request.orderby or "create_time",
                desc=request.desc,
                name=request.name or None,
                session_id=request.id or None,
                user_id=request.user_id or None,
            )

            sessions = []
            if result["status"] and result.get("data"):
                for session_data in result["data"]:
                    messages = []
                    for msg in session_data.get("messages", []):
                        message = ragflow_pb2.ChatMessage(
                            role=msg.get("role", ""),
                            content=msg.get("content", ""),
                        )
                        messages.append(message)

                    session = ragflow_pb2.Session(
                        id=session_data.get("id", ""),
                        chat_id=session_data.get("chat_id", ""),
                        name=session_data.get("name", ""),
                        user_id=session_data.get("user_id") or "",
                        messages=messages,
                        create_date=session_data.get("create_date", ""),
                        update_date=session_data.get("update_date", ""),
                    )
                    sessions.append(session)

            return ragflow_pb2.ListSessionsResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                sessions=sessions,
            )
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return ragflow_pb2.ListSessionsResponse(
                status=False, message=str(e), sessions=[]
            )

    async def UpdateSession(
        self,
        request: ragflow_pb2.UpdateSessionRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Update session configuration."""
        try:
            update_data = {}
            if request.name:
                update_data["name"] = request.name
            if request.user_id:
                update_data["user_id"] = request.user_id

            result = await self.ragflow_client.update_session(
                request.chat_id, request.session_id, update_data
            )

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    async def DeleteSessions(
        self,
        request: ragflow_pb2.DeleteSessionsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Delete chat sessions."""
        try:
            session_ids = list(request.ids) if request.ids else []
            result = await self.ragflow_client.delete_sessions(
                request.chat_id, session_ids
            )

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error deleting sessions: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    # Chunk Management Methods
    async def CreateChunk(
        self,
        request: ragflow_pb2.CreateChunkRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.CreateChunkResponse:
        """Create document chunk."""
        try:
            result = await self.ragflow_client.create_chunk(
                dataset_id=request.dataset_id,
                document_id=request.document_id,
                content=request.content,
                metadata=request.metadata or None,
                position=request.position if request.position != 0 else None,
            )
            chunk_id = ""
            if result["status"] and result.get("data"):
                chunk_id = result["data"].get("data", {}).get("id", "") or result[
                    "data"
                ].get("id", "")

            return ragflow_pb2.CreateChunkResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                chunk_id=chunk_id,
            )
        except Exception as e:
            logger.error(f"Error creating chunk: {e}")
            return ragflow_pb2.CreateChunkResponse(status=False, message=str(e))

    async def ListChunks(
        self,
        request: ragflow_pb2.ListChunksRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.ListChunksResponse:
        """List document chunks."""
        try:
            result = await self.ragflow_client.list_chunks(
                dataset_id=request.dataset_id,
                document_id=request.document_id or None,
                page=request.page or 1,
                page_size=request.page_size or 30,
                orderby=request.orderby or "create_time",
                desc=request.desc,
                keywords=request.keywords or None,
                chunk_id=request.id or None,
            )

            chunks = []
            total = 0
            if result["status"] and result.get("data"):
                chunk_data = result["data"]
                total = chunk_data.get("total", 0)
                chunks_list = chunk_data.get("chunks", [])

                for chunk_info in chunks_list:
                    chunk = ragflow_pb2.Chunk(
                        id=chunk_info.get("id", ""),
                        document_id=chunk_info.get("document_id", ""),
                        dataset_id=chunk_info.get("dataset_id", ""),
                        content=chunk_info.get("content", ""),
                        metadata=chunk_info.get("metadata") or "",
                        position=chunk_info.get("position", 0),
                        similarity_score=chunk_info.get("similarity_score", 0.0),
                        create_date=chunk_info.get("create_date", ""),
                        update_date=chunk_info.get("update_date", ""),
                    )
                    chunks.append(chunk)

            return ragflow_pb2.ListChunksResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                chunks=chunks,
                total=total,
            )
        except Exception as e:
            logger.error(f"Error listing chunks: {e}")
            return ragflow_pb2.ListChunksResponse(
                status=False, message=str(e), chunks=[], total=0
            )

    async def UpdateChunk(
        self,
        request: ragflow_pb2.UpdateChunkRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Update chunk configuration."""
        try:
            update_data = {}
            if request.content:
                update_data["content"] = request.content
            if request.metadata:
                update_data["metadata"] = request.metadata
            if request.position != 0:
                update_data["position"] = request.position

            result = await self.ragflow_client.update_chunk(
                request.dataset_id, request.chunk_id, update_data
            )

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error updating chunk: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    async def DeleteChunks(
        self,
        request: ragflow_pb2.DeleteChunksRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        """Delete chunks from dataset."""
        try:
            chunk_ids = list(request.ids) if request.ids else []
            result = await self.ragflow_client.delete_chunks(
                request.dataset_id, chunk_ids
            )

            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
            )
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    # OpenAI Compatible API Methods
    async def ChatCompletions(
        self,
        request: ragflow_pb2.ChatCompletionsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.ChatCompletionsResponse:
        """OpenAI-compatible chat completions."""
        try:
            # Convert messages to RAGFlow format
            messages = []
            for msg in request.messages:
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "name": msg.name if msg.name else None,
                    }
                )

            # Build request parameters
            chat_params = {
                "messages": messages,
                "model": request.model or "ragflow-default",
                "temperature": (
                    request.temperature
                ),
                "max_tokens": (
                    request.max_tokens
                ),
                "top_p": request.top_p,
                "frequency_penalty": (
                    request.frequency_penalty
                    if request.frequency_penalty != 0.0
                    else 0.0
                ),
                "presence_penalty": (
                    request.presence_penalty
                    if request.presence_penalty != 0.0
                    else 0.0
                ),
                "stream": request.stream,
                "user": request.user if request.user else None,
            }

            # Use dataset_id if provided for RAG
            if request.dataset_id:
                chat_params["dataset_id"] = request.dataset_id

            result = await self.ragflow_client.chat_completions(chat_params)

            choices = []
            usage = None
            response_id = ""
            model_name = ""
            created_timestamp = 0

            if result["status"] and result.get("data"):
                response_data = result["data"]

                # Extract response fields
                response_id = response_data.get(
                    "id", "chatcmpl-" + str(hash(str(messages)))[:8]
                )
                model_name = response_data.get(
                    "model", request.model or "ragflow-default"
                )
                created_timestamp = response_data.get("created", 0)

                # Build choices
                choices_data = response_data.get("choices", [])
                for choice_data in choices_data:
                    message_data = choice_data.get("message", {})

                    choice_message = ragflow_pb2.OpenAIChatMessage(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content", ""),
                        name=(
                            message_data.get("name", "")
                            if message_data.get("name")
                            else None
                        ),
                    )

                    choice = ragflow_pb2.ChatChoice(
                        index=choice_data.get("index", 0),
                        message=choice_message,
                        finish_reason=choice_data.get("finish_reason", "stop"),
                    )
                    choices.append(choice)

                # Build usage info
                usage_data = response_data.get("usage", {})
                if usage_data:
                    usage = ragflow_pb2.ChatUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )

            return ragflow_pb2.ChatCompletionsResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                id=response_id,
                object="chat.completion",
                created=created_timestamp,
                model=model_name,
                choices=choices,
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
            return ragflow_pb2.ChatCompletionsResponse(
                status=False, message=str(e), choices=[]
            )

    async def Embeddings(
        self,
        request: ragflow_pb2.EmbeddingsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.EmbeddingsResponse:
        """Generate embeddings for text."""
        try:
            # Extract input text(s)
            texts = []
            if request.text:
                texts = [request.text]
            elif request.texts:
                texts = list(request.texts)
            else:
                return ragflow_pb2.EmbeddingsResponse(
                    status=False, message="No input text provided", data=[]
                )

            # Build embedding request
            embedding_params = {
                "input": texts,
                "model": request.model or "ragflow-embedding",
                "encoding_format": request.encoding_format or "float",
                "user": request.user if request.user else None,
            }

            result = await self.ragflow_client.create_embeddings(embedding_params)

            embedding_data = []
            model_name = ""
            usage = None

            if result["status"] and result.get("data"):
                response_data = result["data"]
                model_name = response_data.get(
                    "model", request.model or "ragflow-embedding"
                )

                # Build embedding data
                embeddings_list = response_data.get("data", [])
                for idx, embedding_info in enumerate(embeddings_list):
                    embedding_vector = embedding_info.get("embedding", [])

                    embedding_item = ragflow_pb2.EmbeddingData(
                        object="embedding", embedding=embedding_vector, index=idx
                    )
                    embedding_data.append(embedding_item)

                # Build usage info
                usage_data = response_data.get("usage", {})
                if usage_data:
                    usage = ragflow_pb2.EmbeddingsUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )

            return ragflow_pb2.EmbeddingsResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                object="list",
                data=embedding_data,
                model=model_name,
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return ragflow_pb2.EmbeddingsResponse(status=False, message=str(e), data=[])

    async def Models(
        self,
        request: ragflow_pb2.ModelsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.ModelsResponse:
        """List available models."""
        try:
            result = await self.ragflow_client.list_models()

            models_data = []
            if result["status"] and result.get("data"):
                models_list = result["data"].get("data", [])

                for model_info in models_list:
                    model_item = ragflow_pb2.ModelInfo(
                        id=model_info.get("id", ""),
                        object="model",
                        created=model_info.get("created", 0),
                        owned_by=model_info.get("owned_by", "ragflow"),
                        permission=model_info.get("permission", []),
                        root=model_info.get("root", model_info.get("id", "")),
                        parent=(
                            model_info.get("parent")
                            if model_info.get("parent")
                            else None
                        ),
                    )
                    models_data.append(model_item)

            return ragflow_pb2.ModelsResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                object="list",
                data=models_data,
            )
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return ragflow_pb2.ModelsResponse(status=False, message=str(e), data=[])

    # Retrieval/Search Methods
    async def SearchDocuments(
        self,
        request: ragflow_pb2.SearchDocumentsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.SearchDocumentsResponse:
        """Search documents using semantic search."""
        try:
            result = await self.ragflow_client.search_documents(
                dataset_id=request.dataset_id,
                query=request.query,
                top_k=request.top_k,
                similarity_threshold=(
                    request.similarity_threshold
                    if request.similarity_threshold != 0.0
                    else 0.7
                ),
                filter_criteria=request.filter or None,
                include_content=(
                    request.include_content
                    if request.include_content
                    else False
                ),
            )

            search_results = []
            total = 0
            if result["status"] and result.get("data"):
                search_data = result["data"]
                total = search_data.get("total", 0)
                results_list = search_data.get("results", [])

                for result_info in results_list:
                    search_result = ragflow_pb2.SearchResult(
                        id=result_info.get("id", ""),
                        type=result_info.get("type", "document"),
                        title=result_info.get("title", ""),
                        content=result_info.get("content") or "",
                        similarity_score=result_info.get("similarity_score", 0.0),
                        metadata=result_info.get("metadata") or "",
                        dataset_id=result_info.get("dataset_id", ""),
                        document_id=result_info.get("document_id") or "",
                    )
                    search_results.append(search_result)

            return ragflow_pb2.SearchDocumentsResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                results=search_results,
                total=total,
            )
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return ragflow_pb2.SearchDocumentsResponse(
                status=False, message=str(e), results=[], total=0
            )

    async def RetrieveChunks(
        self,
        request: ragflow_pb2.RetrieveChunksRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.RetrieveChunksResponse:
        """Retrieve relevant chunks for RAG."""
        try:
            result = await self.ragflow_client.retrieve_chunks(
                dataset_id=request.dataset_id,
                query=request.query,
                top_k=request.top_k,
                similarity_threshold=(
                    request.similarity_threshold
                    if request.similarity_threshold != 0.0
                    else 0.2
                ),
                document_id=request.document_id or None,
                rerank=request.rerank,
            )

            chunks = []
            query_embedding = ""
            if result["status"] and result.get("data"):
                chunk_data = result["data"]
                chunks_list = chunk_data.get("chunks", [])
                query_embedding = chunk_data.get("query_embedding", "")

                for chunk_info in chunks_list:
                    chunk_result = ragflow_pb2.SearchResult(
                        id=chunk_info.get("id", ""),
                        type="chunk",
                        title=chunk_info.get("title", ""),
                        content=chunk_info.get("content", ""),
                        similarity_score=chunk_info.get("similarity_score", 0.0),
                        metadata=chunk_info.get("metadata") or "",
                        dataset_id=chunk_info.get("dataset_id", ""),
                        document_id=chunk_info.get("document_id", ""),
                    )
                    chunks.append(chunk_result)

            return ragflow_pb2.RetrieveChunksResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                chunks=chunks,
                query_embedding=query_embedding,
            )
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return ragflow_pb2.RetrieveChunksResponse(
                status=False, message=str(e), chunks=[], query_embedding=""
            )

    async def SimilaritySearch(
        self,
        request: ragflow_pb2.SimilaritySearchRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.SimilaritySearchResponse:
        """Perform similarity search using embeddings."""
        try:
            result = await self.ragflow_client.similarity_search(
                dataset_id=request.dataset_id,
                text=request.text or None,
                embedding=request.embedding or None,
                top_k=request.top_k,
                similarity_threshold=(
                    request.similarity_threshold
                    if request.similarity_threshold != 0.0
                    else 0.5
                ),
                content_type=request.content_type or "both",
            )

            search_results = []
            if result["status"] and result.get("data"):
                results_list = result["data"].get("results", [])

                for result_info in results_list:
                    search_result = ragflow_pb2.SearchResult(
                        id=result_info.get("id", ""),
                        type=result_info.get("type", "document"),
                        title=result_info.get("title", ""),
                        content=result_info.get("content") or "",
                        similarity_score=result_info.get("similarity_score", 0.0),
                        metadata=result_info.get("metadata") or "",
                        dataset_id=result_info.get("dataset_id", ""),
                        document_id=result_info.get("document_id") or "",
                    )
                    search_results.append(search_result)

            return ragflow_pb2.SimilaritySearchResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                results=search_results,
            )
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return ragflow_pb2.SimilaritySearchResponse(
                status=False, message=str(e), results=[]
            )

    # Chat Methods
    async def Chat(
        self, request: ragflow_pb2.ChatRequest, context: grpc.aio.ServicerContext
    ) -> ragflow_pb2.ChatResponse:
        """Chat with knowledge base."""
        try:
            result = await self.ragflow_client.chat(request.kb_id, request.question)
            answer = ""
            if result["status"] and result.get("data"):
                answer = result["data"].get("data", {}).get("answer", "") or result[
                    "data"
                ].get("answer", "")

            return ragflow_pb2.ChatResponse(
                status=result["status"],
                message=(
                    "Success" if result["status"] else result.get("error", "Failed")
                ),
                answer=answer,
            )
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return ragflow_pb2.ChatResponse(status=False, message=str(e))


async def serve() -> None:
    """Start the gRPC server."""
    server = grpc.aio.server(ThreadPoolExecutor(max_workers=10))
    ragflow_pb2_grpc.add_RagServicesServicer_to_server(RagServicesServicer(), server)

    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting server on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
