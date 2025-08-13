"""Async gRPC client for RAGFlow."""

import asyncio
import logging
import os
import sys
from typing import Optional, List, Dict
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

import grpc
import ragflow_pb2
import ragflow_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RagFlowGRPCClient:
    """Async gRPC client for RAGFlow services."""

    def __init__(self, server_address: str = "localhost:50051") -> None:
        self.server_address = server_address
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[ragflow_pb2_grpc.RagServicesStub] = None

    async def connect(self) -> None:
        """Connect to gRPC server."""
        self.channel = grpc.aio.insecure_channel(self.server_address)
        self.stub = ragflow_pb2_grpc.RagServicesStub(self.channel)
        logger.info(f"Connected to gRPC server at {self.server_address}")

    async def disconnect(self) -> None:
        """Disconnect from gRPC server."""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from gRPC server")

    # Dataset/Knowledge Base methods
    async def create_knowledge_base(
        self, name: str, description: str = ""
    ) -> ragflow_pb2.CreateKnowledgeBaseResponse:
        """Create a knowledge base."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.CreateKnowledgeBaseRequest(
            name=name, description=description
        )
        response = await self.stub.CreateKnowledgeBase(request)
        return response

    async def list_datasets(
        self,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> ragflow_pb2.ListDatasetsResponse:
        """List datasets with pagination and filtering."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.ListDatasetsRequest(
            page=page, page_size=page_size, orderby=orderby, desc=desc
        )
        if name:
            request.name = name
        if dataset_id:
            request.id = dataset_id

        response = await self.stub.ListDatasets(request)
        return response

    async def update_dataset(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
        permission: Optional[str] = None,
        chunk_method: Optional[str] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update dataset configuration."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.UpdateDatasetRequest(dataset_id=dataset_id)
        if name:
            request.name = name
        if description:
            request.description = description
        if embedding_model:
            request.embedding_model = embedding_model
        if permission:
            request.permission = permission
        if chunk_method:
            request.chunk_method = chunk_method

        response = await self.stub.UpdateDataset(request)
        return response

    async def delete_datasets(
        self, dataset_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete datasets by IDs. If None, deletes all datasets."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.DeleteDatasetsRequest()
        if dataset_ids:
            request.ids.extend(dataset_ids)

        response = await self.stub.DeleteDatasets(request)
        return response

    # Document Management methods
    async def upload_document(
        self, kb_id: str, file_data: bytes, filename: str
    ) -> ragflow_pb2.StatusResponse:
        """Upload document to knowledge base."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.UploadDocumentRequest(
            kb_id=kb_id, file_data=file_data, filename=filename
        )
        response = await self.stub.UploadDocument(request)
        return response

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
    ) -> ragflow_pb2.ListDocumentsResponse:
        """List documents in a dataset."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.ListDocumentsRequest(
            dataset_id=dataset_id,
            page=page,
            page_size=page_size,
            orderby=orderby,
            desc=desc,
        )
        if keywords:
            request.keywords = keywords
        if document_id:
            request.id = document_id
        if name:
            request.name = name

        response = await self.stub.ListDocuments(request)
        return response

    async def update_document(
        self,
        dataset_id: str,
        document_id: str,
        name: Optional[str] = None,
        chunk_method: Optional[str] = None,
        parser_config: Optional[dict] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update document configuration."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.UpdateDocumentRequest(
            dataset_id=dataset_id, document_id=document_id
        )
        if name:
            request.name = name
        if chunk_method:
            request.chunk_method = chunk_method
        if parser_config:
            request.parser_config = json.dumps(parser_config)

        response = await self.stub.UpdateDocument(request)
        return response

    async def download_document(
        self, dataset_id: str, document_id: str
    ) -> ragflow_pb2.DownloadDocumentResponse:
        """Download document file."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.DownloadDocumentRequest(
            dataset_id=dataset_id, document_id=document_id
        )
        response = await self.stub.DownloadDocument(request)
        return response

    async def delete_documents(
        self, dataset_id: str, document_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete documents from dataset."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.DeleteDocumentsRequest(dataset_id=dataset_id)
        if document_ids:
            request.ids.extend(document_ids)

        response = await self.stub.DeleteDocuments(request)
        return response

    async def parse_documents(
        self, dataset_id: str, document_ids: List[str]
    ) -> ragflow_pb2.StatusResponse:
        """Start parsing documents into chunks."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.ParseDocumentsRequest(dataset_id=dataset_id)
        request.document_ids.extend(document_ids)

        response = await self.stub.ParseDocuments(request)
        return response

    # Chat Assistant Management methods
    async def create_chat_assistant(
        self,
        name: str,
        description: str = "",
        avatar: str = "",
        dataset_ids: Optional[List[str]] = None,
        llm_model: str = "default",
        temperature: float = 0.1,
        top_p: float = 0.3,
        presence_penalty: float = 0.4,
        frequency_penalty: float = 0.7,
        prompt: str = "You are a helpful assistant.",
        similarity_threshold: float = 0.2,
        keywords_similarity_weight: float = 0.7,
        top_n: int = 6,
    ) -> ragflow_pb2.CreateChatAssistantResponse:
        """Create a chat assistant."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.CreateChatAssistantRequest(
            name=name,
            description=description,
            avatar=avatar,
            llm_model=llm_model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            prompt=prompt,
            similarity_threshold=similarity_threshold,
            keywords_similarity_weight=keywords_similarity_weight,
            top_n=top_n,
        )
        if dataset_ids:
            request.dataset_ids.extend(dataset_ids)

        response = await self.stub.CreateChatAssistant(request)
        return response

    async def list_chat_assistants(
        self,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        name: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> ragflow_pb2.ListChatAssistantsResponse:
        """List chat assistants with pagination and filtering."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.ListChatAssistantsRequest(
            page=page, page_size=page_size, orderby=orderby, desc=desc
        )
        if name:
            request.name = name
        if chat_id:
            request.id = chat_id

        response = await self.stub.ListChatAssistants(request)
        return response

    async def update_chat_assistant(
        self,
        chat_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        avatar: Optional[str] = None,
        dataset_ids: Optional[List[str]] = None,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        prompt: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        keywords_similarity_weight: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update chat assistant configuration."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.UpdateChatAssistantRequest(chat_id=chat_id)

        if name:
            request.name = name
        if description:
            request.description = description
        if avatar:
            request.avatar = avatar
        if dataset_ids:
            request.dataset_ids.extend(dataset_ids)
        if llm_model:
            request.llm_model = llm_model
        if temperature is not None:
            request.temperature = temperature
        if top_p is not None:
            request.top_p = top_p
        if presence_penalty is not None:
            request.presence_penalty = presence_penalty
        if frequency_penalty is not None:
            request.frequency_penalty = frequency_penalty
        if prompt:
            request.prompt = prompt
        if similarity_threshold is not None:
            request.similarity_threshold = similarity_threshold
        if keywords_similarity_weight is not None:
            request.keywords_similarity_weight = keywords_similarity_weight
        if top_n is not None:
            request.top_n = top_n

        response = await self.stub.UpdateChatAssistant(request)
        return response

    async def delete_chat_assistants(
        self, assistant_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete chat assistants by IDs. If None, deletes all assistants."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.DeleteChatAssistantsRequest()
        if assistant_ids:
            request.ids.extend(assistant_ids)

        response = await self.stub.DeleteChatAssistants(request)
        return response

    # Session Management methods
    async def create_session(
        self, chat_id: str, name: str, user_id: Optional[str] = None
    ) -> ragflow_pb2.CreateSessionResponse:
        """Create a chat session."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.CreateSessionRequest(chat_id=chat_id, name=name)
        if user_id:
            request.user_id = user_id

        response = await self.stub.CreateSession(request)
        return response

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
    ) -> ragflow_pb2.ListSessionsResponse:
        """List chat sessions with pagination and filtering."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.ListSessionsRequest(
            chat_id=chat_id,
            page=page,
            page_size=page_size,
            orderby=orderby,
            desc=desc,
        )
        if name:
            request.name = name
        if session_id:
            request.id = session_id
        if user_id:
            request.user_id = user_id

        response = await self.stub.ListSessions(request)
        return response

    async def update_session(
        self,
        chat_id: str,
        session_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update session configuration."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.UpdateSessionRequest(
            chat_id=chat_id, session_id=session_id
        )
        if name:
            request.name = name
        if user_id:
            request.user_id = user_id

        response = await self.stub.UpdateSession(request)
        return response

    async def delete_sessions(
        self, chat_id: str, session_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete sessions from chat assistant."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.DeleteSessionsRequest(chat_id=chat_id)
        if session_ids:
            request.ids.extend(session_ids)

        response = await self.stub.DeleteSessions(request)
        return response

    # Chunk Management methods
    async def create_chunk(
        self,
        dataset_id: str,
        document_id: str,
        content: str,
        metadata: Optional[str] = None,
        position: Optional[int] = None,
    ) -> ragflow_pb2.CreateChunkResponse:
        """Create a document chunk."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.CreateChunkRequest(
            dataset_id=dataset_id,
            document_id=document_id,
            content=content,
        )
        if metadata:
            request.metadata = metadata
        if position is not None:
            request.position = position

        response = await self.stub.CreateChunk(request)
        return response

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
    ) -> ragflow_pb2.ListChunksResponse:
        """List chunks in a dataset."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.ListChunksRequest(
            dataset_id=dataset_id,
            page=page,
            page_size=page_size,
            orderby=orderby,
            desc=desc,
        )
        if document_id:
            request.document_id = document_id
        if keywords:
            request.keywords = keywords
        if chunk_id:
            request.id = chunk_id

        response = await self.stub.ListChunks(request)
        return response

    async def update_chunk(
        self,
        dataset_id: str,
        chunk_id: str,
        content: Optional[str] = None,
        metadata: Optional[str] = None,
        position: Optional[int] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update chunk configuration."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.UpdateChunkRequest(
            dataset_id=dataset_id, chunk_id=chunk_id
        )
        if content:
            request.content = content
        if metadata:
            request.metadata = metadata
        if position is not None:
            request.position = position

        response = await self.stub.UpdateChunk(request)
        return response

    async def delete_chunks(
        self, dataset_id: str, chunk_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete chunks from dataset."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.DeleteChunksRequest(dataset_id=dataset_id)
        if chunk_ids:
            request.ids.extend(chunk_ids)

        response = await self.stub.DeleteChunks(request)
        return response

    # Retrieval/Search methods
    async def search_documents(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filter_criteria: Optional[str] = None,
        include_content: bool = False,
    ) -> ragflow_pb2.SearchDocumentsResponse:
        """Search documents using semantic search."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.SearchDocumentsRequest(
            dataset_id=dataset_id,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            include_content=include_content,
        )
        if filter_criteria:
            request.filter = filter_criteria

        response = await self.stub.SearchDocuments(request)
        return response

    async def retrieve_chunks(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.2,
        document_id: Optional[str] = None,
        rerank: bool = True,
    ) -> ragflow_pb2.RetrieveChunksResponse:
        """Retrieve relevant chunks for RAG."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.RetrieveChunksRequest(
            dataset_id=dataset_id,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            rerank=rerank,
        )
        if document_id:
            request.document_id = document_id

        response = await self.stub.RetrieveChunks(request)
        return response

    async def similarity_search(
        self,
        dataset_id: str,
        text: Optional[str] = None,
        embedding: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        content_type: str = "both",
    ) -> ragflow_pb2.SimilaritySearchResponse:
        """Perform similarity search using embeddings."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        if not text and not embedding:
            raise ValueError("Either text or embedding must be provided")

        request = ragflow_pb2.SimilaritySearchRequest(
            dataset_id=dataset_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            content_type=content_type,
        )
        if text:
            request.text = text
        if embedding:
            request.embedding = embedding

        response = await self.stub.SimilaritySearch(request)
        return response

    # Retrieval/Search methods
    def search_documents(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filter_criteria: Optional[str] = None,
        include_content: bool = False,
    ) -> ragflow_pb2.SearchDocumentsResponse:
        """Search documents (sync)."""
        return asyncio.run(
            self._run_async_method(
                "search_documents",
                dataset_id,
                query,
                top_k,
                similarity_threshold,
                filter_criteria,
                include_content,
            )
        )

    def retrieve_chunks(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.2,
        document_id: Optional[str] = None,
        rerank: bool = True,
    ) -> ragflow_pb2.RetrieveChunksResponse:
        """Retrieve chunks (sync)."""
        return asyncio.run(
            self._run_async_method(
                "retrieve_chunks",
                dataset_id,
                query,
                top_k,
                similarity_threshold,
                document_id,
                rerank,
            )
        )

    def similarity_search(
        self,
        dataset_id: str,
        text: Optional[str] = None,
        embedding: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        content_type: str = "both",
    ) -> ragflow_pb2.SimilaritySearchResponse:
        """Similarity search (sync)."""
        return asyncio.run(
            self._run_async_method(
                "similarity_search",
                dataset_id,
                text,
                embedding,
                top_k,
                similarity_threshold,
                content_type,
            )
        )

    # Chat methods
    async def chat(self, kb_id: str, question: str) -> ragflow_pb2.ChatResponse:
        """Chat with knowledge base."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.ChatRequest(kb_id=kb_id, question=question)
        response = await self.stub.Chat(request)
        return response


class RagFlowSyncClient:
    """Synchronous wrapper for gRPC client."""

    def __init__(self, server_address: str = "localhost:50051") -> None:
        self.async_client = RagFlowGRPCClient(server_address)

    # Dataset/Knowledge Base methods
    def create_knowledge_base(
        self, name: str, description: str = ""
    ) -> ragflow_pb2.CreateKnowledgeBaseResponse:
        """Create knowledge base (sync)."""
        return asyncio.run(
            self._run_async_method("create_knowledge_base", name, description)
        )

    def list_datasets(
        self,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> ragflow_pb2.ListDatasetsResponse:
        """List datasets (sync)."""
        return asyncio.run(
            self._run_async_method(
                "list_datasets", page, page_size, orderby, desc, name, dataset_id
            )
        )

    def update_dataset(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
        permission: Optional[str] = None,
        chunk_method: Optional[str] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update dataset (sync)."""
        return asyncio.run(
            self._run_async_method(
                "update_dataset",
                dataset_id,
                name,
                description,
                embedding_model,
                permission,
                chunk_method,
            )
        )

    def delete_datasets(
        self, dataset_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete datasets (sync)."""
        return asyncio.run(self._run_async_method("delete_datasets", dataset_ids))

    # Document Management methods
    def upload_document(
        self, kb_id: str, file_data: bytes, filename: str
    ) -> ragflow_pb2.StatusResponse:
        """Upload document (sync)."""
        return asyncio.run(
            self._run_async_method("upload_document", kb_id, file_data, filename)
        )

    def list_documents(
        self,
        dataset_id: str,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        keywords: Optional[str] = None,
        document_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> ragflow_pb2.ListDocumentsResponse:
        """List documents (sync)."""
        return asyncio.run(
            self._run_async_method(
                "list_documents",
                dataset_id,
                page,
                page_size,
                orderby,
                desc,
                keywords,
                document_id,
                name,
            )
        )

    def update_document(
        self,
        dataset_id: str,
        document_id: str,
        name: Optional[str] = None,
        chunk_method: Optional[str] = None,
        parser_config: Optional[dict] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update document (sync)."""
        return asyncio.run(
            self._run_async_method(
                "update_document",
                dataset_id,
                document_id,
                name,
                chunk_method,
                parser_config,
            )
        )

    def download_document(
        self, dataset_id: str, document_id: str
    ) -> ragflow_pb2.DownloadDocumentResponse:
        """Download document (sync)."""
        return asyncio.run(
            self._run_async_method("download_document", dataset_id, document_id)
        )

    def delete_documents(
        self, dataset_id: str, document_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete documents (sync)."""
        return asyncio.run(
            self._run_async_method("delete_documents", dataset_id, document_ids)
        )

    def parse_documents(
        self, dataset_id: str, document_ids: List[str]
    ) -> ragflow_pb2.StatusResponse:
        """Parse documents (sync)."""
        return asyncio.run(
            self._run_async_method("parse_documents", dataset_id, document_ids)
        )

    # Chat Assistant Management methods
    def create_chat_assistant(
        self,
        name: str,
        description: str = "",
        avatar: str = "",
        dataset_ids: Optional[List[str]] = None,
        llm_model: str = "default",
        temperature: float = 0.1,
        top_p: float = 0.3,
        presence_penalty: float = 0.4,
        frequency_penalty: float = 0.7,
        prompt: str = "You are a helpful assistant.",
        similarity_threshold: float = 0.2,
        keywords_similarity_weight: float = 0.7,
        top_n: int = 6,
    ) -> ragflow_pb2.CreateChatAssistantResponse:
        """Create chat assistant (sync)."""
        return asyncio.run(
            self._run_async_method(
                "create_chat_assistant",
                name,
                description,
                avatar,
                dataset_ids,
                llm_model,
                temperature,
                top_p,
                presence_penalty,
                frequency_penalty,
                prompt,
                similarity_threshold,
                keywords_similarity_weight,
                top_n,
            )
        )

    def list_chat_assistants(
        self,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        name: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> ragflow_pb2.ListChatAssistantsResponse:
        """List chat assistants (sync)."""
        return asyncio.run(
            self._run_async_method(
                "list_chat_assistants", page, page_size, orderby, desc, name, chat_id
            )
        )

    def update_chat_assistant(
        self,
        chat_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        avatar: Optional[str] = None,
        dataset_ids: Optional[List[str]] = None,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        prompt: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        keywords_similarity_weight: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update chat assistant (sync)."""
        return asyncio.run(
            self._run_async_method(
                "update_chat_assistant",
                chat_id,
                name,
                description,
                avatar,
                dataset_ids,
                llm_model,
                temperature,
                top_p,
                presence_penalty,
                frequency_penalty,
                prompt,
                similarity_threshold,
                keywords_similarity_weight,
                top_n,
            )
        )

    def delete_chat_assistants(
        self, assistant_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete chat assistants (sync)."""
        return asyncio.run(
            self._run_async_method("delete_chat_assistants", assistant_ids)
        )

    # Session Management methods
    def create_session(
        self, chat_id: str, name: str, user_id: Optional[str] = None
    ) -> ragflow_pb2.CreateSessionResponse:
        """Create session (sync)."""
        return asyncio.run(
            self._run_async_method("create_session", chat_id, name, user_id)
        )

    def list_sessions(
        self,
        chat_id: str,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ragflow_pb2.ListSessionsResponse:
        """List sessions (sync)."""
        return asyncio.run(
            self._run_async_method(
                "list_sessions",
                chat_id,
                page,
                page_size,
                orderby,
                desc,
                name,
                session_id,
                user_id,
            )
        )

    def update_session(
        self,
        chat_id: str,
        session_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update session (sync)."""
        return asyncio.run(
            self._run_async_method("update_session", chat_id, session_id, name, user_id)
        )

    def delete_sessions(
        self, chat_id: str, session_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete sessions (sync)."""
        return asyncio.run(
            self._run_async_method("delete_sessions", chat_id, session_ids)
        )

    # Chunk Management methods
    def create_chunk(
        self,
        dataset_id: str,
        document_id: str,
        content: str,
        metadata: Optional[str] = None,
        position: Optional[int] = None,
    ) -> ragflow_pb2.CreateChunkResponse:
        """Create chunk (sync)."""
        return asyncio.run(
            self._run_async_method(
                "create_chunk", dataset_id, document_id, content, metadata, position
            )
        )

    def list_chunks(
        self,
        dataset_id: str,
        document_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        keywords: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> ragflow_pb2.ListChunksResponse:
        """List chunks (sync)."""
        return asyncio.run(
            self._run_async_method(
                "list_chunks",
                dataset_id,
                document_id,
                page,
                page_size,
                orderby,
                desc,
                keywords,
                chunk_id,
            )
        )

    def update_chunk(
        self,
        dataset_id: str,
        chunk_id: str,
        content: Optional[str] = None,
        metadata: Optional[str] = None,
        position: Optional[int] = None,
    ) -> ragflow_pb2.StatusResponse:
        """Update chunk (sync)."""
        return asyncio.run(
            self._run_async_method(
                "update_chunk", dataset_id, chunk_id, content, metadata, position
            )
        )

    def delete_chunks(
        self, dataset_id: str, chunk_ids: Optional[List[str]] = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete chunks (sync)."""
        return asyncio.run(
            self._run_async_method("delete_chunks", dataset_id, chunk_ids)
        )

    # Chat methods
    def chat(self, kb_id: str, question: str) -> ragflow_pb2.ChatResponse:
        """Chat (sync)."""
        return asyncio.run(self._run_async_method("chat", kb_id, question))

    async def _run_async_method(self, method_name: str, *args) -> any:
        """Run async method with connection management."""
        await self.async_client.connect()
        try:
            method = getattr(self.async_client, method_name)
            return await method(*args)
        finally:
            await self.async_client.disconnect()
