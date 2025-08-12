"""Async gRPC client for RAGFlow."""

import asyncio
import logging
import os
import sys

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
        self.channel: grpc.aio.Channel | None = None
        self.stub: ragflow_pb2_grpc.RagServicesStub | None = None

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
        name: str | None = None,
        dataset_id: str | None = None,
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
        name: str | None = None,
        description: str | None = None,
        embedding_model: str | None = None,
        permission: str | None = None,
        chunk_method: str | None = None,
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
        self, dataset_ids: list[str] | None = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete datasets by IDs. If None, deletes all datasets."""
        if not self.stub:
            raise RuntimeError("Client not connected")

        request = ragflow_pb2.DeleteDatasetsRequest()
        if dataset_ids:
            request.ids.extend(dataset_ids)

        response = await self.stub.DeleteDatasets(request)
        return response

    # Document methods
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
        name: str | None = None,
        dataset_id: str | None = None,
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
        name: str | None = None,
        description: str | None = None,
        embedding_model: str | None = None,
        permission: str | None = None,
        chunk_method: str | None = None,
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
        self, dataset_ids: list[str] | None = None
    ) -> ragflow_pb2.StatusResponse:
        """Delete datasets (sync)."""
        return asyncio.run(self._run_async_method("delete_datasets", dataset_ids))

    # Document methods
    def upload_document(
        self, kb_id: str, file_data: bytes, filename: str
    ) -> ragflow_pb2.StatusResponse:
        """Upload document (sync)."""
        return asyncio.run(
            self._run_async_method("upload_document", kb_id, file_data, filename)
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
