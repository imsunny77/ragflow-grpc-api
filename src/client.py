"""Async gRPC client for RAGFlow."""
import asyncio
import logging
import os
import sys
from typing import Optional

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
    
    async def create_knowledge_base(self, name: str, description: str = "") -> ragflow_pb2.CreateKnowledgeBaseResponse:
        """Create a knowledge base."""
        if not self.stub:
            raise RuntimeError("Client not connected")
        
        request = ragflow_pb2.CreateKnowledgeBaseRequest(name=name, description=description)
        response = await self.stub.CreateKnowledgeBase(request)
        return response
    
    async def upload_document(self, kb_id: str, file_data: bytes, filename: str) -> ragflow_pb2.StatusResponse:
        """Upload document to knowledge base."""
        if not self.stub:
            raise RuntimeError("Client not connected")
        
        request = ragflow_pb2.UploadDocumentRequest(
            kb_id=kb_id, file_data=file_data, filename=filename
        )
        response = await self.stub.UploadDocument(request)
        return response
    
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
    
    def create_knowledge_base(self, name: str, description: str = "") -> ragflow_pb2.CreateKnowledgeBaseResponse:
        """Create knowledge base (sync)."""
        return asyncio.run(self._run_async_method("create_knowledge_base", name, description))
    
    def upload_document(self, kb_id: str, file_data: bytes, filename: str) -> ragflow_pb2.StatusResponse:
        """Upload document (sync)."""
        return asyncio.run(self._run_async_method("upload_document", kb_id, file_data, filename))
    
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