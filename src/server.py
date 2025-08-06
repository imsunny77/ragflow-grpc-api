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
            api_token=os.getenv("RAGFLOW_API_TOKEN", "demo_token")
        )
        self.ragflow_client = RAGFlowClient(config)
    
    async def CreateKnowledgeBase(
        self, request: ragflow_pb2.CreateKnowledgeBaseRequest, context: grpc.aio.ServicerContext
    ) -> ragflow_pb2.CreateKnowledgeBaseResponse:
        """Create knowledge base."""
        try:
            result = await self.ragflow_client.create_knowledge_base(
                request.name, request.description or ""
            )
            kb_id = result.get("data", {}).get("id", "") if result["status"] else ""
            return ragflow_pb2.CreateKnowledgeBaseResponse(
                status=result["status"],
                message="Success" if result["status"] else "Failed",
                kb_id=kb_id
            )
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            return ragflow_pb2.CreateKnowledgeBaseResponse(
                status=False, message=str(e)
            )
    
    async def UploadDocument(
        self, request: ragflow_pb2.UploadDocumentRequest, context: grpc.aio.ServicerContext
    ) -> ragflow_pb2.StatusResponse:
        """Upload document."""
        try:
            result = await self.ragflow_client.upload_document(
                request.kb_id, request.file_data, request.filename
            )
            return ragflow_pb2.StatusResponse(
                status=result["status"],
                message="Success" if result["status"] else "Failed"
            )
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return ragflow_pb2.StatusResponse(status=False, message=str(e))
    
    async def Chat(
        self, request: ragflow_pb2.ChatRequest, context: grpc.aio.ServicerContext
    ) -> ragflow_pb2.ChatResponse:
        """Chat with knowledge base."""
        try:
            result = await self.ragflow_client.chat(request.kb_id, request.question)
            answer = result.get("data", {}).get("answer", "") if result["status"] else ""
            return ragflow_pb2.ChatResponse(
                status=result["status"],
                message="Success" if result["status"] else "Failed",
                answer=answer
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