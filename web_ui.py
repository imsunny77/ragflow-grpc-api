# Fix for server.py - rename methods to lowercase as per ruff N802
import asyncio

import grpc
from grpc import aio

import ragflow_pb2
import ragflow_pb2_grpc
from src.ragflow_api import RAGFlowClient, RAGFlowConfig


class RagServicesServicer(ragflow_pb2_grpc.RagServicesServicer):
    def __init__(self, config: RAGFlowConfig) -> None:
        self.ragflow_client = RAGFlowClient(config)

    async def create_knowledge_base(
        self,
        request: ragflow_pb2.CreateKnowledgeBaseRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.CreateKnowledgeBaseResponse:
        try:
            result = await self.ragflow_client.create_knowledge_base(
                request.name, request.description
            )
            if result.get("status"):
                return ragflow_pb2.CreateKnowledgeBaseResponse(
                    status=True,
                    message="Knowledge base created successfully",
                    knowledge_base_id=result.get("id", ""),
                )
            return ragflow_pb2.CreateKnowledgeBaseResponse(
                status=False, message=result.get("message", "Failed to create")
            )
        except Exception as e:
            return ragflow_pb2.CreateKnowledgeBaseResponse(status=False, message=str(e))

    async def upload_document(
        self,
        request: ragflow_pb2.UploadDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> ragflow_pb2.StatusResponse:
        try:
            result = await self.ragflow_client.upload_document(
                request.knowledge_base_id, request.file_path
            )
            return ragflow_pb2.StatusResponse(
                status=result.get("status", False),
                message=result.get("message", "Upload completed"),
            )
        except Exception as e:
            return ragflow_pb2.StatusResponse(status=False, message=str(e))

    async def chat(
        self, request: ragflow_pb2.ChatRequest, context: grpc.aio.ServicerContext
    ) -> ragflow_pb2.ChatResponse:
        try:
            result = await self.ragflow_client.chat(
                request.knowledge_base_id, request.question
            )
            if result.get("status"):
                return ragflow_pb2.ChatResponse(
                    status=True,
                    message="Chat successful",
                    answer=result.get("answer", ""),
                )
            return ragflow_pb2.ChatResponse(
                status=False, message=result.get("message", "Chat failed"), answer=""
            )
        except Exception as e:
            return ragflow_pb2.ChatResponse(status=False, message=str(e), answer="")


async def serve() -> None:
    config = RAGFlowConfig()
    server = aio.server()
    ragflow_pb2_grpc.add_RagServicesServicer_to_server(
        RagServicesServicer(config), server
    )
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    await server.start()
    print(f"Server started on {listen_addr}")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
