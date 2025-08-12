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
                desc=request.desc if request.HasField("desc") else True,
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
                desc=request.desc if request.HasField("desc") else True,
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
                        request.temperature if request.HasField("temperature") else 0.1
                    ),
                    "top_p": request.top_p if request.HasField("top_p") else 0.3,
                    "presence_penalty": (
                        request.presence_penalty
                        if request.HasField("presence_penalty")
                        else 0.4
                    ),
                    "frequency_penalty": (
                        request.frequency_penalty
                        if request.HasField("frequency_penalty")
                        else 0.7
                    ),
                },
                "prompt": {
                    "prompt": request.prompt or "You are a helpful assistant.",
                    "similarity_threshold": (
                        request.similarity_threshold
                        if request.HasField("similarity_threshold")
                        else 0.2
                    ),
                    "keywords_similarity_weight": (
                        request.keywords_similarity_weight
                        if request.HasField("keywords_similarity_weight")
                        else 0.7
                    ),
                    "top_n": request.top_n if request.HasField("top_n") else 6,
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
                desc=request.desc if request.HasField("desc") else True,
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
            if request.HasField("temperature"):
                llm_config["temperature"] = request.temperature
            if request.HasField("top_p"):
                llm_config["top_p"] = request.top_p
            if request.HasField("presence_penalty"):
                llm_config["presence_penalty"] = request.presence_penalty
            if request.HasField("frequency_penalty"):
                llm_config["frequency_penalty"] = request.frequency_penalty

            if llm_config:
                update_data["llm"] = llm_config

            # Prompt configuration
            prompt_config = {}
            if request.prompt:
                prompt_config["prompt"] = request.prompt
            if request.HasField("similarity_threshold"):
                prompt_config["similarity_threshold"] = request.similarity_threshold
            if request.HasField("keywords_similarity_weight"):
                prompt_config["keywords_similarity_weight"] = (
                    request.keywords_similarity_weight
                )
            if request.HasField("top_n"):
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
