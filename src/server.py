"""Async gRPC server for RAGFlow."""

import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

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
                # Parse JSON string to dict
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
