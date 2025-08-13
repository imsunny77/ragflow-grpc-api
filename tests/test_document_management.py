"""Tests for Document Management functionality."""

import pytest
from unittest.mock import AsyncMock, patch
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from server import RagServicesServicer
import ragflow_pb2


class TestDocumentManagement:
    """Test cases for Document Management operations."""

    @pytest.fixture
    def servicer(self):
        """Create servicer instance for testing."""
        with patch("ragflow_api.RAGFlowClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            servicer = RagServicesServicer()
            servicer.ragflow_client = mock_instance
            return servicer

    # List Documents Tests
    @pytest.mark.asyncio
    async def test_list_documents_success(self, servicer):
        """Test successful document listing."""
        mock_documents = {
            "data": {
                "docs": [
                    {
                        "id": "doc_1",
                        "name": "test_doc.txt",
                        "dataset_id": "kb_123",
                        "knowledgebase_id": "kb_123",
                        "size": 1024,
                        "type": "doc",
                        "chunk_method": "naive",
                        "chunk_count": 5,
                        "status": "1",
                        "create_date": "2024-01-01",
                        "update_date": "2024-01-01",
                        "thumbnail": "",
                    },
                    {
                        "id": "doc_2",
                        "name": "guide.pdf",
                        "dataset_id": "kb_123",
                        "size": 2048,
                        "type": "pdf",
                        "chunk_method": "book",
                        "chunk_count": 10,
                        "status": "1",
                        "create_date": "2024-01-02",
                        "update_date": "2024-01-02",
                        "thumbnail": None,
                    },
                ],
                "total": 2,
            }
        }

        servicer.ragflow_client.list_documents.return_value = {
            "status": True,
            "data": mock_documents["data"],
        }

        request = ragflow_pb2.ListDocumentsRequest(
            dataset_id="kb_123", page=1, page_size=10, orderby="create_time", desc=True
        )

        response = await servicer.ListDocuments(request, None)

        assert response.status is True
        assert response.message == "Success"
        assert len(response.documents) == 2
        assert response.total == 2

        # Check first document
        doc1 = response.documents[0]
        assert doc1.id == "doc_1"
        assert doc1.name == "test_doc.txt"
        assert doc1.dataset_id == "kb_123"
        assert doc1.size == 1024
        assert doc1.chunk_count == 5

        # Check second document
        doc2 = response.documents[1]
        assert doc2.id == "doc_2"
        assert doc2.name == "guide.pdf"
        assert doc2.chunk_method == "book"

    @pytest.mark.asyncio
    async def test_list_documents_with_filters(self, servicer):
        """Test document listing with filters."""
        servicer.ragflow_client.list_documents.return_value = {
            "status": True,
            "data": {"docs": [], "total": 0},
        }

        request = ragflow_pb2.ListDocumentsRequest(
            dataset_id="kb_123",
            page=1,
            page_size=5,
            keywords="python",
            id="doc_specific",
            name="test.txt",
        )

        response = await servicer.ListDocuments(request, None)

        # Verify the client was called with correct parameters
        servicer.ragflow_client.list_documents.assert_called_once_with(
            dataset_id="kb_123",
            page=1,
            page_size=5,
            orderby="create_time",
            desc=False,
            keywords="python",
            document_id="doc_specific",
            name="test.txt",
        )
        assert response.status is True

    @pytest.mark.asyncio
    async def test_list_documents_error(self, servicer):
        """Test document listing with error."""
        servicer.ragflow_client.list_documents.return_value = {
            "status": False,
            "error": "Dataset not found",
        }

        request = ragflow_pb2.ListDocumentsRequest(dataset_id="invalid_kb")
        response = await servicer.ListDocuments(request, None)

        assert response.status is False
        assert response.message == "Dataset not found"
        assert len(response.documents) == 0
        assert response.total == 0

    # Update Document Tests
    @pytest.mark.asyncio
    async def test_update_document_success(self, servicer):
        """Test successful document update."""
        servicer.ragflow_client.update_document.return_value = {"status": True}

        request = ragflow_pb2.UpdateDocumentRequest(
            dataset_id="kb_123",
            document_id="doc_456",
            name="updated_name.txt",
            chunk_method="book",
            parser_config='{"chunk_token_num": 256}',
        )

        response = await servicer.UpdateDocument(request, None)

        # Verify the client was called with correct update data
        expected_update_data = {
            "name": "updated_name.txt",
            "chunk_method": "book",
            "parser_config": {"chunk_token_num": 256},
        }
        servicer.ragflow_client.update_document.assert_called_once_with(
            "kb_123", "doc_456", expected_update_data
        )
        assert response.status is True
        assert response.message == "Success"

    @pytest.mark.asyncio
    async def test_update_document_partial(self, servicer):
        """Test partial document update."""
        servicer.ragflow_client.update_document.return_value = {"status": True}

        request = ragflow_pb2.UpdateDocumentRequest(
            dataset_id="kb_123",
            document_id="doc_456",
            name="new_name_only.txt",
            # Only name provided
        )

        response = await servicer.UpdateDocument(request, None)

        # Verify only name was included in update data
        servicer.ragflow_client.update_document.assert_called_once_with(
            "kb_123", "doc_456", {"name": "new_name_only.txt"}
        )
        assert response.status is True

    @pytest.mark.asyncio
    async def test_update_document_invalid_json(self, servicer):
        """Test document update with invalid JSON parser_config."""
        request = ragflow_pb2.UpdateDocumentRequest(
            dataset_id="kb_123",
            document_id="doc_456",
            parser_config='{"invalid": json}',  # Invalid JSON
        )

        response = await servicer.UpdateDocument(request, None)

        assert response.status is False
        assert "Invalid parser_config JSON" in response.message
        # Client should not be called due to JSON error
        servicer.ragflow_client.update_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_error(self, servicer):
        """Test document update with error."""
        servicer.ragflow_client.update_document.return_value = {
            "status": False,
            "error": "Document not found",
        }

        request = ragflow_pb2.UpdateDocumentRequest(
            dataset_id="kb_123", document_id="nonexistent_doc", name="new_name.txt"
        )

        response = await servicer.UpdateDocument(request, None)

        assert response.status is False
        assert response.message == "Document not found"

    # Download Document Tests
    @pytest.mark.asyncio
    async def test_download_document_success(self, servicer):
        """Test successful document download."""
        mock_file_data = b"This is test file content for download."
        servicer.ragflow_client.download_document.return_value = {
            "status": True,
            "data": mock_file_data,
            "filename": "test_download.txt",
        }

        request = ragflow_pb2.DownloadDocumentRequest(
            dataset_id="kb_123", document_id="doc_789"
        )

        response = await servicer.DownloadDocument(request, None)

        servicer.ragflow_client.download_document.assert_called_once_with(
            "kb_123", "doc_789"
        )
        assert response.status is True
        assert response.message == "Success"
        assert response.file_data == mock_file_data
        assert response.filename == "test_download.txt"

    @pytest.mark.asyncio
    async def test_download_document_error(self, servicer):
        """Test document download with error."""
        servicer.ragflow_client.download_document.return_value = {
            "status": False,
            "error": "File not found",
        }

        request = ragflow_pb2.DownloadDocumentRequest(
            dataset_id="kb_123", document_id="missing_doc"
        )

        response = await servicer.DownloadDocument(request, None)

        assert response.status is False
        assert response.message == "File not found"
        assert response.file_data == b""
        assert not response.HasField("filename")

    # Delete Documents Tests
    @pytest.mark.asyncio
    async def test_delete_documents_specific_ids(self, servicer):
        """Test deleting specific documents by IDs."""
        servicer.ragflow_client.delete_documents.return_value = {"status": True}

        request = ragflow_pb2.DeleteDocumentsRequest(dataset_id="kb_123")
        request.ids.extend(["doc_1", "doc_2", "doc_3"])

        response = await servicer.DeleteDocuments(request, None)

        servicer.ragflow_client.delete_documents.assert_called_once_with(
            "kb_123", ["doc_1", "doc_2", "doc_3"]
        )
        assert response.status is True
        assert response.message == "Success"

    @pytest.mark.asyncio
    async def test_delete_documents_all(self, servicer):
        """Test deleting all documents in dataset."""
        servicer.ragflow_client.delete_documents.return_value = {"status": True}

        request = ragflow_pb2.DeleteDocumentsRequest(dataset_id="kb_123")
        # No IDs provided = delete all

        response = await servicer.DeleteDocuments(request, None)

        servicer.ragflow_client.delete_documents.assert_called_once_with("kb_123", [])
        assert response.status is True

    @pytest.mark.asyncio
    async def test_delete_documents_error(self, servicer):
        """Test document deletion with error."""
        servicer.ragflow_client.delete_documents.return_value = {
            "status": False,
            "error": "Permission denied",
        }

        request = ragflow_pb2.DeleteDocumentsRequest(dataset_id="kb_123")
        request.ids.extend(["protected_doc"])

        response = await servicer.DeleteDocuments(request, None)

        assert response.status is False
        assert response.message == "Permission denied"

    # Parse Documents Tests
    @pytest.mark.asyncio
    async def test_parse_documents_success(self, servicer):
        """Test successful document parsing."""
        servicer.ragflow_client.parse_documents.return_value = {"status": True}

        request = ragflow_pb2.ParseDocumentsRequest(dataset_id="kb_123")
        request.document_ids.extend(["doc_1", "doc_2"])

        response = await servicer.ParseDocuments(request, None)

        servicer.ragflow_client.parse_documents.assert_called_once_with(
            "kb_123", ["doc_1", "doc_2"]
        )
        assert response.status is True
        assert response.message == "Success"

    @pytest.mark.asyncio
    async def test_parse_documents_error(self, servicer):
        """Test document parsing with error."""
        servicer.ragflow_client.parse_documents.return_value = {
            "status": False,
            "error": "Parsing failed",
        }

        request = ragflow_pb2.ParseDocumentsRequest(dataset_id="kb_123")
        request.document_ids.extend(["invalid_doc"])

        response = await servicer.ParseDocuments(request, None)

        assert response.status is False
        assert response.message == "Parsing failed"

    # Integration Workflow Test
    @pytest.mark.asyncio
    async def test_document_management_workflow(self, servicer):
        """Test complete document management workflow."""
        # 1. Upload document (already tested in existing tests)
        servicer.ragflow_client.upload_document.return_value = {"status": True}

        upload_request = ragflow_pb2.UploadDocumentRequest(
            kb_id="workflow_kb", file_data=b"Test content", filename="workflow_test.txt"
        )
        upload_response = await servicer.UploadDocument(upload_request, None)
        assert upload_response.status is True

        # 2. List documents
        servicer.ragflow_client.list_documents.return_value = {
            "status": True,
            "data": {
                "docs": [
                    {
                        "id": "workflow_doc",
                        "name": "workflow_test.txt",
                        "dataset_id": "workflow_kb",
                        "size": 12,
                        "type": "doc",
                        "chunk_method": "naive",
                        "chunk_count": 0,
                        "status": "1",
                        "create_date": "2024-01-01",
                        "update_date": "2024-01-01",
                        "thumbnail": None,
                    }
                ],
                "total": 1,
            },
        }

        list_request = ragflow_pb2.ListDocumentsRequest(dataset_id="workflow_kb")
        list_response = await servicer.ListDocuments(list_request, None)
        assert list_response.status is True
        assert len(list_response.documents) == 1
        assert list_response.documents[0].id == "workflow_doc"

        # 3. Update document
        servicer.ragflow_client.update_document.return_value = {"status": True}

        update_request = ragflow_pb2.UpdateDocumentRequest(
            dataset_id="workflow_kb",
            document_id="workflow_doc",
            name="updated_workflow.txt",
        )
        update_response = await servicer.UpdateDocument(update_request, None)
        assert update_response.status is True

        # 4. Parse document
        servicer.ragflow_client.parse_documents.return_value = {"status": True}

        parse_request = ragflow_pb2.ParseDocumentsRequest(dataset_id="workflow_kb")
        parse_request.document_ids.extend(["workflow_doc"])
        parse_response = await servicer.ParseDocuments(parse_request, None)
        assert parse_response.status is True

        # 5. Download document
        servicer.ragflow_client.download_document.return_value = {
            "status": True,
            "data": b"Test content",
            "filename": "updated_workflow.txt",
        }

        download_request = ragflow_pb2.DownloadDocumentRequest(
            dataset_id="workflow_kb", document_id="workflow_doc"
        )
        download_response = await servicer.DownloadDocument(download_request, None)
        assert download_response.status is True
        assert download_response.file_data == b"Test content"

        # 6. Delete document
        servicer.ragflow_client.delete_documents.return_value = {"status": True}

        delete_request = ragflow_pb2.DeleteDocumentsRequest(dataset_id="workflow_kb")
        delete_request.ids.extend(["workflow_doc"])
        delete_response = await servicer.DeleteDocuments(delete_request, None)
        assert delete_response.status is True
