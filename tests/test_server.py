"""Tests for gRPC server with Dataset CRUD functionality."""

import pytest
from unittest.mock import AsyncMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from server import RagServicesServicer
import ragflow_pb2


class TestRagServicesServicer:
    """Test cases for RagServicesServicer - Dataset Operations."""

    @pytest.fixture
    def servicer(self):
        """Create servicer instance for testing."""
        with patch("ragflow_api.RAGFlowClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            servicer = RagServicesServicer()
            servicer.ragflow_client = mock_instance
            return servicer

    # Existing tests
    @pytest.mark.asyncio
    async def test_create_knowledge_base_success(self, servicer):
        """Test successful knowledge base creation."""
        servicer.ragflow_client.create_knowledge_base.return_value = {
            "status": True,
            "data": {"id": "test_kb_id"},
        }

        request = ragflow_pb2.CreateKnowledgeBaseRequest(
            name="Test KB", description="Test Description"
        )

        response = await servicer.CreateKnowledgeBase(request, None)

        assert response.status is True
        assert response.message == "Success"
        assert response.kb_id == "test_kb_id"

    @pytest.mark.asyncio
    async def test_upload_document_success(self, servicer):
        """Test successful document upload."""
        servicer.ragflow_client.upload_document.return_value = {"status": True}

        request = ragflow_pb2.UploadDocumentRequest(
            kb_id="test_kb", file_data=b"test content", filename="test.txt"
        )

        response = await servicer.UploadDocument(request, None)

        assert response.status is True
        assert response.message == "Success"

    @pytest.mark.asyncio
    async def test_chat_success(self, servicer):
        """Test successful chat."""
        servicer.ragflow_client.chat.return_value = {
            "status": True,
            "data": {"answer": "Test answer"},
        }

        request = ragflow_pb2.ChatRequest(kb_id="test_kb", question="Test question?")
        response = await servicer.Chat(request, None)

        assert response.status is True
        assert response.message == "Success"
        assert response.answer == "Test answer"

    # New Dataset CRUD tests
    @pytest.mark.asyncio
    async def test_list_datasets_success(self, servicer):
        """Test successful dataset listing."""
        mock_datasets = [
            {
                "id": "dataset_1",
                "name": "Test Dataset 1",
                "description": "First test dataset",
                "avatar": "",
                "embedding_model": "test-model",
                "permission": "me",
                "chunk_method": "naive",
                "chunk_count": 10,
                "document_count": 2,
                "create_date": "2024-01-01",
                "update_date": "2024-01-01",
            },
            {
                "id": "dataset_2",
                "name": "Test Dataset 2",
                "description": "Second test dataset",
                "avatar": "",
                "embedding_model": "test-model",
                "permission": "team",
                "chunk_method": "book",
                "chunk_count": 5,
                "document_count": 1,
                "create_date": "2024-01-02",
                "update_date": "2024-01-02",
            },
        ]

        servicer.ragflow_client.list_datasets.return_value = {
            "status": True,
            "data": mock_datasets,
        }

        request = ragflow_pb2.ListDatasetsRequest(
            page=1, page_size=10, orderby="create_time", desc=True
        )

        response = await servicer.ListDatasets(request, None)

        assert response.status is True
        assert response.message == "Success"
        assert len(response.datasets) == 2
        assert response.datasets[0].id == "dataset_1"
        assert response.datasets[0].name == "Test Dataset 1"
        assert response.datasets[0].chunk_count == 10
        assert response.datasets[1].id == "dataset_2"
        assert response.datasets[1].permission == "team"

    @pytest.mark.asyncio
    async def test_list_datasets_with_filters(self, servicer):
        """Test dataset listing with name and ID filters."""
        servicer.ragflow_client.list_datasets.return_value = {
            "status": True,
            "data": [],
        }

        request = ragflow_pb2.ListDatasetsRequest(
            page=1, page_size=5, name="Python", id="specific_id"
        )

        response = await servicer.ListDatasets(request, None)

        # Verify the client was called with correct parameters
        servicer.ragflow_client.list_datasets.assert_called_once_with(
            page=1,
            page_size=5,
            orderby="create_time",
            desc=False,
            name="Python",
            dataset_id="specific_id",
        )
        assert response.status is True

    @pytest.mark.asyncio
    async def test_list_datasets_error(self, servicer):
        """Test dataset listing with error."""
        servicer.ragflow_client.list_datasets.return_value = {
            "status": False,
            "error": "Connection failed",
        }

        request = ragflow_pb2.ListDatasetsRequest()
        response = await servicer.ListDatasets(request, None)

        assert response.status is False
        assert response.message == "Connection failed"
        assert len(response.datasets) == 0

    @pytest.mark.asyncio
    async def test_update_dataset_success(self, servicer):
        """Test successful dataset update."""
        servicer.ragflow_client.update_dataset.return_value = {"status": True}

        request = ragflow_pb2.UpdateDatasetRequest(
            dataset_id="test_dataset",
            name="Updated Name",
            description="Updated Description",
            permission="team",
        )

        response = await servicer.UpdateDataset(request, None)

        # Verify the client was called with correct update data
        servicer.ragflow_client.update_dataset.assert_called_once_with(
            "test_dataset",
            {
                "name": "Updated Name",
                "description": "Updated Description",
                "permission": "team",
            },
        )
        assert response.status is True
        assert response.message == "Success"

    @pytest.mark.asyncio
    async def test_update_dataset_partial(self, servicer):
        """Test partial dataset update with only some fields."""
        servicer.ragflow_client.update_dataset.return_value = {"status": True}

        request = ragflow_pb2.UpdateDatasetRequest(
            dataset_id="test_dataset",
            name="New Name Only",
            # Only name field provided
        )

        response = await servicer.UpdateDataset(request, None)

        # Verify only name was included in update data
        servicer.ragflow_client.update_dataset.assert_called_once_with(
            "test_dataset", {"name": "New Name Only"}
        )
        assert response.status is True

    @pytest.mark.asyncio
    async def test_update_dataset_error(self, servicer):
        """Test dataset update with error."""
        servicer.ragflow_client.update_dataset.return_value = {
            "status": False,
            "error": "Dataset not found",
        }

        request = ragflow_pb2.UpdateDatasetRequest(
            dataset_id="nonexistent_dataset", name="New Name"
        )

        response = await servicer.UpdateDataset(request, None)

        assert response.status is False
        assert response.message == "Dataset not found"

    @pytest.mark.asyncio
    async def test_delete_datasets_specific_ids(self, servicer):
        """Test deleting specific datasets by IDs."""
        servicer.ragflow_client.delete_datasets.return_value = {"status": True}

        request = ragflow_pb2.DeleteDatasetsRequest()
        request.ids.extend(["dataset_1", "dataset_2", "dataset_3"])

        response = await servicer.DeleteDatasets(request, None)

        # Verify the client was called with correct ID list
        servicer.ragflow_client.delete_datasets.assert_called_once_with(
            ["dataset_1", "dataset_2", "dataset_3"]
        )
        assert response.status is True
        assert response.message == "Success"

    @pytest.mark.asyncio
    async def test_delete_datasets_all(self, servicer):
        """Test deleting all datasets (empty IDs list)."""
        servicer.ragflow_client.delete_datasets.return_value = {"status": True}

        request = ragflow_pb2.DeleteDatasetsRequest()
        # No IDs provided = delete all

        response = await servicer.DeleteDatasets(request, None)

        # Verify the client was called with None/empty list
        servicer.ragflow_client.delete_datasets.assert_called_once_with([])
        assert response.status is True

    @pytest.mark.asyncio
    async def test_delete_datasets_error(self, servicer):
        """Test dataset deletion with error."""
        servicer.ragflow_client.delete_datasets.return_value = {
            "status": False,
            "error": "Permission denied",
        }

        request = ragflow_pb2.DeleteDatasetsRequest()
        request.ids.extend(["protected_dataset"])

        response = await servicer.DeleteDatasets(request, None)

        assert response.status is False
        assert response.message == "Permission denied"

    # Integration test
    @pytest.mark.asyncio
    async def test_dataset_crud_workflow(self, servicer):
        """Test complete dataset CRUD workflow."""
        # 1. Create dataset
        servicer.ragflow_client.create_knowledge_base.return_value = {
            "status": True,
            "data": {"id": "workflow_kb"},
        }

        create_request = ragflow_pb2.CreateKnowledgeBaseRequest(
            name="Workflow Test", description="Testing CRUD workflow"
        )
        create_response = await servicer.CreateKnowledgeBase(create_request, None)
        assert create_response.status is True
        assert create_response.kb_id == "workflow_kb"

        # 2. List datasets
        servicer.ragflow_client.list_datasets.return_value = {
            "status": True,
            "data": [
                {
                    "id": "workflow_kb",
                    "name": "Workflow Test",
                    "description": "Testing CRUD workflow",
                    "avatar": "",
                    "embedding_model": "test-model",
                    "permission": "me",
                    "chunk_method": "naive",
                    "chunk_count": 0,
                    "document_count": 0,
                    "create_date": "2024-01-01",
                    "update_date": "2024-01-01",
                }
            ],
        }

        list_request = ragflow_pb2.ListDatasetsRequest()
        list_response = await servicer.ListDatasets(list_request, None)
        assert list_response.status is True
        assert len(list_response.datasets) == 1
        assert list_response.datasets[0].id == "workflow_kb"

        # 3. Update dataset
        servicer.ragflow_client.update_dataset.return_value = {"status": True}

        update_request = ragflow_pb2.UpdateDatasetRequest(
            dataset_id="workflow_kb", description="Updated workflow description"
        )
        update_response = await servicer.UpdateDataset(update_request, None)
        assert update_response.status is True

        # 4. Delete dataset
        servicer.ragflow_client.delete_datasets.return_value = {"status": True}

        delete_request = ragflow_pb2.DeleteDatasetsRequest()
        delete_request.ids.extend(["workflow_kb"])
        delete_response = await servicer.DeleteDatasets(delete_request, None)
        assert delete_response.status is True


def test_grpc_communication():
    """Test that gRPC server can be imported and initialized."""
    servicer = RagServicesServicer()
    assert servicer is not None
