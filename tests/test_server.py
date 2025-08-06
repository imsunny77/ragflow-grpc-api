"""Tests for gRPC server."""
import pytest
from unittest.mock import AsyncMock, patch

from src.server import RagServicesServicer
from src import ragflow_pb2


class TestRagServicesServicer:
    """Test cases for RagServicesServicer."""
    
    @pytest.fixture
    def servicer(self):
        """Create servicer instance for testing."""
        with patch('src.server.RAGFlowClient') as mock_client:
            mock_client.return_value = AsyncMock()
            return RagServicesServicer()
    
    @pytest.mark.asyncio
    async def test_create_knowledge_base_success(self, servicer):
        """Test successful knowledge base creation."""
        # Mock the ragflow client response
        servicer.ragflow_client.create_knowledge_base.return_value = {
            "status": True,
            "data": {"id": "test_kb_id"}
        }
        
        request = ragflow_pb2.CreateKnowledgeBaseRequest(
            name="Test KB",
            description="Test Description"
        )
        
        response = await servicer.CreateKnowledgeBase(request, None)
        
        assert response.status is True
        assert response.message == "Success"
        assert response.kb_id == "test_kb_id"
    
    @pytest.mark.asyncio
    async def test_create_knowledge_base_failure(self, servicer):
        """Test failed knowledge base creation."""
        servicer.ragflow_client.create_knowledge_base.return_value = {
            "status": False,
            "data": {}
        }
        
        request = ragflow_pb2.CreateKnowledgeBaseRequest(name="Test KB")
        response = await servicer.CreateKnowledgeBase(request, None)
        
        assert response.status is False
        assert response.message == "Failed"
    
    @pytest.mark.asyncio
    async def test_upload_document_success(self, servicer):
        """Test successful document upload."""
        servicer.ragflow_client.upload_document.return_value = {"status": True}
        
        request = ragflow_pb2.UploadDocumentRequest(
            kb_id="test_kb",
            file_data=b"test content",
            filename="test.txt"
        )
        
        response = await servicer.UploadDocument(request, None)
        
        assert response.status is True
        assert response.message == "Success"
    
    @pytest.mark.asyncio
    async def test_chat_success(self, servicer):
        """Test successful chat."""
        servicer.ragflow_client.chat.return_value = {
            "status": True,
            "data": {"answer": "Test answer"}
        }
        
        request = ragflow_pb2.ChatRequest(kb_id="test_kb", question="Test question?")
        response = await servicer.Chat(request, None)
        
        assert response.status is True
        assert response.message == "Success"
        assert response.answer == "Test answer"