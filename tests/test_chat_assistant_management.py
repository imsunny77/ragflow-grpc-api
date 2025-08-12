"""Tests for Chat Assistant Management functionality."""

import pytest
from unittest.mock import AsyncMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from server import RagServicesServicer
import ragflow_pb2


class TestChatAssistantManagement:
    """Test cases for Chat Assistant Management operations."""

    @pytest.fixture
    def servicer(self):
        """Create servicer instance for testing."""
        with patch("ragflow_api.RAGFlowClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            servicer = RagServicesServicer()
            servicer.ragflow_client = mock_instance
            return servicer

    # Create Chat Assistant Tests
    @pytest.mark.asyncio
    async def test_create_chat_assistant_success(self, servicer):
        """Test successful chat assistant creation."""
        servicer.ragflow_client.create_chat_assistant.return_value = {
            "status": True,
            "data": {"id": "chat_assistant_123"},
        }

        request = ragflow_pb2.CreateChatAssistantRequest(
            name="Test Assistant",
            description="Test description",
            avatar="test_avatar",
            llm_model="gpt-4",
            temperature=0.2,
            top_p=0.5,
            presence_penalty=0.3,
            frequency_penalty=0.6,
            prompt="You are a test assistant.",
            similarity_threshold=0.25,
            keywords_similarity_weight=0.8,
            top_n=10,
        )
        request.dataset_ids.extend(["kb_1", "kb_2"])

        response = await servicer.CreateChatAssistant(request, None)

        # Verify the client was called with correct configuration
        expected_config = {
            "name": "Test Assistant",
            "description": "Test description",
            "avatar": "test_avatar",
            "dataset_ids": ["kb_1", "kb_2"],
            "llm": {
                "model_name": "gpt-4",
                "temperature": 0.2,
                "top_p": 0.5,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.6,
            },
            "prompt": {
                "prompt": "You are a test assistant.",
                "similarity_threshold": 0.25,
                "keywords_similarity_weight": 0.8,
                "top_n": 10,
            },
        }
        servicer.ragflow_client.create_chat_assistant.assert_called_once_with(
            expected_config
        )

        assert response.status is True
        assert response.message == "Success"
        assert response.chat_id == "chat_assistant_123"

    @pytest.mark.asyncio
    async def test_create_chat_assistant_with_defaults(self, servicer):
        """Test chat assistant creation with default values."""
        servicer.ragflow_client.create_chat_assistant.return_value = {
            "status": True,
            "data": {"id": "chat_default"},
        }

        request = ragflow_pb2.CreateChatAssistantRequest(name="Minimal Assistant")

        response = await servicer.CreateChatAssistant(request, None)

        # Verify default values were applied
        call_args = servicer.ragflow_client.create_chat_assistant.call_args[0][0]
        assert call_args["name"] == "Minimal Assistant"
        assert call_args["description"] == ""
        assert call_args["avatar"] == ""
        assert call_args["dataset_ids"] == []
        assert call_args["llm"]["model_name"] == "default"
        assert call_args["llm"]["temperature"] == 0.1
        assert call_args["llm"]["top_p"] == 0.3
        assert call_args["prompt"]["prompt"] == "You are a helpful assistant."
        assert call_args["prompt"]["similarity_threshold"] == 0.2

        assert response.status is True
        assert response.chat_id == "chat_default"

    @pytest.mark.asyncio
    async def test_create_chat_assistant_error(self, servicer):
        """Test chat assistant creation with error."""
        servicer.ragflow_client.create_chat_assistant.return_value = {
            "status": False,
            "error": "Invalid configuration",
        }

        request = ragflow_pb2.CreateChatAssistantRequest(name="Error Assistant")
        response = await servicer.CreateChatAssistant(request, None)

        assert response.status is False
        assert response.message == "Invalid configuration"
        assert not response.HasField("chat_id")

    # List Chat Assistants Tests
    @pytest.mark.asyncio
    async def test_list_chat_assistants_success(self, servicer):
        """Test successful chat assistants listing."""
        mock_assistants = [
            {
                "id": "chat_1",
                "name": "Python Assistant",
                "description": "Python expert",
                "avatar": "python_avatar",
                "dataset_ids": ["kb_1"],
                "llm": {
                    "model_name": "gpt-4",
                    "temperature": 0.1,
                    "top_p": 0.3,
                    "presence_penalty": 0.4,
                    "frequency_penalty": 0.7,
                },
                "prompt": {
                    "prompt": "You are a Python expert.",
                    "similarity_threshold": 0.2,
                    "keywords_similarity_weight": 0.7,
                    "top_n": 6,
                },
                "create_date": "2024-01-01",
                "update_date": "2024-01-01",
            },
            {
                "id": "chat_2",
                "name": "AI Assistant",
                "description": "AI expert",
                "avatar": "",
                "dataset_ids": ["kb_1", "kb_2"],
                "llm": {
                    "model_name": "claude-3",
                    "temperature": 0.3,
                    "top_p": 0.6,
                    "presence_penalty": 0.2,
                    "frequency_penalty": 0.5,
                },
                "prompt": {
                    "prompt": "You are an AI expert.",
                    "similarity_threshold": 0.15,
                    "keywords_similarity_weight": 0.8,
                    "top_n": 10,
                },
                "create_date": "2024-01-02",
                "update_date": "2024-01-02",
            },
        ]

        servicer.ragflow_client.list_chat_assistants.return_value = {
            "status": True,
            "data": mock_assistants,
        }

        request = ragflow_pb2.ListChatAssistantsRequest(
            page=1, page_size=10, orderby="create_time", desc=True
        )

        response = await servicer.ListChatAssistants(request, None)

        assert response.status is True
        assert response.message == "Success"
        assert len(response.assistants) == 2

        # Check first assistant
        assistant1 = response.assistants[0]
        assert assistant1.id == "chat_1"
        assert assistant1.name == "Python Assistant"
        assert assistant1.llm_model == "gpt-4"
        assert assistant1.temperature == 0.1
        assert assistant1.top_n == 6
        assert len(assistant1.dataset_ids) == 1
        assert assistant1.dataset_ids[0] == "kb_1"

        # Check second assistant
        assistant2 = response.assistants[1]
        assert assistant2.id == "chat_2"
        assert assistant2.name == "AI Assistant"
        assert assistant2.llm_model == "claude-3"
        assert assistant2.temperature == 0.3
        assert len(assistant2.dataset_ids) == 2

    @pytest.mark.asyncio
    async def test_list_chat_assistants_with_filters(self, servicer):
        """Test chat assistants listing with filters."""
        servicer.ragflow_client.list_chat_assistants.return_value = {
            "status": True,
            "data": [],
        }

        request = ragflow_pb2.ListChatAssistantsRequest(
            page=1, page_size=5, name="Python", id="specific_chat"
        )

        response = await servicer.ListChatAssistants(request, None)

        # Verify the client was called with correct parameters
        servicer.ragflow_client.list_chat_assistants.assert_called_once_with(
            page=1,
            page_size=5,
            orderby="create_time",
            desc=True,
            name="Python",
            chat_id="specific_chat",
        )
        assert response.status is True

    @pytest.mark.asyncio
    async def test_list_chat_assistants_error(self, servicer):
        """Test chat assistants listing with error."""
        servicer.ragflow_client.list_chat_assistants.return_value = {
            "status": False,
            "error": "Database connection failed",
        }

        request = ragflow_pb2.ListChatAssistantsRequest()
        response = await servicer.ListChatAssistants(request, None)

        assert response.status is False
        assert response.message == "Database connection failed"
        assert len(response.assistants) == 0

    # Update Chat Assistant Tests
    @pytest.mark.asyncio
    async def test_update_chat_assistant_success(self, servicer):
        """Test successful chat assistant update."""
        servicer.ragflow_client.update_chat_assistant.return_value = {"status": True}

        request = ragflow_pb2.UpdateChatAssistantRequest(
            chat_id="chat_123",
            name="Updated Assistant",
            description="Updated description",
            temperature=0.4,
            top_p=0.8,
            prompt="Updated prompt",
        )
        request.dataset_ids.extend(["kb_new"])

        response = await servicer.UpdateChatAssistant(request, None)

        # Verify the client was called with correct update data
        expected_update = {
            "name": "Updated Assistant",
            "description": "Updated description",
            "dataset_ids": ["kb_new"],
            "llm": {"temperature": 0.4, "top_p": 0.8},
            "prompt": {"prompt": "Updated prompt"},
        }
        servicer.ragflow_client.update_chat_assistant.assert_called_once_with(
            "chat_123", expected_update
        )
        assert response.status is True
        assert response.message == "Success"

    @pytest.mark.asyncio
    async def test_update_chat_assistant_partial(self, servicer):
        """Test partial chat assistant update."""
        servicer.ragflow_client.update_chat_assistant.return_value = {"status": True}

        request = ragflow_pb2.UpdateChatAssistantRequest(
            chat_id="chat_123",
            name="New Name Only",
            # Only name provided
        )

        response = await servicer.UpdateChatAssistant(request, None)

        # Verify only name was included in update data
        servicer.ragflow_client.update_chat_assistant.assert_called_once_with(
            "chat_123", {"name": "New Name Only"}
        )
        assert response.status is True

    @pytest.mark.asyncio
    async def test_update_chat_assistant_llm_only(self, servicer):
        """Test chat assistant update with only LLM parameters."""
        servicer.ragflow_client.update_chat_assistant.return_value = {"status": True}

        request = ragflow_pb2.UpdateChatAssistantRequest(
            chat_id="chat_123",
            llm_model="gpt-4o",
            temperature=0.15,
            presence_penalty=0.5,
        )

        response = await servicer.UpdateChatAssistant(request, None)

        # Verify only LLM config was included
        expected_update = {
            "llm": {
                "model_name": "gpt-4o",
                "temperature": 0.15,
                "presence_penalty": 0.5,
            }
        }
        servicer.ragflow_client.update_chat_assistant.assert_called_once_with(
            "chat_123", expected_update
        )
        assert response.status is True

    @pytest.mark.asyncio
    async def test_update_chat_assistant_prompt_only(self, servicer):
        """Test chat assistant update with only prompt parameters."""
        servicer.ragflow_client.update_chat_assistant.return_value = {"status": True}

        request = ragflow_pb2.UpdateChatAssistantRequest(
            chat_id="chat_123", similarity_threshold=0.35, top_n=12
        )

        response = await servicer.UpdateChatAssistant(request, None)

        # Verify only prompt config was included
        expected_update = {"prompt": {"similarity_threshold": 0.35, "top_n": 12}}
        servicer.ragflow_client.update_chat_assistant.assert_called_once_with(
            "chat_123", expected_update
        )
        assert response.status is True

    @pytest.mark.asyncio
    async def test_update_chat_assistant_error(self, servicer):
        """Test chat assistant update with error."""
        servicer.ragflow_client.update_chat_assistant.return_value = {
            "status": False,
            "error": "Assistant not found",
        }

        request = ragflow_pb2.UpdateChatAssistantRequest(
            chat_id="nonexistent_chat", name="New Name"
        )

        response = await servicer.UpdateChatAssistant(request, None)

        assert response.status is False
        assert response.message == "Assistant not found"

    # Delete Chat Assistants Tests
    @pytest.mark.asyncio
    async def test_delete_chat_assistants_specific_ids(self, servicer):
        """Test deleting specific chat assistants by IDs."""
        servicer.ragflow_client.delete_chat_assistants.return_value = {"status": True}

        request = ragflow_pb2.DeleteChatAssistantsRequest()
        request.ids.extend(["chat_1", "chat_2", "chat_3"])

        response = await servicer.DeleteChatAssistants(request, None)

        servicer.ragflow_client.delete_chat_assistants.assert_called_once_with(
            ["chat_1", "chat_2", "chat_3"]
        )
        assert response.status is True
        assert response.message == "Success"

    @pytest.mark.asyncio
    async def test_delete_chat_assistants_all(self, servicer):
        """Test deleting all chat assistants."""
        servicer.ragflow_client.delete_chat_assistants.return_value = {"status": True}

        request = ragflow_pb2.DeleteChatAssistantsRequest()
        # No IDs provided = delete all

        response = await servicer.DeleteChatAssistants(request, None)

        servicer.ragflow_client.delete_chat_assistants.assert_called_once_with([])
        assert response.status is True

    @pytest.mark.asyncio
    async def test_delete_chat_assistants_error(self, servicer):
        """Test chat assistants deletion with error."""
        servicer.ragflow_client.delete_chat_assistants.return_value = {
            "status": False,
            "error": "Permission denied",
        }

        request = ragflow_pb2.DeleteChatAssistantsRequest()
        request.ids.extend(["protected_chat"])

        response = await servicer.DeleteChatAssistants(request, None)

        assert response.status is False
        assert response.message == "Permission denied"

    # Integration Workflow Test
    @pytest.mark.asyncio
    async def test_chat_assistant_management_workflow(self, servicer):
        """Test complete chat assistant management workflow."""
        # 1. Create chat assistant
        servicer.ragflow_client.create_chat_assistant.return_value = {
            "status": True,
            "data": {"id": "workflow_chat"},
        }

        create_request = ragflow_pb2.CreateChatAssistantRequest(
            name="Workflow Assistant", description="Testing workflow"
        )
        create_response = await servicer.CreateChatAssistant(create_request, None)
        assert create_response.status is True
        assert create_response.chat_id == "workflow_chat"

        # 2. List chat assistants
        servicer.ragflow_client.list_chat_assistants.return_value = {
            "status": True,
            "data": [
                {
                    "id": "workflow_chat",
                    "name": "Workflow Assistant",
                    "description": "Testing workflow",
                    "avatar": "",
                    "dataset_ids": [],
                    "llm": {
                        "model_name": "default",
                        "temperature": 0.1,
                        "top_p": 0.3,
                        "presence_penalty": 0.4,
                        "frequency_penalty": 0.7,
                    },
                    "prompt": {
                        "prompt": "You are a helpful assistant.",
                        "similarity_threshold": 0.2,
                        "keywords_similarity_weight": 0.7,
                        "top_n": 6,
                    },
                    "create_date": "2024-01-01",
                    "update_date": "2024-01-01",
                }
            ],
        }

        list_request = ragflow_pb2.ListChatAssistantsRequest()
        list_response = await servicer.ListChatAssistants(list_request, None)
        assert list_response.status is True
        assert len(list_response.assistants) == 1
        assert list_response.assistants[0].id == "workflow_chat"

        # 3. Update chat assistant
        servicer.ragflow_client.update_chat_assistant.return_value = {"status": True}

        update_request = ragflow_pb2.UpdateChatAssistantRequest(
            chat_id="workflow_chat",
            description="Updated workflow description",
            temperature=0.2,
        )
        update_response = await servicer.UpdateChatAssistant(update_request, None)
        assert update_response.status is True

        # 4. Delete chat assistant
        servicer.ragflow_client.delete_chat_assistants.return_value = {"status": True}

        delete_request = ragflow_pb2.DeleteChatAssistantsRequest()
        delete_request.ids.extend(["workflow_chat"])
        delete_response = await servicer.DeleteChatAssistants(delete_request, None)
        assert delete_response.status is True
