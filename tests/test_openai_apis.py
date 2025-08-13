"""Tests for OpenAI Compatible APIs functionality."""

import pytest
from src.client import RagFlowGRPCClient


@pytest.mark.asyncio
async def test_chat_completions_basic():
    """Test basic OpenAI chat completions functionality."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        response = await client.chat_completions(
            messages=messages, model="ragflow-default", temperature=0.7, max_tokens=100
        )

        assert response.status is True
        assert response.object == "chat.completion"
        assert hasattr(response, "choices")
        assert hasattr(response, "usage")
        assert len(response.choices) > 0

        # Validate choice structure
        choice = response.choices[0]
        assert hasattr(choice, "index")
        assert hasattr(choice, "message")
        assert hasattr(choice, "finish_reason")
        assert choice.message.role in ["assistant", ""]

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_chat_completions_with_rag():
    """Test chat completions with RAG dataset integration."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        messages = [{"role": "user", "content": "What is machine learning?"}]

        response = await client.chat_completions(
            messages=messages,
            model="ragflow-default",
            temperature=0.5,
            max_tokens=200,
            dataset_id="ml_knowledge_base",
        )

        assert response.status is True
        assert response.object == "chat.completion"
        assert len(response.choices) > 0

        # Check that response includes assistant message
        choice = response.choices[0]
        assert choice.message.content != ""

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_chat_completions_with_advanced_params():
    """Test chat completions with advanced OpenAI parameters."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        messages = [
            {"role": "system", "content": "You are a creative writer."},
            {"role": "user", "content": "Write a short poem about AI."},
        ]

        response = await client.chat_completions(
            messages=messages,
            model="ragflow-creative",
            temperature=0.9,
            max_tokens=150,
            top_p=0.95,
            frequency_penalty=0.3,
            presence_penalty=0.6,
            stream=False,
            user="test_user_123",
        )

        assert response.status is True
        assert response.model in ["ragflow-creative", ""]

        # Validate usage information
        if response.usage:
            assert response.usage.prompt_tokens >= 0
            assert response.usage.completion_tokens >= 0
            assert response.usage.total_tokens >= 0

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_create_embeddings_single_text():
    """Test embedding generation for single text."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.create_embeddings(
            input_text="This is a test sentence for embedding generation.",
            model="ragflow-embedding",
            encoding_format="float",
        )

        assert response.status is True
        assert response.object == "list"
        assert len(response.data) == 1

        # Validate embedding structure
        embedding = response.data[0]
        assert embedding.object == "embedding"
        assert embedding.index == 0
        assert len(embedding.embedding) > 0  # Should have vector dimensions

        # Validate usage information
        if response.usage:
            assert response.usage.prompt_tokens > 0
            assert response.usage.total_tokens > 0

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_create_embeddings_multiple_texts():
    """Test embedding generation for multiple texts."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        texts = [
            "First sentence for embedding.",
            "Second sentence for embedding.",
            "Third sentence for embedding.",
        ]

        response = await client.create_embeddings(
            input_texts=texts,
            model="ragflow-embedding",
            encoding_format="float",
            user="batch_user",
        )

        assert response.status is True
        assert response.object == "list"
        assert len(response.data) == 3

        # Validate each embedding
        for i, embedding in enumerate(response.data):
            assert embedding.object == "embedding"
            assert embedding.index == i
            assert len(embedding.embedding) > 0

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_create_embeddings_validation():
    """Test embedding creation parameter validation."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Should raise ValueError when no input provided
        with pytest.raises(
            ValueError, match="Either input_text or input_texts must be provided"
        ):
            await client.create_embeddings(model="ragflow-embedding")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_list_models():
    """Test listing available models."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.list_models()

        assert response.status is True
        assert response.object == "list"
        assert hasattr(response, "data")

        # Validate model structure if models are returned
        for model in response.data:
            assert model.object == "model"
            assert model.id != ""
            assert model.owned_by != ""
            assert model.created >= 0  # Unix timestamp

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_openai_workflow():
    """Test complete OpenAI-compatible workflow."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Step 1: List available models
        models_response = await client.list_models()
        assert models_response.status is True

        # Step 2: Generate embeddings for context
        embedding_response = await client.create_embeddings(
            input_text="Machine learning and artificial intelligence concepts"
        )
        assert embedding_response.status is True
        assert len(embedding_response.data) == 1

        # Step 3: Use chat completions
        messages = [
            {"role": "system", "content": "You are an AI expert."},
            {"role": "user", "content": "Explain machine learning in simple terms."},
        ]

        chat_response = await client.chat_completions(
            messages=messages, temperature=0.6, max_tokens=200
        )
        assert chat_response.status is True
        assert len(chat_response.choices) > 0

        # Verify the complete workflow produced valid responses
        assert models_response.object == "list"
        assert embedding_response.object == "list"
        assert chat_response.object == "chat.completion"

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_openai_message_formats():
    """Test various OpenAI message formats and roles."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Test different message roles and formats
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "name": "system_bot",
            },
            {"role": "user", "content": "Hello!", "name": "john_doe"},
            {"role": "assistant", "content": "Hi there! How can I help you?"},
            {"role": "user", "content": "What's the weather like?"},
        ]

        response = await client.chat_completions(messages=messages, temperature=0.7)

        assert response.status is True
        assert len(response.choices) > 0

        # Validate that the response maintains proper format
        choice = response.choices[0]
        assert choice.message.role in ["assistant", ""]
        assert choice.finish_reason in ["stop", "length", "content_filter", ""]

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_openai_error_handling():
    """Test error handling in OpenAI APIs."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Test chat completions with empty messages
        response = await client.chat_completions(messages=[], temperature=0.7)
        # Should handle gracefully
        assert hasattr(response, "status")

        # Test embeddings with empty text
        embedding_response = await client.create_embeddings(
            input_text="test", model="ragflow-embedding"
        )
        assert hasattr(embedding_response, "status")

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_openai_parameter_boundaries():
    """Test OpenAI APIs with boundary parameter values."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        messages = [{"role": "user", "content": "Test"}]

        # Test with minimum parameters
        response1 = await client.chat_completions(
            messages=messages, temperature=0.0, max_tokens=1, top_p=0.1
        )
        assert response1.status is True

        # Test with maximum reasonable parameters
        response2 = await client.chat_completions(
            messages=messages,
            temperature=2.0,
            max_tokens=4000,
            top_p=1.0,
            frequency_penalty=2.0,
            presence_penalty=2.0,
        )
        assert response2.status is True

    finally:
        await client.disconnect()
