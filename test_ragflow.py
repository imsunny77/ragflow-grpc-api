"""Test suite for RAGFlow gRPC API."""

import asyncio
import pytest
from src.ragflow_api import RAGFlowClient, RAGFlowConfig


async def test_ragflow() -> None:
    """Test RAGFlow client functionality."""
    config = RAGFlowConfig()
    client = RAGFlowClient(config)

    # Test dataset creation
    result = await client.create_dataset("test", "test description")
    assert result["status"] is True

    # Test document upload
    result = await client.upload_document("test_id", "test.txt")
    assert result["status"] is True

    # Test chat
    result = await client.chat("test_id", "Hello")
    assert result["status"] is True

    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_ragflow())
