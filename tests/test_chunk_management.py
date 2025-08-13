"""Tests for Chunk Management functionality."""

import pytest
from client import RagFlowGRPCClient


@pytest.mark.asyncio
async def test_create_chunk():
    """Test chunk creation."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.create_chunk(
            dataset_id="test_dataset_id",
            document_id="test_document_id",
            content="This is a test chunk content.",
            metadata='{"type": "test", "importance": "high"}',
            position=1,
        )
        assert response.status is True
        assert hasattr(response, "chunk_id")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_list_chunks():
    """Test chunk listing."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.list_chunks(
            dataset_id="test_dataset_id",
            document_id="test_document_id",
            page=1,
            page_size=10,
        )
        assert response.status is True
        assert hasattr(response, "chunks")
        assert hasattr(response, "total")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_update_chunk():
    """Test chunk update."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.update_chunk(
            dataset_id="test_dataset_id",
            chunk_id="test_chunk_id",
            content="Updated chunk content.",
            metadata='{"type": "updated", "importance": "medium"}',
            position=2,
        )
        assert response.status is True
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_delete_chunks():
    """Test chunk deletion."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.delete_chunks(
            dataset_id="test_dataset_id", chunk_ids=["chunk1", "chunk2"]
        )
        assert response.status is True
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_chunk_workflow():
    """Test complete chunk workflow."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Create chunk
        create_response = await client.create_chunk(
            dataset_id="workflow_dataset",
            document_id="workflow_document",
            content="Workflow test chunk content.",
            metadata='{"workflow": "test"}',
        )
        assert create_response.status is True

        # List chunks
        list_response = await client.list_chunks(
            dataset_id="workflow_dataset", document_id="workflow_document"
        )
        assert list_response.status is True

        # Update chunk
        update_response = await client.update_chunk(
            dataset_id="workflow_dataset",
            chunk_id="test_chunk",
            content="Updated workflow test content.",
        )
        assert update_response.status is True

        # Delete chunk
        delete_response = await client.delete_chunks(
            dataset_id="workflow_dataset", chunk_ids=["test_chunk"]
        )
        assert delete_response.status is True

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_list_chunks_with_filters():
    """Test chunk listing with various filters."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Test with keywords filter
        response = await client.list_chunks(
            dataset_id="test_dataset",
            keywords="test keyword",
            page=1,
            page_size=5,
            orderby="create_time",
            desc=True,
        )
        assert response.status is True

        # Test with specific chunk ID filter
        response = await client.list_chunks(
            dataset_id="test_dataset", chunk_id="specific_chunk_id"
        )
        assert response.status is True

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_chunk_metadata_handling():
    """Test chunk creation and update with metadata."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Create chunk with complex metadata
        metadata = '{"category": "technical", "tags": ["ai", "ml"], "priority": 9}'
        response = await client.create_chunk(
            dataset_id="metadata_test_dataset",
            document_id="metadata_test_document",
            content="Content with rich metadata.",
            metadata=metadata,
            position=5,
        )
        assert response.status is True

        # Update chunk metadata
        updated_metadata = (
            '{"category": "technical", "tags": ["ai", "ml", "nlp"], "priority": 10}'
        )
        update_response = await client.update_chunk(
            dataset_id="metadata_test_dataset",
            chunk_id="test_chunk",
            metadata=updated_metadata,
        )
        assert update_response.status is True

    finally:
        await client.disconnect()
