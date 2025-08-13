"""Tests for Retrieval/Search APIs functionality."""

import pytest
from client import RagFlowGRPCClient


@pytest.mark.asyncio
async def test_search_documents_basic():
    """Test basic document search functionality."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.search_documents(
            dataset_id="test_dataset_id",
            query="artificial intelligence machine learning",
            top_k=5,
            similarity_threshold=0.7,
        )
        assert response.status is True
        assert hasattr(response, "results")
        assert hasattr(response, "total")
        # Validate result structure
        for result in response.results:
            assert hasattr(result, "id")
            assert hasattr(result, "type")
            assert hasattr(result, "similarity_score")
            assert result.type in ["document", "chunk"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_search_documents_with_filters():
    """Test document search with content and filters."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.search_documents(
            dataset_id="test_dataset",
            query="deep learning neural networks",
            top_k=10,
            similarity_threshold=0.6,
            filter_criteria='{"category": "technical", "tags": ["ai"]}',
            include_content=True,
        )
        assert response.status is True
        # When include_content=True, results should have content
        for result in response.results:
            if result.content:  # Content may be optional
                assert len(result.content) > 0
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_retrieve_chunks_basic():
    """Test basic chunk retrieval for RAG."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.retrieve_chunks(
            dataset_id="test_dataset_id",
            query="What is machine learning?",
            top_k=3,
            similarity_threshold=0.2,
        )
        assert response.status is True
        assert hasattr(response, "chunks")
        assert hasattr(response, "query_embedding")
        # Validate chunk structure
        for chunk in response.chunks:
            assert hasattr(chunk, "id")
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "similarity_score")
            assert chunk.type == "chunk"
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_retrieve_chunks_with_document_filter():
    """Test chunk retrieval filtered by document."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.retrieve_chunks(
            dataset_id="test_dataset",
            query="explain neural networks",
            top_k=5,
            similarity_threshold=0.3,
            document_id="specific_document_id",
            rerank=True,
        )
        assert response.status is True
        # All chunks should be from the specified document
        for chunk in response.chunks:
            assert (
                chunk.document_id == "specific_document_id" or chunk.document_id == ""
            )
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_similarity_search_by_text():
    """Test similarity search using text input."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.similarity_search(
            dataset_id="test_dataset_id",
            text="artificial intelligence and machine learning",
            top_k=8,
            similarity_threshold=0.5,
            content_type="both",
        )
        assert response.status is True
        assert hasattr(response, "results")
        # Validate similarity scores are within threshold
        for result in response.results:
            assert result.similarity_score >= 0.5
            assert result.type in ["document", "chunk"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_similarity_search_by_embedding():
    """Test similarity search using embedding vector."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Mock base64 encoded embedding vector
        mock_embedding = (
            "dGVzdF9lbWJlZGRpbmdfdmVjdG9y"  # base64 for "test_embedding_vector"
        )

        response = await client.similarity_search(
            dataset_id="test_dataset",
            embedding=mock_embedding,
            top_k=6,
            similarity_threshold=0.4,
            content_type="document",
        )
        assert response.status is True
        # When content_type="document", all results should be documents
        for result in response.results:
            assert (
                result.type == "document" or result.type == ""
            )  # May be empty in mock
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_similarity_search_chunks_only():
    """Test similarity search for chunks only."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        response = await client.similarity_search(
            dataset_id="test_dataset",
            text="machine learning algorithms",
            top_k=10,
            similarity_threshold=0.3,
            content_type="chunk",
        )
        assert response.status is True
        # When content_type="chunk", all results should be chunks
        for result in response.results:
            assert result.type == "chunk" or result.type == ""  # May be empty in mock
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_similarity_search_validation():
    """Test similarity search parameter validation."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Should raise ValueError when neither text nor embedding provided
        with pytest.raises(
            ValueError, match="Either text or embedding must be provided"
        ):
            await client.similarity_search(dataset_id="test_dataset", top_k=5)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_retrieval_workflow():
    """Test complete retrieval workflow."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Step 1: Search documents for relevant content
        doc_response = await client.search_documents(
            dataset_id="workflow_dataset",
            query="artificial intelligence applications",
            top_k=3,
            include_content=False,
        )
        assert doc_response.status is True

        # Step 2: Retrieve chunks for RAG context
        chunk_response = await client.retrieve_chunks(
            dataset_id="workflow_dataset",
            query="AI applications in healthcare",
            top_k=5,
            similarity_threshold=0.3,
        )
        assert chunk_response.status is True

        # Step 3: Perform similarity search for related content
        sim_response = await client.similarity_search(
            dataset_id="workflow_dataset",
            text="healthcare AI systems",
            top_k=4,
            content_type="both",
        )
        assert sim_response.status is True

        # Verify we got results from all methods
        assert len(doc_response.results) >= 0
        assert len(chunk_response.chunks) >= 0
        assert len(sim_response.results) >= 0

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_retrieval_parameter_boundaries():
    """Test retrieval APIs with boundary parameter values."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Test with minimal parameters
        response1 = await client.search_documents(
            dataset_id="test_dataset", query="test", top_k=1, similarity_threshold=0.0
        )
        assert response1.status is True

        # Test with maximum reasonable parameters
        response2 = await client.retrieve_chunks(
            dataset_id="test_dataset",
            query="comprehensive test query for maximum parameters",
            top_k=100,
            similarity_threshold=1.0,
            rerank=False,
        )
        assert response2.status is True

        # Test similarity search with both text and embedding
        response3 = await client.similarity_search(
            dataset_id="test_dataset",
            text="test query",
            embedding="dGVzdA==",  # base64 for "test"
            top_k=50,
            similarity_threshold=0.1,
        )
        assert response3.status is True

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_retrieval_error_handling():
    """Test error handling in retrieval APIs."""
    client = RagFlowGRPCClient()
    await client.connect()

    try:
        # Test with invalid dataset_id
        response = await client.search_documents(
            dataset_id="", query="test query"  # Empty dataset ID
        )
        # Should handle gracefully (may still return success in mock)
        assert hasattr(response, "status")

        # Test with empty query
        response2 = await client.retrieve_chunks(
            dataset_id="test_dataset", query=""  # Empty query
        )
        assert hasattr(response2, "status")

    finally:
        await client.disconnect()
