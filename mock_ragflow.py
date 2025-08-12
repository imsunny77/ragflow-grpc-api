"""Mock RAGFlow server for testing with Document Management."""

from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

app = FastAPI()

# Mock data storage
mock_datasets = {
    "mock_kb_123": {
        "id": "mock_kb_123",
        "name": "Test Knowledge Base",
        "description": "A test knowledge base",
        "avatar": "",
        "embedding_model": "BAAI/bge-large-zh-v1.5@BAAI",
        "permission": "me",
        "chunk_method": "naive",
        "chunk_count": 0,
        "document_count": 0,
        "create_date": "Mon, 01 Jan 2024 12:00:00 GMT",
        "update_date": "Mon, 01 Jan 2024 12:00:00 GMT",
    }
}

mock_documents = {
    "doc_123": {
        "id": "doc_123",
        "name": "sample_document.txt",
        "dataset_id": "mock_kb_123",
        "knowledgebase_id": "mock_kb_123",  # Alternative field name
        "size": 1024,
        "type": "doc",
        "chunk_method": "naive",
        "chunk_count": 5,
        "status": "1",
        "create_date": "Mon, 01 Jan 2024 12:00:00 GMT",
        "update_date": "Mon, 01 Jan 2024 12:00:00 GMT",
        "thumbnail": None,
        "file_content": b"This is sample document content for testing.",
    },
    "doc_124": {
        "id": "doc_124",
        "name": "python_guide.pdf",
        "dataset_id": "mock_kb_123",
        "knowledgebase_id": "mock_kb_123",
        "size": 2048,
        "type": "pdf",
        "chunk_method": "book",
        "chunk_count": 10,
        "status": "1",
        "create_date": "Mon, 01 Jan 2024 13:00:00 GMT",
        "update_date": "Mon, 01 Jan 2024 13:00:00 GMT",
        "thumbnail": "",
        "file_content": b"Python programming guide content here...",
    },
}


@app.get("/")
async def root():
    return {"status": "ok"}


# Dataset Management Endpoints
@app.post("/api/v1/datasets")
async def create_dataset(data: dict):
    """Create a new dataset."""
    dataset_id = f"mock_kb_{len(mock_datasets) + 1}"
    current_time = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

    new_dataset = {
        "id": dataset_id,
        "name": data.get("name", ""),
        "description": data.get("description", ""),
        "avatar": data.get("avatar", ""),
        "embedding_model": data.get("embedding_model", "BAAI/bge-large-zh-v1.5@BAAI"),
        "permission": data.get("permission", "me"),
        "chunk_method": data.get("chunk_method", "naive"),
        "chunk_count": 0,
        "document_count": 0,
        "create_date": current_time,
        "update_date": current_time,
    }

    mock_datasets[dataset_id] = new_dataset
    return {"code": 0, "data": {"id": dataset_id}}


@app.get("/api/v1/datasets")
async def list_datasets(
    page: int | None = Query(1),
    page_size: int | None = Query(30),
    orderby: str | None = Query("create_time"),
    desc: str | None = Query("true"),
    name: str | None = Query(None),
    id: str | None = Query(None),
):
    """List datasets with pagination and filtering."""
    datasets = list(mock_datasets.values())

    if name:
        datasets = [d for d in datasets if name.lower() in d["name"].lower()]
    if id:
        datasets = [d for d in datasets if d["id"] == id]

    reverse_order = desc.lower() == "true"
    if orderby == "create_time":
        datasets.sort(key=lambda x: x["create_date"], reverse=reverse_order)
    elif orderby == "update_time":
        datasets.sort(key=lambda x: x["update_date"], reverse=reverse_order)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_datasets = datasets[start_idx:end_idx]

    return {"code": 0, "data": paginated_datasets}


@app.put("/api/v1/datasets/{dataset_id}")
async def update_dataset(dataset_id: str, data: dict):
    """Update dataset configuration."""
    if dataset_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    dataset = mock_datasets[dataset_id]
    current_time = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

    if "name" in data:
        dataset["name"] = data["name"]
    if "description" in data:
        dataset["description"] = data["description"]
    if "embedding_model" in data:
        dataset["embedding_model"] = data["embedding_model"]
    if "permission" in data:
        dataset["permission"] = data["permission"]
    if "chunk_method" in data:
        dataset["chunk_method"] = data["chunk_method"]

    dataset["update_date"] = current_time
    return {"code": 0}


@app.delete("/api/v1/datasets")
async def delete_datasets(data: dict):
    """Delete datasets by IDs."""
    ids = data.get("ids")

    if ids is None:
        mock_datasets.clear()
    elif isinstance(ids, list):
        for dataset_id in ids:
            if dataset_id in mock_datasets:
                del mock_datasets[dataset_id]
                # Also delete associated documents
                docs_to_delete = [
                    doc_id
                    for doc_id, doc in mock_documents.items()
                    if doc["dataset_id"] == dataset_id
                ]
                for doc_id in docs_to_delete:
                    del mock_documents[doc_id]

    return {"code": 0}


# Document Management Endpoints
@app.post("/api/v1/datasets/{dataset_id}/documents")
async def upload_document(dataset_id: str, file: UploadFile = File(...)):
    """Upload document to dataset."""
    if dataset_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    # Create new document
    doc_id = f"doc_{len(mock_documents) + 1}"
    current_time = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
    file_content = await file.read()

    new_document = {
        "id": doc_id,
        "name": file.filename or "uploaded_file",
        "dataset_id": dataset_id,
        "knowledgebase_id": dataset_id,
        "size": len(file_content),
        "type": "doc",
        "chunk_method": "naive",
        "chunk_count": 0,
        "status": "1",
        "create_date": current_time,
        "update_date": current_time,
        "thumbnail": None,
        "file_content": file_content,
    }

    mock_documents[doc_id] = new_document

    # Update dataset document count
    mock_datasets[dataset_id]["document_count"] += 1

    return {"code": 0, "data": [{"id": doc_id, "name": file.filename}]}


@app.get("/api/v1/datasets/{dataset_id}/documents")
async def list_documents(
    dataset_id: str,
    page: int | None = Query(1),
    page_size: int | None = Query(30),
    orderby: str | None = Query("create_time"),
    desc: str | None = Query("true"),
    keywords: str | None = Query(None),
    id: str | None = Query(None),
    name: str | None = Query(None),
):
    """List documents in dataset."""
    if dataset_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    # Filter documents by dataset
    documents = [
        doc for doc in mock_documents.values() if doc["dataset_id"] == dataset_id
    ]

    # Apply additional filters
    if keywords:
        documents = [d for d in documents if keywords.lower() in d["name"].lower()]
    if id:
        documents = [d for d in documents if d["id"] == id]
    if name:
        documents = [d for d in documents if name.lower() in d["name"].lower()]

    # Apply sorting
    reverse_order = desc.lower() == "true"
    if orderby == "create_time":
        documents.sort(key=lambda x: x["create_date"], reverse=reverse_order)
    elif orderby == "update_time":
        documents.sort(key=lambda x: x["update_date"], reverse=reverse_order)

    # Apply pagination
    total = len(documents)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_docs = documents[start_idx:end_idx]

    # Remove file_content from response
    clean_docs = []
    for doc in paginated_docs:
        clean_doc = {k: v for k, v in doc.items() if k != "file_content"}
        clean_docs.append(clean_doc)

    return {"code": 0, "data": {"docs": clean_docs, "total": total}}


@app.put("/api/v1/datasets/{dataset_id}/documents/{document_id}")
async def update_document(dataset_id: str, document_id: str, data: dict):
    """Update document configuration."""
    if dataset_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    if document_id not in mock_documents:
        return {"code": 102, "message": "Document not found"}

    document = mock_documents[document_id]
    if document["dataset_id"] != dataset_id:
        return {"code": 102, "message": "Document not in specified dataset"}

    current_time = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Update fields
    if "name" in data:
        document["name"] = data["name"]
    if "chunk_method" in data:
        document["chunk_method"] = data["chunk_method"]
    if "parser_config" in data:
        # Store parser config (in real implementation, this would affect parsing)
        pass

    document["update_date"] = current_time
    return {"code": 0}


@app.get("/api/v1/datasets/{dataset_id}/documents/{document_id}")
async def download_document(dataset_id: str, document_id: str):
    """Download document file."""
    if dataset_id not in mock_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if document_id not in mock_documents:
        raise HTTPException(status_code=404, detail="Document not found")

    document = mock_documents[document_id]
    if document["dataset_id"] != dataset_id:
        raise HTTPException(status_code=404, detail="Document not in specified dataset")

    file_content = document["file_content"]
    filename = document["name"]

    return Response(
        content=file_content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/v1/datasets/{dataset_id}/documents")
async def delete_documents(dataset_id: str, data: dict):
    """Delete documents from dataset."""
    if dataset_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    ids = data.get("ids")
    deleted_count = 0

    if ids is None:
        # Delete all documents in dataset
        docs_to_delete = [
            doc_id
            for doc_id, doc in mock_documents.items()
            if doc["dataset_id"] == dataset_id
        ]
        for doc_id in docs_to_delete:
            del mock_documents[doc_id]
            deleted_count += 1
    elif isinstance(ids, list):
        # Delete specified documents
        for doc_id in ids:
            if (
                doc_id in mock_documents
                and mock_documents[doc_id]["dataset_id"] == dataset_id
            ):
                del mock_documents[doc_id]
                deleted_count += 1

    # Update dataset document count
    mock_datasets[dataset_id]["document_count"] -= deleted_count

    return {"code": 0}


@app.post("/api/v1/datasets/{dataset_id}/chunks")
async def parse_documents(dataset_id: str, data: dict):
    """Start parsing documents into chunks."""
    if dataset_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    document_ids = data.get("document_ids", [])
    if not document_ids:
        return {"code": 102, "message": "document_ids is required"}

    # Simulate parsing process
    total_chunks = 0
    for doc_id in document_ids:
        if (
            doc_id in mock_documents
            and mock_documents[doc_id]["dataset_id"] == dataset_id
        ):
            # Simulate chunk creation based on file size
            doc_size = mock_documents[doc_id]["size"]
            chunks = max(1, doc_size // 200)  # 200 bytes per chunk
            mock_documents[doc_id]["chunk_count"] = chunks
            total_chunks += chunks

    # Update dataset chunk count
    mock_datasets[dataset_id]["chunk_count"] += total_chunks

    return {
        "code": 0,
        "data": {"message": f"Started parsing {len(document_ids)} documents"},
    }


# Chat Endpoint
@app.post("/api/v1/datasets/{kb_id}/chat")
async def chat(kb_id: str, data: dict):
    """Chat with knowledge base."""
    if kb_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    question = data.get("question", "")
    return {"code": 0, "data": {"answer": f"Mock answer for: {question}"}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9380)
