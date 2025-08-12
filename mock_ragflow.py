"""Mock RAGFlow server for testing with Chat Assistant Management."""

from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.responses import Response
from typing import Optional, List
import uvicorn
from datetime import datetime
import io

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
        "knowledgebase_id": "mock_kb_123",
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

mock_chat_assistants = {
    "chat_123": {
        "id": "chat_123",
        "name": "Python Assistant",
        "description": "Helps with Python programming questions",
        "avatar": "",
        "dataset_ids": ["mock_kb_123"],
        "llm": {
            "model_name": "gpt-4",
            "temperature": 0.1,
            "top_p": 0.3,
            "presence_penalty": 0.4,
            "frequency_penalty": 0.7,
        },
        "prompt": {
            "prompt": "You are a Python programming expert. Help users with their Python questions.",
            "similarity_threshold": 0.2,
            "keywords_similarity_weight": 0.7,
            "top_n": 6,
        },
        "create_date": "Mon, 01 Jan 2024 14:00:00 GMT",
        "update_date": "Mon, 01 Jan 2024 14:00:00 GMT",
    }
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
    page: Optional[int] = Query(1),
    page_size: Optional[int] = Query(30),
    orderby: Optional[str] = Query("create_time"),
    desc: Optional[str] = Query("true"),
    name: Optional[str] = Query(None),
    id: Optional[str] = Query(None),
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
    mock_datasets[dataset_id]["document_count"] += 1

    return {"code": 0, "data": [{"id": doc_id, "name": file.filename}]}


@app.get("/api/v1/datasets/{dataset_id}/documents")
async def list_documents(
    dataset_id: str,
    page: Optional[int] = Query(1),
    page_size: Optional[int] = Query(30),
    orderby: Optional[str] = Query("create_time"),
    desc: Optional[str] = Query("true"),
    keywords: Optional[str] = Query(None),
    id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
):
    """List documents in dataset."""
    if dataset_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    documents = [
        doc for doc in mock_documents.values() if doc["dataset_id"] == dataset_id
    ]

    if keywords:
        documents = [d for d in documents if keywords.lower() in d["name"].lower()]
    if id:
        documents = [d for d in documents if d["id"] == id]
    if name:
        documents = [d for d in documents if name.lower() in d["name"].lower()]

    reverse_order = desc.lower() == "true"
    if orderby == "create_time":
        documents.sort(key=lambda x: x["create_date"], reverse=reverse_order)
    elif orderby == "update_time":
        documents.sort(key=lambda x: x["update_date"], reverse=reverse_order)

    total = len(documents)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_docs = documents[start_idx:end_idx]

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

    if "name" in data:
        document["name"] = data["name"]
    if "chunk_method" in data:
        document["chunk_method"] = data["chunk_method"]
    if "parser_config" in data:
        pass  # Store parser config

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
        docs_to_delete = [
            doc_id
            for doc_id, doc in mock_documents.items()
            if doc["dataset_id"] == dataset_id
        ]
        for doc_id in docs_to_delete:
            del mock_documents[doc_id]
            deleted_count += 1
    elif isinstance(ids, list):
        for doc_id in ids:
            if (
                doc_id in mock_documents
                and mock_documents[doc_id]["dataset_id"] == dataset_id
            ):
                del mock_documents[doc_id]
                deleted_count += 1

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

    total_chunks = 0
    for doc_id in document_ids:
        if (
            doc_id in mock_documents
            and mock_documents[doc_id]["dataset_id"] == dataset_id
        ):
            doc_size = mock_documents[doc_id]["size"]
            chunks = max(1, doc_size // 200)
            mock_documents[doc_id]["chunk_count"] = chunks
            total_chunks += chunks

    mock_datasets[dataset_id]["chunk_count"] += total_chunks
    return {
        "code": 0,
        "data": {"message": f"Started parsing {len(document_ids)} documents"},
    }


# Chat Assistant Management Endpoints
@app.post("/api/v1/chats")
async def create_chat_assistant(data: dict):
    """Create a new chat assistant."""
    chat_id = f"chat_{len(mock_chat_assistants) + 1}"
    current_time = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Extract LLM configuration
    llm_config = data.get("llm", {})
    default_llm = {
        "model_name": llm_config.get("model_name", "default"),
        "temperature": llm_config.get("temperature", 0.1),
        "top_p": llm_config.get("top_p", 0.3),
        "presence_penalty": llm_config.get("presence_penalty", 0.4),
        "frequency_penalty": llm_config.get("frequency_penalty", 0.7),
    }

    # Extract prompt configuration
    prompt_config = data.get("prompt", {})
    default_prompt = {
        "prompt": prompt_config.get("prompt", "You are a helpful assistant."),
        "similarity_threshold": prompt_config.get("similarity_threshold", 0.2),
        "keywords_similarity_weight": prompt_config.get(
            "keywords_similarity_weight", 0.7
        ),
        "top_n": prompt_config.get("top_n", 6),
    }

    new_assistant = {
        "id": chat_id,
        "name": data.get("name", ""),
        "description": data.get("description", ""),
        "avatar": data.get("avatar", ""),
        "dataset_ids": data.get("dataset_ids", []),
        "llm": default_llm,
        "prompt": default_prompt,
        "create_date": current_time,
        "update_date": current_time,
    }

    mock_chat_assistants[chat_id] = new_assistant
    return {"code": 0, "data": {"id": chat_id}}


@app.get("/api/v1/chats")
async def list_chat_assistants(
    page: Optional[int] = Query(1),
    page_size: Optional[int] = Query(30),
    orderby: Optional[str] = Query("create_time"),
    desc: Optional[str] = Query("true"),
    name: Optional[str] = Query(None),
    id: Optional[str] = Query(None),
):
    """List chat assistants with pagination and filtering."""
    assistants = list(mock_chat_assistants.values())

    # Apply filtering
    if name:
        assistants = [a for a in assistants if name.lower() in a["name"].lower()]
    if id:
        assistants = [a for a in assistants if a["id"] == id]

    # Apply sorting
    reverse_order = desc.lower() == "true"
    if orderby == "create_time":
        assistants.sort(key=lambda x: x["create_date"], reverse=reverse_order)
    elif orderby == "update_time":
        assistants.sort(key=lambda x: x["update_date"], reverse=reverse_order)

    # Apply pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_assistants = assistants[start_idx:end_idx]

    return {"code": 0, "data": paginated_assistants}


@app.put("/api/v1/chats/{chat_id}")
async def update_chat_assistant(chat_id: str, data: dict):
    """Update chat assistant configuration."""
    if chat_id not in mock_chat_assistants:
        return {"code": 102, "message": "Chat assistant not found"}

    assistant = mock_chat_assistants[chat_id]
    current_time = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Update basic fields
    if "name" in data:
        assistant["name"] = data["name"]
    if "description" in data:
        assistant["description"] = data["description"]
    if "avatar" in data:
        assistant["avatar"] = data["avatar"]
    if "dataset_ids" in data:
        assistant["dataset_ids"] = data["dataset_ids"]

    # Update LLM configuration
    if "llm" in data:
        llm_update = data["llm"]
        assistant["llm"].update(llm_update)

    # Update prompt configuration
    if "prompt" in data:
        prompt_update = data["prompt"]
        assistant["prompt"].update(prompt_update)

    assistant["update_date"] = current_time
    return {"code": 0}


@app.delete("/api/v1/chats")
async def delete_chat_assistants(data: dict):
    """Delete chat assistants by IDs."""
    ids = data.get("ids")

    if ids is None:
        # Delete all chat assistants
        mock_chat_assistants.clear()
    elif isinstance(ids, list):
        # Delete specified chat assistants
        for chat_id in ids:
            if chat_id in mock_chat_assistants:
                del mock_chat_assistants[chat_id]

    return {"code": 0}


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
