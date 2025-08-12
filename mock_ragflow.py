"""Mock RAGFlow server for testing."""

from fastapi import FastAPI, Query
from typing import Optional, List
import uvicorn
from datetime import datetime

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


@app.get("/")
async def root():
    return {"status": "ok"}


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

    # Apply filtering
    if name:
        datasets = [d for d in datasets if name.lower() in d["name"].lower()]
    if id:
        datasets = [d for d in datasets if d["id"] == id]

    # Apply sorting
    reverse_order = desc.lower() == "true"
    if orderby == "create_time":
        datasets.sort(key=lambda x: x["create_date"], reverse=reverse_order)
    elif orderby == "update_time":
        datasets.sort(key=lambda x: x["update_date"], reverse=reverse_order)

    # Apply pagination
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

    # Update only provided fields
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
        # Delete all datasets
        mock_datasets.clear()
    elif isinstance(ids, list):
        # Delete specified datasets
        for dataset_id in ids:
            if dataset_id in mock_datasets:
                del mock_datasets[dataset_id]

    return {"code": 0}


@app.post("/api/v1/datasets/{kb_id}/documents")
async def upload_doc(kb_id: str):
    """Upload document to knowledge base."""
    if kb_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    # Increment document count
    mock_datasets[kb_id]["document_count"] += 1

    return {"code": 0, "data": {"id": "doc_123"}}


@app.post("/api/v1/datasets/{kb_id}/chat")
async def chat(kb_id: str, data: dict):
    """Chat with knowledge base."""
    if kb_id not in mock_datasets:
        return {"code": 102, "message": "Dataset not found"}

    question = data.get("question", "")
    return {"code": 0, "data": {"answer": f"Mock answer for: {question}"}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9380)
