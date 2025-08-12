"""Mock RAGFlow API server for testing."""

from typing import Any

from flask import Flask

app = Flask(__name__)


def create_response(status: bool, message: str, **kwargs: Any) -> dict[str, Any]:
    """Create standardized response."""
    response = {"status": status, "message": message}
    response.update(kwargs)
    return response


@app.route("/api/v1/dataset", methods=["POST"])
def create_dataset() -> dict[str, Any]:
    return create_response(True, "Dataset created", id="test_dataset_123")


@app.route("/api/v1/document", methods=["POST"])
def upload_doc() -> dict[str, Any]:
    return create_response(True, "Document uploaded successfully")


@app.route("/api/v1/chat", methods=["POST"])
def chat() -> dict[str, Any]:
    return create_response(True, "Chat successful", answer="This is a mock response")


if __name__ == "__main__":
    app.run(debug=True, port=8000)
