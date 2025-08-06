"""Web UI for testing RAGFlow gRPC API."""
import sys
import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add src to path
sys.path.append('src')
from client import RagFlowGRPCClient

app = FastAPI(title="RAGFlow gRPC API Test UI")
templates = Jinja2Templates(directory="templates")

# Global client instance
grpc_client = RagFlowGRPCClient("localhost:50051")

@app.on_event("startup")
async def startup_event():
    """Connect to gRPC server on startup."""
    try:
        await grpc_client.connect()
        print("✅ Connected to gRPC server")
    except Exception as e:
        print(f"❌ Failed to connect to gRPC server: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from gRPC server on shutdown."""
    try:
        await grpc_client.disconnect()
        print("✅ Disconnected from gRPC server")
    except Exception as e:
        print(f"❌ Failed to disconnect: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with test interface."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>RAGFlow gRPC API Test</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .success { color: green; font-weight: bold; }
        .error { color: red; font-weight: bold; }
        input, textarea, button { margin: 5px; padding: 8px; }
        button { background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .kb-id { font-family: monospace; background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>🚀 RAGFlow gRPC API Test Interface</h1>
    
    <div class="section">
        <h2>📊 Server Status</h2>
        <button onclick="checkStatus()">Check gRPC Connection</button>
        <div id="status-result" class="result"></div>
    </div>
    
    <div class="section">
        <h2>📁 1. Create Knowledge Base</h2>
        <input type="text" id="kb-name" placeholder="Knowledge Base Name" required>
        <input type="text" id="kb-desc" placeholder="Description (optional)">
        <button onclick="createKB()">Create Knowledge Base</button>
        <div id="kb-result" class="result"></div>
    </div>
    
    <div class="section">
        <h2>📄 2. Upload Document</h2>
        <input type="text" id="upload-kb-id" placeholder="Knowledge Base ID" required>
        <input type="file" id="file-upload" required>
        <button onclick="uploadDoc()">Upload Document</button>
        <div id="upload-result" class="result"></div>
    </div>
    
    <div class="section">
        <h2>💬 3. Chat with Documents</h2>
        <input type="text" id="chat-kb-id" placeholder="Knowledge Base ID" required>
        <input type="text" id="question" placeholder="Your question" required style="width: 400px;">
        <button onclick="askQuestion()">Ask Question</button>
        <div id="chat-result" class="result"></div>
    </div>

    <script>
        async function checkStatus() {
            const result = document.getElementById('status-result');
            result.innerHTML = 'Checking connection...';
            
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                if (data.status) {
                    result.innerHTML = '<span class="success">✅ gRPC Server is connected and ready!</span>';
                } else {
                    result.innerHTML = '<span class="error">❌ gRPC Server connection failed: ' + data.message + '</span>';
                }
            } catch (error) {
                result.innerHTML = '<span class="error">❌ Error checking status: ' + error.message + '</span>';
            }
        }

        async function createKB() {
            const name = document.getElementById('kb-name').value;
            const desc = document.getElementById('kb-desc').value;
            const result = document.getElementById('kb-result');
            
            if (!name) {
                result.innerHTML = '<span class="error">Please enter a knowledge base name</span>';
                return;
            }
            
            result.innerHTML = 'Creating knowledge base...';
            
            try {
                const response = await fetch('/api/create-kb', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: name, description: desc})
                });
                
                const data = await response.json();
                if (data.status) {
                    result.innerHTML = '<span class="success">✅ Knowledge Base created successfully!</span><br>' +
                                     'KB ID: <span class="kb-id">' + data.kb_id + '</span><br>' +
                                     '<small>Copy this ID for uploading documents and chatting</small>';
                } else {
                    result.innerHTML = '<span class="error">❌ Failed: ' + data.message + '</span>';
                }
            } catch (error) {
                result.innerHTML = '<span class="error">❌ Error: ' + error.message + '</span>';
            }
        }

        async function uploadDoc() {
            const kbId = document.getElementById('upload-kb-id').value;
            const fileInput = document.getElementById('file-upload');
            const result = document.getElementById('upload-result');
            
            if (!kbId || !fileInput.files[0]) {
                result.innerHTML = '<span class="error">Please enter KB ID and select a file</span>';
                return;
            }
            
            result.innerHTML = 'Uploading document...';
            
            const formData = new FormData();
            formData.append('kb_id', kbId);
            formData.append('file', fileInput.files[0]);
            
            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.status) {
                    result.innerHTML = '<span class="success">✅ Document uploaded successfully!</span><br>' +
                                     'File: ' + fileInput.files[0].name;
                } else {
                    result.innerHTML = '<span class="error">❌ Upload failed: ' + data.message + '</span>';
                }
            } catch (error) {
                result.innerHTML = '<span class="error">❌ Error: ' + error.message + '</span>';
            }
        }

        async function askQuestion() {
            const kbId = document.getElementById('chat-kb-id').value;
            const question = document.getElementById('question').value;
            const result = document.getElementById('chat-result');
            
            if (!kbId || !question) {
                result.innerHTML = '<span class="error">Please enter KB ID and question</span>';
                return;
            }
            
            result.innerHTML = 'Processing question...';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({kb_id: kbId, question: question})
                });
                
                const data = await response.json();
                if (data.status) {
                    result.innerHTML = '<span class="success">✅ Answer received:</span><br>' +
                                     '<strong>Q:</strong> ' + question + '<br>' +
                                     '<strong>A:</strong> ' + (data.answer || 'No answer received');
                } else {
                    result.innerHTML = '<span class="error">❌ Chat failed: ' + data.message + '</span>';
                }
            } catch (error) {
                result.innerHTML = '<span class="error">❌ Error: ' + error.message + '</span>';
            }
        }

        // Check status on page load
        window.onload = function() {
            checkStatus();
        };
    </script>
</body>
</html>
    """)

@app.get("/api/status")
async def check_status():
    """Check if gRPC server is responding."""
    try:
        # Try to create a simple request to test connection
        response = await grpc_client.create_knowledge_base("_test_connection", "test")
        return {"status": True, "message": "gRPC server is responsive"}
    except Exception as e:
        return {"status": False, "message": str(e)}

@app.post("/api/create-kb")
async def create_knowledge_base(request: Request):
    """Create knowledge base via gRPC."""
    try:
        data = await request.json()
        response = await grpc_client.create_knowledge_base(
            data.get("name", ""), 
            data.get("description", "")
        )
        return {
            "status": response.status,
            "message": response.message,
            "kb_id": response.kb_id if hasattr(response, 'kb_id') else ""
        }
    except Exception as e:
        return {"status": False, "message": str(e)}

@app.post("/api/upload")
async def upload_document(kb_id: str = Form(...), file: UploadFile = File(...)):
    """Upload document via gRPC."""
    try:
        content = await file.read()
        response = await grpc_client.upload_document(kb_id, content, file.filename)
        return {
            "status": response.status,
            "message": response.message
        }
    except Exception as e:
        return {"status": False, "message": str(e)}

@app.post("/api/chat")
async def chat_with_kb(request: Request):
    """Chat with knowledge base via gRPC."""
    try:
        data = await request.json()
        response = await grpc_client.chat(data.get("kb_id", ""), data.get("question", ""))
        return {
            "status": response.status,
            "message": response.message,
            "answer": response.answer if hasattr(response, 'answer') else ""
        }
    except Exception as e:
        return {"status": False, "message": str(e)}

if __name__ == "__main__":
    print("🚀 Starting RAGFlow gRPC Test UI...")
    print("📡 Make sure your gRPC server is running: uv run python -m src.server")
    print("🌐 Web UI will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)