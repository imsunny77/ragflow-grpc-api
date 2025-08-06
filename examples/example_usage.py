"""Example usage of RAGFlow gRPC client."""
import asyncio
import sys
import os

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import RagFlowGRPCClient, RagFlowSyncClient


async def async_example():
    """Async client example."""
    client = RagFlowGRPCClient("localhost:50051")
    
    try:
        await client.connect()
        
        # Create knowledge base
        kb_response = await client.create_knowledge_base("Test KB", "Test description")
        print(f"KB Creation - Status: {kb_response.status}, Message: {kb_response.message}")
        
        if kb_response.status and kb_response.kb_id:
            kb_id = kb_response.kb_id
            
            # Upload document
            test_content = b"This is a test document content for RAGFlow."
            upload_response = await client.upload_document(kb_id, test_content, "test.txt")
            print(f"Upload - Status: {upload_response.status}, Message: {upload_response.message}")
            
            # Chat
            chat_response = await client.chat(kb_id, "What is in the document?")
            print(f"Chat - Status: {chat_response.status}, Answer: {chat_response.answer}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await client.disconnect()


def sync_example():
    """Sync client example."""
    client = RagFlowSyncClient("localhost:50051")
    
    try:
        # Create knowledge base
        kb_response = client.create_knowledge_base("Sync Test KB", "Sync test description")
        print(f"Sync KB Creation - Status: {kb_response.status}, Message: {kb_response.message}")
        
        if kb_response.status and kb_response.kb_id:
            kb_id = kb_response.kb_id
            
            # Upload document
            test_content = b"This is a sync test document content."
            upload_response = client.upload_document(kb_id, test_content, "sync_test.txt")
            print(f"Sync Upload - Status: {upload_response.status}")
            
            # Chat
            chat_response = client.chat(kb_id, "What is this document about?")
            print(f"Sync Chat - Status: {chat_response.status}, Answer: {chat_response.answer}")
    
    except Exception as e:
        print(f"Sync Error: {e}")


if __name__ == "__main__":
    print("=== Async Example ===")
    asyncio.run(async_example())
    
    print("\n=== Sync Example ===")
    sync_example()