"""Example usage of RAGFlow gRPC client with Dataset CRUD operations."""

import asyncio
import os
import sys

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import RagFlowGRPCClient, RagFlowSyncClient


async def dataset_crud_example():
    """Demonstrate Dataset CRUD operations."""
    client = RagFlowGRPCClient("localhost:50051")

    try:
        await client.connect()
        print("=== Dataset CRUD Example ===")

        # 1. Create multiple knowledge bases
        print("\n1. Creating knowledge bases...")
        kb1_response = await client.create_knowledge_base(
            "Python Documentation", "Python language documentation"
        )
        print(
            f"KB1 Creation - Status: {kb1_response.status}, KB ID: {kb1_response.kb_id}"
        )

        kb2_response = await client.create_knowledge_base(
            "Machine Learning", "ML algorithms and concepts"
        )
        print(
            f"KB2 Creation - Status: {kb2_response.status}, KB ID: {kb2_response.kb_id}"
        )

        # 2. List all datasets
        print("\n2. Listing all datasets...")
        list_response = await client.list_datasets(page=1, page_size=10)
        print(f"List Status: {list_response.status}")
        print(f"Found {len(list_response.datasets)} datasets:")
        for dataset in list_response.datasets:
            print(
                f"  - ID: {dataset.id}, Name: {dataset.name}, Documents: {dataset.document_count}"
            )

        # 3. Filter datasets by name
        print("\n3. Filtering datasets by name 'Python'...")
        filtered_response = await client.list_datasets(name="Python")
        print(f"Filtered Status: {filtered_response.status}")
        print(f"Found {len(filtered_response.datasets)} matching datasets")

        # 4. Update a dataset
        if kb1_response.status and kb1_response.kb_id:
            print(f"\n4. Updating dataset {kb1_response.kb_id}...")
            update_response = await client.update_dataset(
                kb1_response.kb_id,
                description="Updated: Comprehensive Python documentation and tutorials",
                chunk_method="book",
            )
            print(
                f"Update Status: {update_response.status}, Message: {update_response.message}"
            )

        # 5. Upload a document
        if kb1_response.status and kb1_response.kb_id:
            print(f"\n5. Uploading document to {kb1_response.kb_id}...")
            test_content = b"Python is a high-level programming language. It supports multiple programming paradigms."
            upload_response = await client.upload_document(
                kb1_response.kb_id, test_content, "python_intro.txt"
            )
            print(
                f"Upload Status: {upload_response.status}, Message: {upload_response.message}"
            )

        # 6. Chat with the knowledge base
        if kb1_response.status and kb1_response.kb_id:
            print(f"\n6. Chatting with {kb1_response.kb_id}...")
            chat_response = await client.chat(kb1_response.kb_id, "What is Python?")
            print(f"Chat Status: {chat_response.status}")
            print(f"Answer: {chat_response.answer}")

        # 7. Delete specific datasets
        if kb2_response.status and kb2_response.kb_id:
            print(f"\n7. Deleting dataset {kb2_response.kb_id}...")
            delete_response = await client.delete_datasets([kb2_response.kb_id])
            print(
                f"Delete Status: {delete_response.status}, Message: {delete_response.message}"
            )

        # 8. List datasets after deletion
        print("\n8. Listing datasets after deletion...")
        final_list_response = await client.list_datasets()
        print(f"Final List Status: {final_list_response.status}")
        print(f"Remaining datasets: {len(final_list_response.datasets)}")
        for dataset in final_list_response.datasets:
            print(f"  - ID: {dataset.id}, Name: {dataset.name}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.disconnect()


def sync_dataset_example():
    """Sync client example for dataset operations."""
    client = RagFlowSyncClient("localhost:50051")

    try:
        print("\n=== Sync Dataset Example ===")

        # Create and list datasets
        kb_response = client.create_knowledge_base(
            "Sync Test KB", "Testing sync client"
        )
        print(
            f"Sync KB Creation - Status: {kb_response.status}, KB ID: {kb_response.kb_id}"
        )

        list_response = client.list_datasets(page_size=5)
        print(
            f"Sync List - Status: {list_response.status}, Count: {len(list_response.datasets)}"
        )

        # Clean up
        if kb_response.status and kb_response.kb_id:
            delete_response = client.delete_datasets([kb_response.kb_id])
            print(f"Sync Delete - Status: {delete_response.status}")

    except Exception as e:
        print(f"Sync Error: {e}")


if __name__ == "__main__":
    print("=== RAGFlow gRPC Dataset CRUD Examples ===")

    # Run async example
    asyncio.run(dataset_crud_example())

    # Run sync example
    sync_dataset_example()
