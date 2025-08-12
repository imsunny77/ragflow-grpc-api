"""Chat Assistant Management example for RAGFlow gRPC client."""

import asyncio
import sys
import os

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import RagFlowGRPCClient


async def chat_assistant_management_example():
    """Demonstrate complete Chat Assistant Management workflow."""
    client = RagFlowGRPCClient("localhost:50051")

    try:
        await client.connect()
        print("=== Chat Assistant Management Example ===")

        # 1. Create knowledge bases for testing
        print("\n1. Setting up knowledge bases for assistants...")
        kb1_response = await client.create_knowledge_base(
            "Python Programming KB", "Knowledge base for Python programming questions"
        )
        print(
            f"KB1 Creation - Status: {kb1_response.status}, KB ID: {kb1_response.kb_id}"
        )

        kb2_response = await client.create_knowledge_base(
            "Machine Learning KB", "Knowledge base for ML and AI topics"
        )
        print(
            f"KB2 Creation - Status: {kb2_response.status}, KB ID: {kb2_response.kb_id}"
        )

        # Store KB IDs for assistant creation
        kb_ids = []
        if kb1_response.status and kb1_response.kb_id:
            kb_ids.append(kb1_response.kb_id)
        if kb2_response.status and kb2_response.kb_id:
            kb_ids.append(kb2_response.kb_id)

        # 2. Create multiple chat assistants with different configurations
        print("\n2. Creating chat assistants...")

        # Create Python expert assistant
        python_assistant = await client.create_chat_assistant(
            name="Python Expert",
            description="Specialized assistant for Python programming questions",
            dataset_ids=[kb_ids[0]] if kb_ids else [],
            llm_model="gpt-4",
            temperature=0.1,
            top_p=0.3,
            prompt="You are a Python programming expert. Provide clear, accurate answers about Python concepts, syntax, and best practices.",
            similarity_threshold=0.3,
            keywords_similarity_weight=0.8,
            top_n=8,
        )
        print(
            f"Python Assistant - Status: {python_assistant.status}, ID: {python_assistant.chat_id}"
        )

        # Create general AI assistant with multiple knowledge bases
        ai_assistant = await client.create_chat_assistant(
            name="AI Research Assistant",
            description="General AI assistant with access to multiple knowledge bases",
            dataset_ids=kb_ids,
            llm_model="gpt-4o",
            temperature=0.2,
            top_p=0.4,
            presence_penalty=0.3,
            frequency_penalty=0.6,
            prompt="You are an AI research assistant. Help users with programming and machine learning questions using the available knowledge bases.",
            similarity_threshold=0.2,
            keywords_similarity_weight=0.7,
            top_n=10,
        )
        print(
            f"AI Assistant - Status: {ai_assistant.status}, ID: {ai_assistant.chat_id}"
        )

        # Create creative assistant with different parameters
        creative_assistant = await client.create_chat_assistant(
            name="Creative Coding Assistant",
            description="Assistant for creative programming and algorithmic art",
            dataset_ids=[kb_ids[0]] if kb_ids else [],
            llm_model="claude-3",
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.6,
            frequency_penalty=0.4,
            prompt="You are a creative programming assistant. Help users explore creative coding, algorithmic art, and innovative programming solutions.",
            similarity_threshold=0.15,
            keywords_similarity_weight=0.6,
            top_n=5,
        )
        print(
            f"Creative Assistant - Status: {creative_assistant.status}, ID: {creative_assistant.chat_id}"
        )

        # Store assistant IDs for later operations
        assistant_ids = []
        for response in [python_assistant, ai_assistant, creative_assistant]:
            if response.status and response.chat_id:
                assistant_ids.append(response.chat_id)

        # 3. List all chat assistants
        print("\n3. Listing all chat assistants...")
        list_response = await client.list_chat_assistants(page=1, page_size=10)
        print(f"List Status: {list_response.status}")
        print(f"Found {len(list_response.assistants)} assistants:")
        for assistant in list_response.assistants:
            print(f"  - ID: {assistant.id}")
            print(f"    Name: {assistant.name}")
            print(f"    Model: {assistant.llm_model}")
            print(f"    Temperature: {assistant.temperature}")
            print(f"    Datasets: {len(assistant.dataset_ids)} connected")
            print(f"    Description: {assistant.description[:50]}...")
            print()

        # 4. Filter assistants by name
        print("4. Filtering assistants by name 'Python'...")
        filtered_response = await client.list_chat_assistants(name="Python")
        print(f"Filtered Status: {filtered_response.status}")
        print(f"Found {len(filtered_response.assistants)} matching assistants")
        for assistant in filtered_response.assistants:
            print(f"  - {assistant.name}: {assistant.llm_model}")

        # 5. Update an assistant
        if assistant_ids:
            first_assistant_id = assistant_ids[0]
            print(f"\n5. Updating assistant {first_assistant_id}...")
            update_response = await client.update_chat_assistant(
                first_assistant_id,
                name="Advanced Python Expert",
                description="Updated: Advanced Python programming specialist with enhanced capabilities",
                temperature=0.05,  # More deterministic
                top_n=12,  # More context
                prompt="You are an advanced Python programming expert with deep knowledge of Python internals, performance optimization, and advanced patterns.",
            )
            print(
                f"Update Status: {update_response.status}, Message: {update_response.message}"
            )

        # 6. Test assistant configuration by listing again
        print("\n6. Verifying updates...")
        if assistant_ids:
            updated_response = await client.list_chat_assistants(
                chat_id=assistant_ids[0]
            )
            if updated_response.status and updated_response.assistants:
                assistant = updated_response.assistants[0]
                print(f"Updated Assistant: {assistant.name}")
                print(f"New Temperature: {assistant.temperature}")
                print(f"New Top N: {assistant.top_n}")
                print(f"New Description: {assistant.description}")

        # 7. Demonstrate different assistant types
        print("\n7. Assistant Configuration Summary:")
        final_list = await client.list_chat_assistants()
        if final_list.status:
            for assistant in final_list.assistants:
                print(f"\nAssistant: {assistant.name}")
                print(f"  Model: {assistant.llm_model}")
                print(f"  Temperature: {assistant.temperature} (creativity level)")
                print(f"  Top P: {assistant.top_p}")
                print(f"  Similarity Threshold: {assistant.similarity_threshold}")
                print(f"  Keywords Weight: {assistant.keywords_similarity_weight}")
                print(f"  Context Size (top_n): {assistant.top_n}")
                print(f"  Connected Datasets: {len(assistant.dataset_ids)}")

        # 8. Delete specific assistant
        if len(assistant_ids) > 1:
            assistant_to_delete = assistant_ids[-1]  # Delete last assistant
            print(f"\n8. Deleting assistant {assistant_to_delete}...")
            delete_response = await client.delete_chat_assistants([assistant_to_delete])
            print(
                f"Delete Status: {delete_response.status}, Message: {delete_response.message}"
            )

        # 9. Final assistant count
        print("\n9. Final assistant count...")
        final_response = await client.list_chat_assistants()
        print(f"Final Status: {final_response.status}")
        print(f"Remaining assistants: {len(final_response.assistants)}")
        for assistant in final_response.assistants:
            print(f"  - {assistant.name} ({assistant.llm_model})")

        # 10. Clean up - delete test knowledge bases and remaining assistants
        print(f"\n10. Cleaning up test data...")

        # Delete remaining assistants
        remaining_ids = [
            a.id for a in final_response.assistants if a.id in assistant_ids
        ]
        if remaining_ids:
            cleanup_assistants = await client.delete_chat_assistants(remaining_ids)
            print(f"Assistant Cleanup Status: {cleanup_assistants.status}")

        # Delete test knowledge bases
        if kb_ids:
            cleanup_kb = await client.delete_datasets(kb_ids)
            print(f"Knowledge Base Cleanup Status: {cleanup_kb.status}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.disconnect()


async def assistant_configuration_showcase():
    """Showcase different assistant configurations for various use cases."""
    client = RagFlowGRPCClient("localhost:50051")

    try:
        await client.connect()
        print("\n=== Assistant Configuration Showcase ===")

        # Create a test knowledge base
        kb_response = await client.create_knowledge_base(
            "Test KB", "For configuration testing"
        )
        if not kb_response.status:
            print("Failed to create test KB")
            return

        kb_id = kb_response.kb_id

        # Configuration 1: Conservative/Factual Assistant
        print(
            "\n1. Creating Conservative Assistant (Low creativity, high precision)..."
        )
        conservative = await client.create_chat_assistant(
            name="Factual Assistant",
            description="Conservative assistant for factual information",
            dataset_ids=[kb_id],
            llm_model="gpt-4",
            temperature=0.0,  # Very deterministic
            top_p=0.1,  # Very focused
            presence_penalty=0.0,  # No penalty for repetition
            frequency_penalty=0.0,
            prompt="Provide accurate, factual information based solely on the knowledge base. Be precise and conservative in your responses.",
            similarity_threshold=0.4,  # High threshold for relevance
            keywords_similarity_weight=0.9,  # Emphasize keyword matching
            top_n=3,  # Small context window
        )
        print(f"Conservative Assistant: {conservative.status}")

        # Configuration 2: Creative/Exploratory Assistant
        print("\n2. Creating Creative Assistant (High creativity, exploratory)...")
        creative = await client.create_chat_assistant(
            name="Creative Assistant",
            description="Creative assistant for brainstorming and exploration",
            dataset_ids=[kb_id],
            llm_model="claude-3",
            temperature=0.8,  # High creativity
            top_p=0.95,  # Very diverse
            presence_penalty=0.7,  # Encourage new topics
            frequency_penalty=0.8,  # Avoid repetition
            prompt="Be creative and exploratory in your responses. Use the knowledge base as inspiration for innovative ideas and connections.",
            similarity_threshold=0.1,  # Low threshold for broad relevance
            keywords_similarity_weight=0.4,  # Less emphasis on exact keywords
            top_n=15,  # Large context window
        )
        print(f"Creative Assistant: {creative.status}")

        # Configuration 3: Balanced Assistant
        print("\n3. Creating Balanced Assistant (Moderate settings)...")
        balanced = await client.create_chat_assistant(
            name="Balanced Assistant",
            description="Balanced assistant for general purpose use",
            dataset_ids=[kb_id],
            llm_model="gpt-4o",
            temperature=0.3,  # Moderate creativity
            top_p=0.6,  # Moderate diversity
            presence_penalty=0.4,  # Moderate penalty
            frequency_penalty=0.5,
            prompt="Provide helpful, balanced responses that combine accuracy with useful insights. Use the knowledge base effectively while being conversational.",
            similarity_threshold=0.25,  # Moderate threshold
            keywords_similarity_weight=0.7,  # Balanced keyword emphasis
            top_n=8,  # Moderate context window
        )
        print(f"Balanced Assistant: {balanced.status}")

        # Display configuration comparison
        print("\n=== Configuration Comparison ===")
        assistants_response = await client.list_chat_assistants()
        if assistants_response.status:
            for assistant in assistants_response.assistants:
                if assistant.name in [
                    "Factual Assistant",
                    "Creative Assistant",
                    "Balanced Assistant",
                ]:
                    print(f"\n{assistant.name}:")
                    print(f"  Use Case: {assistant.description}")
                    print(
                        f"  Temperature: {assistant.temperature} (0=deterministic, 1=creative)"
                    )
                    print(f"  Top P: {assistant.top_p} (0=focused, 1=diverse)")
                    print(
                        f"  Presence Penalty: {assistant.presence_penalty} (encourages new topics)"
                    )
                    print(
                        f"  Frequency Penalty: {assistant.frequency_penalty} (reduces repetition)"
                    )
                    print(
                        f"  Similarity Threshold: {assistant.similarity_threshold} (relevance bar)"
                    )
                    print(
                        f"  Keywords Weight: {assistant.keywords_similarity_weight} (keyword vs semantic)"
                    )
                    print(f"  Context Size: {assistant.top_n} (chunks per response)")

        # Clean up test assistants and KB
        test_assistant_ids = []
        if assistants_response.status:
            test_assistant_ids = [
                a.id
                for a in assistants_response.assistants
                if a.name
                in ["Factual Assistant", "Creative Assistant", "Balanced Assistant"]
            ]

        if test_assistant_ids:
            await client.delete_chat_assistants(test_assistant_ids)
        await client.delete_datasets([kb_id])

        print("\n✅ Configuration showcase completed and cleaned up!")

    except Exception as e:
        print(f"Showcase error: {e}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    print("=== RAGFlow Chat Assistant Management Examples ===")

    # Run main chat assistant management example
    asyncio.run(chat_assistant_management_example())

    # Run configuration showcase
    asyncio.run(assistant_configuration_showcase())
