import aiosqlite
import asyncio
import json

async def test_database_queries():
    """Test the exact queries used by database.py"""
    db_path = 'social_llm.db'
    user_id = "demo-user-y2qjjyb4a"
    
    async with aiosqlite.connect(db_path) as db:
        print(f"Testing queries for user: {user_id}")
        
        # Test processed data query (from get_user_data_stats)
        print("\n1. Testing processed data query:")
        cursor = await db.execute(
            "SELECT content, personality_traits FROM processed_data WHERE user_id = ?",
            (user_id,)
        )
        data_row = await cursor.fetchone()
        print(f"Processed data result: {data_row}")
        
        # Test conversation count
        print("\n2. Testing conversation count:")
        cursor = await db.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
            (user_id,)
        )
        conv_count = await cursor.fetchone()
        print(f"Conversation count: {conv_count}")
        
        # Test training status
        print("\n3. Testing training status:")
        cursor = await db.execute(
            "SELECT status, progress, training_type FROM training_status WHERE user_id = ?",
            (user_id,)
        )
        status_row = await cursor.fetchone()
        print(f"Training status: {status_row}")
        
        # Test uploaded files
        print("\n4. Testing uploaded files:")
        cursor = await db.execute(
            "SELECT COUNT(*), SUM(file_size) FROM uploaded_files WHERE user_id = ?",
            (user_id,)
        )
        files_row = await cursor.fetchone()
        print(f"Uploaded files: {files_row}")
        
        # Test training data size
        print("\n5. Testing training data size:")
        cursor = await db.execute(
            "SELECT SUM(LENGTH(content)) FROM processed_data WHERE user_id = ?",
            (user_id,)
        )
        training_data_size_row = await cursor.fetchone()
        print(f"Training data size: {training_data_size_row}")
        
        # Test user knowledge
        print("\n6. Testing user knowledge:")
        cursor = await db.execute(
            "SELECT content, content_type, source, importance_score FROM user_knowledge WHERE user_id = ? ORDER BY importance_score DESC, created_at DESC LIMIT ?",
            (user_id, 100)
        )
        knowledge_results = await cursor.fetchall()
        print(f"Knowledge items count: {len(knowledge_results)}")
        print(f"Knowledge items: {knowledge_results[:2]}...")  # Show first 2
        
        # Simulate the get_user_data_stats function
        print("\n7. Simulating get_user_data_stats result:")
        training_data_size = training_data_size_row[0] if training_data_size_row and training_data_size_row[0] else 0
        
        result = {
            "user_id": user_id,
            "has_trained_model": status_row and status_row[0] == "completed",
            "training_progress": status_row[1] if status_row else 0.0,
            "training_type": status_row[2] if status_row else "none",
            "conversation_count": conv_count[0] if conv_count else 0,
            "uploaded_files_count": files_row[0] if files_row else 0,
            "total_file_size": files_row[1] if files_row else 0,
            "training_data_size": training_data_size,
            "personality_traits": json.loads(data_row[1]) if data_row and data_row[1] else {},
            "data_available": bool(data_row),
            "api_usage": {}
        }
        
        print(f"Final result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_database_queries())