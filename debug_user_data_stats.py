import sqlite3
import json

class Database:
    def __init__(self):
        self.db_path = "social_llm.db"
    
    async def get_user_data_stats(self, user_id: str):
        """Get comprehensive user data statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get processed data
        cursor.execute(
            "SELECT content, personality_traits FROM processed_data WHERE user_id = ?",
            (user_id,)
        )
        data_row = cursor.fetchone()
        print(f"Data row: {data_row}")
        
        # Get conversation count
        cursor.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
            (user_id,)
        )
        conv_count = cursor.fetchone()
        print(f"Conversation count: {conv_count}")
        
        # Get training status
        cursor.execute(
            "SELECT status, progress FROM training_status WHERE user_id = ?",
            (user_id,)
        )
        status_row = cursor.fetchone()
        print(f"Status row: {status_row}")
        
        # Get uploaded files
        cursor.execute(
            "SELECT COUNT(*) FROM uploaded_files WHERE user_id = ?",
            (user_id,)
        )
        files_row = cursor.fetchone()
        print(f"Files row: {files_row}")
        
        # Get training data size
        cursor.execute(
            "SELECT SUM(LENGTH(content)) FROM processed_data WHERE user_id = ?",
            (user_id,)
        )
        training_data_size_row = cursor.fetchone()
        print(f"Training data size row: {training_data_size_row}")
        training_data_size = training_data_size_row[0] if training_data_size_row and training_data_size_row[0] else 0
        print(f"Calculated training data size: {training_data_size}")
        
        # Get conversation count by date instead of model (since model_used column doesn't exist)
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) 
            FROM conversations 
            WHERE user_id = ? 
            GROUP BY DATE(created_at)
        ''', (user_id,))
        conversation_stats = cursor.fetchall()
        
        conn.close()
        
        result = {
            "user_id": user_id,
            "has_trained_model": status_row and status_row[0] == "completed",
            "training_progress": status_row[1] if status_row else 0.0,
            "training_type": "default",  # Default value since model_name is not available
            "conversation_count": conv_count[0] if conv_count else 0,
            "uploaded_files_count": files_row[0] if files_row else 0,
            "training_data_size": training_data_size,
            "personality_traits": json.loads(data_row[1]) if data_row and data_row[1] else {},
            "data_available": bool(data_row),
            "conversation_by_date": {str(row[0]): row[1] for row in conversation_stats} if conversation_stats else {}
        }
        
        print(f"Final result: {json.dumps(result, indent=2)}")
        return result

# Test the method
import asyncio

async def test():
    db = Database()
    result = await db.get_user_data_stats("demo-user-y2qjjyb4a")
    print(f"\nTraining data size from result: {result.get('training_data_size')}")

asyncio.run(test())