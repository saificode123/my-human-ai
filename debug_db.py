import sqlite3
import aiosqlite
import asyncio
import os

async def debug_database_async():
    db_path = 'social_llm.db'
    if os.path.exists(db_path):
        async with aiosqlite.connect(db_path) as db:
            # Check tables
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = await cursor.fetchall()
            print('Tables:', [t[0] for t in tables])
            
            # Check training status
            try:
                cursor = await db.execute('SELECT * FROM training_status LIMIT 5;')
                training_status = await cursor.fetchall()
                print('Training status:', training_status)
            except Exception as e:
                print('Training status error:', e)
            
            # Check processed_data (this is what database.py looks for)
            try:
                cursor = await db.execute('SELECT user_id, data_type, LENGTH(content), personality_traits FROM processed_data;')
                processed_data = await cursor.fetchall()
                print('Processed data:', processed_data)
            except Exception as e:
                print('Processed data error:', e)
            
            # Check user knowledge
            try:
                cursor = await db.execute('SELECT user_id, COUNT(*) FROM user_knowledge GROUP BY user_id;')
                user_knowledge = await cursor.fetchall()
                print('User knowledge counts:', user_knowledge)
            except Exception as e:
                print('User knowledge error:', e)
            
            # Check uploaded files
            try:
                cursor = await db.execute('SELECT user_id, filename, file_size FROM uploaded_files LIMIT 5;')
                uploaded_files = await cursor.fetchall()
                print('Uploaded files:', uploaded_files)
            except Exception as e:
                print('Uploaded files error:', e)
            
            # Check conversations
            try:
                cursor = await db.execute('SELECT COUNT(*) FROM conversations;')
                conv_count = await cursor.fetchall()
                print('Conversation count:', conv_count)
            except Exception as e:
                print('Conversations error:', e)
    else:
        print('Database file does not exist')

def debug_database():
    asyncio.run(debug_database_async())

if __name__ == "__main__":
    debug_database()