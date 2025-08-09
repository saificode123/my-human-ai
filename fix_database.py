import sqlite3
import aiosqlite
import asyncio
import os
import json
from datetime import datetime

async def initialize_database_async():
    """Initialize the database with proper schema matching database.py expectations"""
    db_path = 'social_llm.db'  # This is what database.py expects
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    async with aiosqlite.connect(db_path) as db:
        # Users table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Social profiles table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS social_profiles (
                user_id TEXT PRIMARY KEY,
                linkedin_url TEXT,
                facebook_url TEXT,
                youtube_channel_id TEXT,
                processed_data TEXT,
                personality_traits TEXT,
                communication_patterns TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Profiles table (fallback)
        await db.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY,
                linkedin_url TEXT,
                facebook_url TEXT,
                youtube_channel_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Training status table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS training_status (
                user_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'pending',
                progress REAL DEFAULT 0.0,
                message TEXT,
                metrics TEXT,
                training_type TEXT DEFAULT 'social_media',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Processed data table (this is what database.py expects)
        await db.execute('''
            CREATE TABLE IF NOT EXISTS processed_data (
                user_id TEXT,
                data_type TEXT,
                content TEXT,
                personality_traits TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Enhanced conversations table with API tracking
        await db.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_message TEXT,
                bot_response TEXT,
                model_used TEXT,
                api_provider TEXT,
                response_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Uploaded files table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                filename TEXT,
                file_path TEXT,
                file_size INTEGER,
                file_type TEXT,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # User knowledge table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS user_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                content TEXT,
                content_type TEXT,
                source TEXT,
                importance_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Model performance table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                model_type TEXT,
                performance_metrics TEXT,
                training_duration REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        await db.commit()
        print("Database tables created successfully")
        
        # Create demo user
        user_id = "demo-user-y2qjjyb4a"
        await db.execute("INSERT OR REPLACE INTO users (user_id) VALUES (?)", (user_id,))
        
        # Add initial training status
        await db.execute("""
            INSERT OR REPLACE INTO training_status 
            (user_id, status, progress, message, training_type) 
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, "pending", 0.0, "Ready to start training", "social_media"))
        
        # Add processed data (this is what the API looks for)
        sample_content = "User is interested in AI and machine learning technologies. User has experience with Python programming. User enjoys working on innovative projects. User values continuous learning and development."
        personality_traits = {
            "enthusiasm": 0.8,
            "technical_focus": 0.9,
            "formality": 0.6,
            "creativity": 0.7
        }
        
        await db.execute("""
            INSERT INTO processed_data 
            (user_id, data_type, content, personality_traits) 
            VALUES (?, ?, ?, ?)
        """, (user_id, "biography", sample_content, json.dumps(personality_traits)))
        
        # Add some sample knowledge
        sample_knowledge = [
            "User is interested in AI and machine learning technologies",
            "User has experience with Python programming",
            "User enjoys working on innovative projects",
            "User values continuous learning and development"
        ]
        
        for knowledge in sample_knowledge:
            await db.execute("""
                INSERT INTO user_knowledge 
                (user_id, content, content_type, source, importance_score) 
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, knowledge, "fact", "biography.pdf", 0.8))
        
        # Add uploaded file record
        await db.execute("""
            INSERT INTO uploaded_files 
            (user_id, filename, file_path, file_size, file_type, config) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, "biography.pdf", "uploads/6a4c2545-969c-4544-85fb-54ab096aff89_biography.pdf", 
               1024, "application/pdf", "{}"))
        
        await db.commit()
        
        print(f"Demo user '{user_id}' created with sample data")
        print("Database initialization complete!")

def initialize_database():
    """Synchronous wrapper for async database initialization"""
    asyncio.run(initialize_database_async())

if __name__ == "__main__":
    initialize_database()