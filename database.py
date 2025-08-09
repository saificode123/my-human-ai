import sqlite3
import aiosqlite
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = "human_clone.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User profiles table (social media URLs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    linkedin_url TEXT,
                    facebook_url TEXT,
                    youtube_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Uploaded files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    file_type TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Training status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE,
                    status TEXT DEFAULT 'not_started',
                    progress INTEGER DEFAULT 0,
                    total_files INTEGER DEFAULT 0,
                    processed_files INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    embedding_model TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
            

    
    async def create_user(self, name: str, email: str) -> int:
        """Create a new user and return user ID"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                (name, email)
            )
            await conn.commit()
            user_id = cursor.lastrowid
            
            # Initialize training status
            await cursor.execute(
                "INSERT INTO training_status (user_id) VALUES (?)",
                (user_id,)
            )
            await conn.commit()
            
            logger.info(f"Created user: {name} ({email}) with ID: {user_id}")
            return user_id
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "SELECT id, name, email, created_at FROM users WHERE id = ?",
                (user_id,)
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "email": row[2],
                    "created_at": row[3]
                }
            return None
    
    async def save_uploaded_file(self, user_id: int, filename: str, file_path: str, file_size: int, file_type: str):
        """Save uploaded file information"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "INSERT INTO uploaded_files (user_id, filename, file_path, file_size, file_type) VALUES (?, ?, ?, ?, ?)",
                (user_id, filename, file_path, file_size, file_type)
            )
            await conn.commit()
            logger.info(f"Saved file info: {filename} ({file_size} bytes) for user {user_id}")
    
    async def get_user_files(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all files for a user"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "SELECT filename, file_path, file_size, file_type, uploaded_at FROM uploaded_files WHERE user_id = ?",
                (user_id,)
            )
            rows = await cursor.fetchall()
            return [
                {
                    "filename": row[0],
                    "file_path": row[1],
                    "file_size": row[2],
                    "file_type": row[3],
                    "uploaded_at": row[4]
                }
                for row in rows
            ]
    
    async def update_training_status(self, user_id: int, status: str, progress: int = None, 
                                   total_files: int = None, processed_files: int = None,
                                   total_chunks: int = None, embedding_model: str = None,
                                   error_message: str = None):
        """Update training status for a user"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            
            # Build dynamic update query
            updates = ["status = ?"]
            params = [status]
            
            if progress is not None:
                updates.append("progress = ?")
                params.append(progress)
            
            if total_files is not None:
                updates.append("total_files = ?")
                params.append(total_files)
            
            if processed_files is not None:
                updates.append("processed_files = ?")
                params.append(processed_files)
            
            if total_chunks is not None:
                updates.append("total_chunks = ?")
                params.append(total_chunks)
            
            if embedding_model is not None:
                updates.append("embedding_model = ?")
                params.append(embedding_model)
            
            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)
            
            if status == "in_progress" and "started_at" not in updates:
                updates.append("started_at = CURRENT_TIMESTAMP")
            elif status == "completed":
                updates.append("completed_at = CURRENT_TIMESTAMP")
            
            params.append(user_id)
            
            query = f"UPDATE training_status SET {', '.join(updates)} WHERE user_id = ?"
            await cursor.execute(query, params)
            await conn.commit()
    
    async def get_training_status(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get training status for a user"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "SELECT status, progress, total_files, processed_files, total_chunks, embedding_model, started_at, completed_at, error_message FROM training_status WHERE user_id = ?",
                (user_id,)
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "status": row[0],
                    "progress": row[1],
                    "total_files": row[2],
                    "processed_files": row[3],
                    "total_chunks": row[4],
                    "embedding_model": row[5],
                    "started_at": row[6],
                    "completed_at": row[7],
                    "error_message": row[8]
                }
            return None
    
    async def save_conversation(self, user_id: int, message: str, response: str, context_used: str = None):
        """Save a conversation"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "INSERT INTO conversations (user_id, message, response, context_used) VALUES (?, ?, ?, ?)",
                (user_id, message, response, context_used)
            )
            await conn.commit()
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user statistics"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            
            # Get file count
            await cursor.execute("SELECT COUNT(*) FROM uploaded_files WHERE user_id = ?", (user_id,))
            file_count = (await cursor.fetchone())[0]
            
            # Get conversation count
            await cursor.execute("SELECT COUNT(*) FROM conversations WHERE user_id = ?", (user_id,))
            conversation_count = (await cursor.fetchone())[0]
            
            # Get training status
            training_status = await self.get_training_status(user_id)
            
            return {
                "file_count": file_count,
                "conversation_count": conversation_count,
                "training_status": training_status
            }