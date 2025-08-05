# ================================
# database.py - Enhanced SQLite Database Manager
# ================================

import sqlite3
import aiosqlite
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

class Database:
    def __init__(self, db_path: str = "social_llm.db"):
        self.db_path = db_path
    
    async def initialize(self):
        """Initialize database with all required tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Users table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Profiles table (for social media profiles)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT,
                    linkedin_url TEXT,
                    facebook_url TEXT,
                    youtube_channel_id TEXT,
                    model_name TEXT DEFAULT 'distilgpt2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            
            # Uploaded files table (for custom training data)
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
            
            # Processed data table
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
    
    async def close(self):
        """Close database connection"""
        pass  # aiosqlite handles connection automatically
    
    async def create_user(self, user_id: str):
        """Create a new user"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO users (user_id) VALUES (?)", 
                (user_id,)
            )
            await db.execute(
                "INSERT INTO training_status (user_id, status, message) VALUES (?, ?, ?)",
                (user_id, "pending", "Waiting for training data")
            )
            await db.commit()
    
    async def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM users WHERE user_id = ?", 
                (user_id,)
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "user_id": row[0],
                    "created_at": row[1],
                    "updated_at": row[2]
                }
            return None
    
    async def save_profile(self, user_id: str, profile_data: Dict):
        """Save user social media profile data"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO profiles 
                (user_id, linkedin_url, facebook_url, youtube_channel_id, model_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                str(profile_data.get('linkedin_url', '')),
                str(profile_data.get('facebook_url', '')),
                profile_data.get('youtube_channel_id', ''),
                profile_data.get('model_name', 'distilgpt2')
            ))
            await db.commit()
    
    async def save_uploaded_file(self, user_id: str, filename: str, file_path: str, config: Dict):
        """Save uploaded file information"""
        import os
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        file_type = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO uploaded_files 
                (user_id, filename, file_path, file_size, file_type, config)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id, filename, file_path, file_size, file_type, json.dumps(config)
            ))
            await db.commit()
    
    async def get_uploaded_file(self, user_id: str) -> Optional[Dict]:
        """Get user's uploaded file information"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                SELECT * FROM uploaded_files 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (user_id,))
            row = await cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "user_id": row[1],
                    "filename": row[2],
                    "file_path": row[3],
                    "file_size": row[4],
                    "file_type": row[5],
                    "config": json.loads(row[6]) if row[6] else {},
                    "created_at": row[7]
                }
            return None
    
    async def update_training_status(self, user_id: str, status: str, progress: float, message: str, metrics: Optional[Dict] = None, training_type: str = "custom"):
        """Update training status with enhanced tracking"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                UPDATE training_status 
                SET status = ?, progress = ?, message = ?, metrics = ?, training_type = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (status, progress, message, json.dumps(metrics) if metrics else None, training_type, user_id))
            await db.commit()
    
    async def get_training_status(self, user_id: str) -> Optional[Dict]:
        """Get training status"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM training_status WHERE user_id = ?", 
                (user_id,)
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "user_id": row[0],
                    "status": row[1],
                    "progress": row[2],
                    "message": row[3],
                    "metrics": json.loads(row[4]) if row[4] else None,
                    "training_type": row[5],
                    "updated_at": row[6]
                }
            return None
    
    async def save_processed_data(self, user_id: str, data: Dict, personality_traits: Dict):
        """Save processed data and personality traits"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO processed_data 
                (user_id, data_type, content, personality_traits)
                VALUES (?, ?, ?, ?)
            ''', (
                user_id,
                "combined",
                json.dumps(data),
                json.dumps(personality_traits)
            ))
            await db.commit()
    
    async def log_conversation(self, user_id: str, user_message: str, bot_response: str, api_provider: Optional[str] = None, response_time: Optional[float] = None):
        """Log conversation with API tracking"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO conversations (user_id, user_message, bot_response, model_used, api_provider, response_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, user_message, bot_response, api_provider or "local", api_provider, response_time))
            await db.commit()
    
    async def get_conversations(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get conversations for retraining"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                SELECT user_message, bot_response, model_used, api_provider, created_at 
                FROM conversations 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            rows = await cursor.fetchall()
            return [
                {
                    "user_message": row[0],
                    "bot_response": row[1],
                    "model_used": row[2],
                    "api_provider": row[3],
                    "created_at": row[4]
                }
                for row in rows
            ]
    
    async def save_model_performance(self, user_id: str, model_type: str, metrics: Dict, training_duration: float):
        """Save model performance metrics"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO model_performance 
                (user_id, model_type, performance_metrics, training_duration)
                VALUES (?, ?, ?, ?)
            ''', (user_id, model_type, json.dumps(metrics), training_duration))
            await db.commit()
    
    async def get_user_data_stats(self, user_id: str) -> Dict:
        """Get comprehensive user data statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            # Get processed data
            cursor = await db.execute(
                "SELECT content, personality_traits FROM processed_data WHERE user_id = ?",
                (user_id,)
            )
            data_row = await cursor.fetchone()
            
            # Get conversation count
            cursor = await db.execute(
                "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
                (user_id,)
            )
            conv_count = await cursor.fetchone()
            
            # Get training status
            cursor = await db.execute(
                "SELECT status, progress, training_type FROM training_status WHERE user_id = ?",
                (user_id,)
            )
            status_row = await cursor.fetchone()
            
            # Get uploaded files
            cursor = await db.execute(
                "SELECT COUNT(*), SUM(file_size) FROM uploaded_files WHERE user_id = ?",
                (user_id,)
            )
            files_row = await cursor.fetchone()
            
            # Get API usage stats
            cursor = await db.execute('''
                SELECT api_provider, COUNT(*) 
                FROM conversations 
                WHERE user_id = ? AND api_provider IS NOT NULL 
                GROUP BY api_provider
            ''', (user_id,))
            api_stats = await cursor.fetchall()
            
            return {
                "user_id": user_id,
                "has_trained_model": status_row and status_row[0] == "completed",
                "training_progress": status_row[1] if status_row else 0.0,
                "training_type": status_row[2] if status_row else "none",
                "conversation_count": conv_count[0] if conv_count else 0,
                "uploaded_files_count": files_row[0] if files_row else 0,
                "total_file_size": files_row[1] if files_row else 0,
                "personality_traits": json.loads(data_row[1]) if data_row and data_row[1] else {},
                "data_available": bool(data_row),
                "api_usage": {row[0]: row[1] for row in api_stats} if api_stats else {}
            }
    
    async def get_all_users_stats(self) -> Dict:
        """Get platform-wide statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            # Total users
            cursor = await db.execute("SELECT COUNT(*) FROM users")
            total_users = await cursor.fetchone()
            
            # Active trainings
            cursor = await db.execute("SELECT COUNT(*) FROM training_status WHERE status = 'training'")
            active_trainings = await cursor.fetchone()
            
            # Completed models
            cursor = await db.execute("SELECT COUNT(*) FROM training_status WHERE status = 'completed'")
            completed_models = await cursor.fetchone()
            
            # Total conversations
            cursor = await db.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = await cursor.fetchone()
            
            return {
                "total_users": total_users[0] if total_users else 0,
                "active_trainings": active_trainings[0] if active_trainings else 0,
                "completed_models": completed_models[0] if completed_models else 0,
                "total_conversations": total_conversations[0] if total_conversations else 0
            }