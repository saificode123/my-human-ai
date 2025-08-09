import sqlite3
import os

def check_ai_training_db():
    db_path = 'ai_training.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print('Tables in ai_training.db:', [t[0] for t in tables])
        
        # Check if demo user exists
        try:
            cursor.execute('SELECT * FROM users LIMIT 5;')
            users = cursor.fetchall()
            print('Users:', users)
        except Exception as e:
            print('Users error:', e)
        
        # Check training status
        try:
            cursor.execute('SELECT * FROM training_status LIMIT 5;')
            training_status = cursor.fetchall()
            print('Training status:', training_status)
        except Exception as e:
            print('Training status error:', e)
            
        conn.close()
    else:
        print('ai_training.db does not exist')

if __name__ == "__main__":
    check_ai_training_db()