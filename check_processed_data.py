import sqlite3

conn = sqlite3.connect('social_llm.db')
cursor = conn.cursor()

# Check processed data for demo user
cursor.execute('SELECT user_id, content, LENGTH(content) FROM processed_data WHERE user_id = ?', ('demo-user-y2qjjyb4a',))
rows = cursor.fetchall()

print(f"Found {len(rows)} processed data entries:")
for row in rows:
    user_id, content, length = row
    print(f"User: {user_id}")
    print(f"Content length: {length}")
    print(f"Content preview: {content[:100]}...")
    print("---")

# Check total training data size calculation
cursor.execute('SELECT SUM(LENGTH(content)) FROM processed_data WHERE user_id = ?', ('demo-user-y2qjjyb4a',))
result = cursor.fetchone()
print(f"Total training data size: {result[0]}")

conn.close()