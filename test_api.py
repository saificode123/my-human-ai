import requests
import json

base_url = "http://localhost:8000"
user_id = "demo-user-y2qjjyb4a"

try:
    # Test user data endpoint
    response = requests.get(f"{base_url}/api/users/{user_id}/data")
    print(f"User data status: {response.status_code}")
    print(f"User data response: {response.text}")
    
    # Test training status endpoint
    response = requests.get(f"{base_url}/api/training/status/{user_id}")
    print(f"Training status: {response.status_code}")
    print(f"Training response: {response.text}")
    
    # Test chat endpoint
    chat_data = {
        "user_id": user_id,
        "message": "Hello, how are you?",
        "use_context": True
    }
    response = requests.post(f"{base_url}/api/chat", json=chat_data)
    print(f"Chat status: {response.status_code}")
    print(f"Chat response: {response.text}")
    
except Exception as e:
    print(f"Error: {e}")