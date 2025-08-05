import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api():
    """Test the main API endpoints"""
    
    print("Testing Social Profile LLM Training API...")
    
    # 1. Create user
    print("\n1. Creating user...")
    response = requests.post(f"{BASE_URL}/api/users")
    user_data = response.json()
    user_id = user_data["user_id"]
    print(f"Created user: {user_id}")
    
    # 2. Submit profile
    print("\n2. Submitting profile...")
    profile_data = {
        "user_id": user_id,
        "linkedin_url": "https://www.linkedin.com/in/zusmani/",
        "facebook_url": "https://www.facebook.com/zusmani",
        "youtube_channel_id": "UCllefjGak7WtAV3sVcRy9xQ",
        "model_name": "distilgpt2"
    }
    
    response = requests.post(f"{BASE_URL}/api/profiles", json=profile_data)
    print(f"Profile submitted: {response.json()}")
    
    # 3. Check training status
    print("\n3. Checking training status...")
    for i in range(10):  # Check for up to 10 iterations
        response = requests.get(f"{BASE_URL}/api/training/status/{user_id}")
        status_data = response.json()
        print(f"Status: {status_data['status']} - {status_data['progress']:.1f}% - {status_data['message']}")
        
        if status_data['status'] == 'completed':
            break
        elif status_data['status'] == 'error':
            print("Training failed!")
            return
        
        time.sleep(30)  # Wait 30 seconds between checks
    
    # 4. Test chat
    print("\n4. Testing chat...")
    chat_data = {
        "user_id": user_id,
        "message": "Hello! Tell me about yourself."
    }
    
    response = requests.post(f"{BASE_URL}/api/chat", json=chat_data)
    chat_response = response.json()
    print(f"Chat response: {chat_response['response']}")
    
    # 5. Get user data stats
    print("\n5. Getting user data stats...")
    response = requests.get(f"{BASE_URL}/api/users/{user_id}/data")
    stats = response.json()
    print(f"User stats: {json.dumps(stats, indent=2)}")
    
    print("\nAPI testing completed!")

if __name__ == "__main__":
    test_api()