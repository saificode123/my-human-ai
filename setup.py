import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.5.0",
        "transformers==4.36.0",
        "datasets==2.14.0",
        "torch==2.1.0",
        "spacy==3.7.2",
        "textblob==0.17.1",
        "beautifulsoup4==4.12.2",
        "requests==2.31.0",
        "aiosqlite==0.19.0",
        "python-multipart==0.0.6",
        "huggingface_hub==0.19.4",
        "accelerate==0.24.1",
        "sentencepiece==0.1.99"
    ]
    
    for requirement in requirements:
        print(f"Installing {requirement}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
    
    # Download spaCy model
    print("Downloading spaCy English model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("Setup completed successfully!")

def create_directories():
    """Create necessary directories"""
    directories = ["models", "logs", "data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directories()
    install_requirements()
