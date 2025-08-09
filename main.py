# main.py - Complete Human Clone AI Training Platform
import os
import sys
import json
import uuid
import asyncio
import aiofiles
import logging
import requests
import re
import zipfile
import aiohttp
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# Document processing libraries
import PyPDF2
from docx import Document

# ChromaDB and embeddings
import chromadb
from chromadb.config import Settings as ChromaSettings
# Lazy import for sentence_transformers to avoid startup issues
# from sentence_transformers import SentenceTransformer # Will be imported dynamically

# YouTube integration
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse # Import JSONResponse for error handling
from pydantic import BaseModel, ConfigDict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False
    logger.warning("⚠️ OpenAI library not found. OpenAI polishing feature will be disabled.")


# --- Configuration ---
class Settings:
    """Manages application settings loaded from environment variables."""
    def __init__(self):
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./human_clone.db") # Not used directly for training status, but kept for potential future DB integration
        self.CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chromadb")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")
        self.YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
        self.MODELS_DIR = os.getenv("MODELS_DIR", "./data/models")
        self.UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./data/uploads")
        self.SOCIAL_EXPORTS_DIR = os.getenv("SOCIAL_EXPORTS_DIR", "./data/social_exports")
        
        # Create necessary directories upon initialization
        for directory in [self.MODELS_DIR, self.UPLOADS_DIR, self.SOCIAL_EXPORTS_DIR, 
                         self.CHROMA_PERSIST_DIRECTORY, "logs"]:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

settings = Settings()

# --- Pydantic Models ---
class UserProfile(BaseModel):
    """Represents a user's profile with social media URLs."""
    model_config = ConfigDict(protected_namespaces=())
    user_id: str
    linkedin_url: Optional[str] = None
    facebook_url: Optional[str] = None
    youtube_channel_id: Optional[str] = None

class ChatMessage(BaseModel):
    """Represents an incoming chat message."""
    user_id: str
    message: str
    use_context: Optional[bool] = True
    use_openai_polish: Optional[bool] = False

class ChatResponse(BaseModel):
    """Represents an outgoing chat response."""
    model_config = ConfigDict(protected_namespaces=())
    response: str
    model_used: str
    context_used: Optional[str] = None
    similarity_score: Optional[float] = None
    sources: Optional[List[str]] = None

class TrainingStatus(BaseModel):
    """Represents the real-time status of a user's training process."""
    model_config = ConfigDict(protected_namespaces=())
    user_id: str
    status: str
    progress: float
    message: str
    social_data_files: Optional[List[Dict[str, Any]]] = None
    uploaded_files: Optional[List[str]] = None
    total_chunks: Optional[int] = None
    embedding_model: Optional[str] = None

class StartTrainingRequest(BaseModel):
    """Request model for starting the training process."""
    model_config = ConfigDict(protected_namespaces=()) # Ensure this is present
    user_id: str
    include_social_media: Optional[bool] = True
    model_name: Optional[str] = None 

class SocialMediaRequest(BaseModel):
    """Request model for explicit social media data extraction."""
    model_config = ConfigDict(protected_namespaces=()) # Ensure this is present
    user_id: str
    linkedin_export_file: Optional[str] = None # Path to the uploaded file on the backend
    facebook_access_token: Optional[str] = None
    youtube_channel_id: Optional[str] = None

# --- ChromaDB Embedding Manager ---
class ChromaEmbeddingManager:
    """Manages ChromaDB collections and embedding operations."""
    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.model_name = settings.EMBEDDING_MODEL
        
    async def initialize(self):
        """Initializes ChromaDB client and sets up the embedding model."""
        try:
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True # Allows resetting the DB, useful for development
                )
            )
            logger.info(f"✅ ChromaDB client initialized. Persist directory: {settings.CHROMA_PERSIST_DIRECTORY}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _get_embedding_model(self):
        """Lazy-loads the SentenceTransformer embedding model."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info(f"✅ Embedding model loaded: {self.model_name}")
            except ImportError:
                logger.critical("❌ 'sentence-transformers' library not found. Please install it to enable embeddings.")
                raise
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model '{self.model_name}': {str(e)}")
                raise
        return self.embedding_model
    
    async def get_or_create_collection(self, user_id: str):
        """Retrieves or creates a ChromaDB collection for a given user."""
        try:
            collection_name = f"user_{user_id}"
            
            # Ensure embedding model is loaded before creating/getting collection
            self._get_embedding_model() 

            try:
                # Attempt to get existing collection
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.model_name
                    )
                )
                logger.info(f"📚 Retrieved existing ChromaDB collection for user {user_id}")
            except Exception as e: # This handles cases where collection doesn't exist or other retrieval errors
                logger.info(f"Collection '{collection_name}' not found or error retrieving. Creating new. Error: {e}")
                # Create new collection if it doesn't exist or if retrieval failed
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.model_name
                    )
                )
                logger.info(f"🆕 Created new ChromaDB collection for user {user_id}")
            
            return collection
            
        except Exception as e:
            logger.error(f"❌ Error managing ChromaDB collection for user {user_id}: {str(e)}")
            raise
    
    async def add_documents(self, user_id: str, documents: List[str], metadatas: List[Dict], source: str) -> int:
        """Adds documents (text chunks) to the user's ChromaDB collection."""
        try:
            collection = await self.get_or_create_collection(user_id)
            
            # Generate unique IDs for documents
            ids = [f"{source}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(documents))]
            
            # Add documents to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"📝 Added {len(documents)} documents from {source} to user {user_id}'s collection")
            return len(documents)
            
        except Exception as e:
            logger.error(f"❌ Error adding documents for user {user_id}: {str(e)}")
            raise
    
    async def similarity_search(self, user_id: str, query: str, n_results: int = 5) -> List[Dict]:
        """Performs a similarity search in the user's collection to find relevant documents."""
        try:
            collection = await self.get_or_create_collection(user_id)
            
            # Check if collection is empty
            if collection.count() == 0:
                logger.info(f"🔎 No documents in collection for user {user_id}. Cannot perform search.")
                return []

            # Perform similarity search
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    })
            
            logger.debug(f"🔎 Found {len(formatted_results)} results for query: '{query[:50]}...'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error searching documents for user {user_id}: {str(e)}")
            return []
    
    async def get_collection_count(self, user_id: str) -> int:
        """Gets the number of documents in a user's collection."""
        try:
            collection = await self.get_or_create_collection(user_id)
            return collection.count()
        except Exception as e:
            logger.warning(f"⚠️ Could not get collection count for user {user_id}: {str(e)}")
            return 0 # Return 0 if collection doesn't exist or an error occurs

# --- Social Media Data Extractor ---
class SocialMediaExtractor:
    """Handles extraction of data from various social media platforms."""
    def __init__(self):
        self.facebook_token = settings.FACEBOOK_ACCESS_TOKEN
        self.youtube_api_key = settings.YOUTUBE_API_KEY
        self.social_exports_dir = settings.SOCIAL_EXPORTS_DIR
    
    async def extract_linkedin_data(self, user_id: str, linkedin_file_path: str) -> Dict[str, Any]:
        """Extracts data from a LinkedIn export file (ZIP or plain text)."""
        logger.info(f"🔍 Extracting LinkedIn data for user {user_id} from {linkedin_file_path}")
        
        extracted_data = []
        file_path = Path(linkedin_file_path)
        
        try:
            if file_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(linkedin_file_path, 'r') as zip_ref:
                    # Look for common LinkedIn export files
                    for file_name in zip_ref.namelist():
                        if file_name.endswith('.csv') and any(keyword in file_name.lower() 
                                                             for keyword in ['profile', 'experience', 'education', 'skills', 'messages']):
                            try:
                                with zip_ref.open(file_name) as csv_file:
                                    content = csv_file.read().decode('utf-8')
                                    extracted_data.append(f"LinkedIn {file_name}:\n{content}\n")
                                logger.info(f"✅ Extracted '{file_name}' from LinkedIn ZIP for user {user_id}")
                            except Exception as e:
                                logger.warning(f"⚠️ Could not read '{file_name}' from LinkedIn ZIP: {e}")
            else:
                # Assume it's a plain text file if not a ZIP
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    extracted_data.append(f"LinkedIn Data from {file_path.name}:\n{content}\n")
                logger.info(f"✅ Extracted data from plain LinkedIn file for user {user_id}")

            if not extracted_data:
                logger.warning(f"⚠️ No relevant data extracted from LinkedIn file: {linkedin_file_path}")
                raise ValueError("No relevant data found in LinkedIn export.")
            
            # Save extracted data to a single file
            output_file = os.path.join(self.social_exports_dir, f"{user_id}_linkedin_data.txt")
            await self._save_social_data(output_file, extracted_data, "LinkedIn")
            
            file_size = os.path.getsize(output_file)
            logger.info(f"💾 LinkedIn data extracted and saved: {output_file} ({file_size} bytes)")
            
            return {
                'platform': 'LinkedIn',
                'file_path': output_file,
                'file_size': file_size,
                'data_count': len(extracted_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting LinkedIn data from {linkedin_file_path}: {str(e)}")
            raise

    async def extract_facebook_data(self, user_id: str, access_token: str = None) -> Dict[str, Any]:
        """Extracts data from Facebook using the Graph API (requires user token)."""
        token = access_token or self.facebook_token
        if not token:
            logger.error("❌ Facebook access token not provided. Cannot extract Facebook data.")
            raise ValueError("Facebook access token is required to extract Facebook data.")
        
        logger.info(f"🔍 Attempting to extract Facebook data for user {user_id}")
        
        extracted_data = []
        
        try:
            # Get user's posts
            # We request 'posts' and 'feed' which includes posts made by user and posts on user's timeline.
            # fields=message,created_time,story is important. 'story' is for shares without message.
            posts_url = f"https://graph.facebook.com/v18.0/me/posts?access_token={token}&fields=message,created_time,story,full_picture,link&limit=100"
            
            async with aiohttp.ClientSession() as session:
                while posts_url: # Handle pagination
                    async with session.get(posts_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            for post in data.get('data', []):
                                post_text = ""
                                if 'message' in post:
                                    post_text += post['message']
                                if 'story' in post:
                                    if post_text: post_text += " | "
                                    post_text += f"Story: {post['story']}"
                                if post_text:
                                    extracted_data.append(f"Facebook Post ({post.get('created_time', 'Unknown')}): {post_text}")
                            posts_url = data.get('paging', {}).get('next') # Get next page URL
                        else:
                            error_msg = await response.text()
                            logger.error(f"❌ Facebook Graph API error for user {user_id}: HTTP {response.status} - {error_msg}")
                            raise HTTPException(status_code=response.status, detail=f"Facebook API error: {error_msg}")
            
            if not extracted_data:
                logger.warning(f"⚠️ No Facebook data extracted for user {user_id}. User might not have public posts or permissions are insufficient.")
                # Don't raise error if no data found, just return empty, it might be expected for some users
                return {
                    'platform': 'Facebook',
                    'file_path': 'N/A',
                    'file_size': 0,
                    'data_count': 0
                }
            
            # Save extracted data
            output_file = os.path.join(self.social_exports_dir, f"{user_id}_facebook_data.txt")
            await self._save_social_data(output_file, extracted_data, "Facebook")
            
            file_size = os.path.getsize(output_file)
            logger.info(f"💾 Facebook data extracted and saved: {output_file} ({file_size} bytes)")
            
            return {
                'platform': 'Facebook',
                'file_path': output_file,
                'file_size': file_size,
                'data_count': len(extracted_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting Facebook data for user {user_id}: {str(e)}")
            raise

    async def extract_youtube_data(self, user_id: str, channel_id: str) -> Dict[str, Any]:
        """Extracts transcripts from YouTube videos within a given channel or a single video ID."""
        logger.info(f"🔍 Extracting YouTube data for user {user_id} from channel/video ID: {channel_id}")
        
        extracted_data = []
        
        if not self.youtube_api_key:
            logger.warning("⚠️ YouTube API key not provided. Attempting to extract single video transcript (if channel_id is actually a video ID).")
            # If no API key, try to treat channel_id as a direct video ID
            try:
                transcript = YouTubeTranscriptApi.get_transcript(channel_id)
                transcript_text = ' '.join([entry['text'] for entry in transcript])
                extracted_data.append(f"YouTube Video Transcript (ID: {channel_id}): {transcript_text}")
                logger.info(f"📹 Extracted transcript for single video: {channel_id}")
            except NoTranscriptFound:
                logger.warning(f"⚠️ No transcript found for video ID: {channel_id}")
            except TranscriptsDisabled:
                logger.warning(f"⚠️ Transcripts are disabled for video ID: {channel_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not extract transcript for video ID {channel_id}: {str(e)}")
            
        else: # YouTube API key is available, proceed with channel videos
            try:
                # First, get upload playlist ID for the channel
                channel_url = f"https://www.googleapis.com/youtube/v3/channels?key={self.youtube_api_key}&id={channel_id}&part=contentDetails"
                uploads_playlist_id = None
                async with aiohttp.ClientSession() as session:
                    async with session.get(channel_url) as response:
                        if response.status == 200:
                            channel_data = await response.json()
                            uploads_playlist_id = channel_data['items'][0]['contentDetails']['relatedPlaylists']['uploads']
                            logger.info(f"Found uploads playlist ID for channel {channel_id}: {uploads_playlist_id}")
                        else:
                            error_msg = await response.text()
                            logger.error(f"❌ YouTube Channel API error for {channel_id}: HTTP {response.status} - {error_msg}")
                            raise HTTPException(status_code=response.status, detail=f"YouTube Channel API error: {error_msg}")

                if uploads_playlist_id:
                    # Get videos from the uploads playlist
                    videos_url = f"https://www.googleapis.com/youtube/v3/playlistItems?key={self.youtube_api_key}&playlistId={uploads_playlist_id}&part=snippet&maxResults=10" # Limit to 10 videos for demo
                    
                    async with aiohttp.ClientSession() as session:
                        while videos_url:
                            async with session.get(videos_url) as response:
                                if response.status == 200:
                                    playlist_data = await response.json()
                                    for item in playlist_data.get('items', []):
                                        video_id = item['snippet']['resourceId']['videoId']
                                        video_title = item['snippet']['title']
                                        
                                        try:
                                            transcript = YouTubeTranscriptApi.get_transcript(video_id)
                                            transcript_text = ' '.join([entry['text'] for entry in transcript])
                                            extracted_data.append(f"YouTube Video '{video_title}' (ID: {video_id}): {transcript_text}")
                                            logger.info(f"📹 Extracted transcript for: {video_title}")
                                        except NoTranscriptFound:
                                            logger.warning(f"⚠️ No transcript found for video: {video_title} (ID: {video_id})")
                                        except TranscriptsDisabled:
                                            logger.warning(f"⚠️ Transcripts disabled for video: {video_title} (ID: {video_id})")
                                        except Exception as e:
                                            logger.warning(f"⚠️ Could not extract transcript for video '{video_title}' (ID: {video_id}): {str(e)}")
                                    videos_url = playlist_data.get('nextPageToken') # For pagination
                                    if videos_url: # Construct next page URL
                                        videos_url = f"https://www.googleapis.com/youtube/v3/playlistItems?key={self.youtube_api_key}&playlistId={uploads_playlist_id}&part=snippet&maxResults=10&pageToken={videos_url}"
                                else:
                                    error_msg = await response.text()
                                    logger.error(f"❌ YouTube Playlist API error for {uploads_playlist_id}: HTTP {response.status} - {error_msg}")
                                    raise HTTPException(status_code=response.status, detail=f"YouTube Playlist API error: {error_msg}")
            except Exception as e:
                logger.error(f"❌ Error during YouTube channel video extraction for {channel_id}: {str(e)}")
                # Continue without raising, as single video might still be attempted or user might not have videos
        
        if not extracted_data:
            logger.warning(f"⚠️ No YouTube data extracted for user {user_id} from channel/video ID: {channel_id}.")
            return {
                'platform': 'YouTube',
                'file_path': 'N/A',
                'file_size': 0,
                'data_count': 0
            }

        # Save extracted data
        output_file = os.path.join(self.social_exports_dir, f"{user_id}_youtube_data.txt")
        await self._save_social_data(output_file, extracted_data, "YouTube")
        
        file_size = os.path.getsize(output_file)
        logger.info(f"💾 YouTube data extracted and saved: {output_file} ({file_size} bytes)")
        
        return {
            'platform': 'YouTube',
            'file_path': output_file,
            'file_size': file_size,
            'data_count': len(extracted_data)
        }
            
    async def _save_social_data(self, file_path: str, data: List[str], platform: str):
        """Helper to save extracted social media data to a text file."""
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(f"# {platform} Data Export\n")
                await f.write(f"# Extracted on: {datetime.now().isoformat()}\n\n")
                
                for i, item in enumerate(data, 1):
                    await f.write(f"## Entry {i}\n{item}\n\n")
            
            logger.info(f"💾 Saved {len(data)} {platform} entries to {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving {platform} data to {file_path}: {str(e)}")
            raise

# --- Document Processor ---
class DocumentProcessor:
    """Handles processing various document types into text chunks."""
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 50
    
    async def process_file(self, file_path: str, filename: str) -> Tuple[List[str], int]:
        """Processes an uploaded file and returns text chunks."""
        logger.info(f"📄 Processing file: {filename}")
        
        file_extension = Path(filename).suffix.lower()
        content = ""
        
        try:
            if file_extension == '.txt':
                content = await self._process_txt_file(file_path)
            elif file_extension == '.pdf':
                content = await self._process_pdf_file(file_path)
            elif file_extension == '.docx':
                content = await self._process_docx_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Create chunks
            chunks = self._create_chunks(content)
            
            logger.info(f"✅ Processed {filename}: {len(chunks)} chunks created")
            return chunks, len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Error processing file {filename}: {str(e)}")
            raise
    
    async def _process_txt_file(self, file_path: str) -> str:
        """Reads content from a plain text file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()
    
    async def _process_pdf_file(self, file_path: str) -> str:
        """Extracts text content from a PDF file."""
        content = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text: # Ensure there's actual text
                        content += page_text + "\n"
        except Exception as e:
            logger.error(f"❌ Error reading PDF file {file_path}: {str(e)}")
            raise
        return content
    
    async def _process_docx_file(self, file_path: str) -> str:
        """Extracts text content from a DOCX file."""
        doc = Document(file_path)
        content = ""
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"
        return content
    
    def _create_chunks(self, content: str) -> List[str]:
        """Splits content into manageable chunks with overlap."""
        # Simple chunking strategy: split by words and then join
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks

# --- Chat Manager ---
class ChatManager:
    """Manages chat interactions, including RAG and optional OpenAI polishing."""
    def __init__(self, embedding_manager: ChromaEmbeddingManager):
        self.embedding_manager = embedding_manager
        self.openai_client = None
        
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                # FIX: Removed the 'proxies' argument from OpenAI client initialization
                # as it's causing an unexpected keyword argument error in some versions.
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ OpenAI client initialized for polishing.")
            except ImportError:
                logger.warning("⚠️ OpenAI library installed, but 'OpenAI' class not found. Make sure you have the latest 'openai' package. Polishing disabled.")
                self.openai_client = None
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize OpenAI client: {str(e)}. Polishing disabled.")
                self.openai_client = None
        else:
            logger.info("ℹ️ OpenAI API key not configured or library not available. OpenAI polishing will be skipped.")

    async def chat_with_clone(self, user_id: str, message: str, use_context: bool = True, 
                            use_openai_polish: bool = False) -> Tuple[str, str, Optional[str], Optional[float], List[str]]:
        """
        Generates a chat response for the user's AI clone.
        Uses RAG from ChromaDB and optionally polishes the response with OpenAI.
        """
        try:
            context = ""
            similarity_score = 0.0
            sources = []
            
            if use_context:
                # Retrieve relevant context from ChromaDB
                search_results = await self.embedding_manager.similarity_search(user_id, message, n_results=3)
                
                if search_results:
                    context_parts = []
                    for result in search_results:
                        context_parts.append(result['content'])
                        # Ensure metadata exists before accessing 'source'
                        source_info = result['metadata'].get('source', 'Unknown')
                        if source_info not in sources: # Avoid duplicate sources
                            sources.append(source_info)
                    
                    context = "\n\n".join(context_parts)
                    # Similarity score from the top result
                    similarity_score = search_results[0]['distance'] if search_results else 0.0
                    logger.info(f"Context retrieved for user {user_id}, query '{message[:30]}...'. Similarity: {similarity_score:.4f}")
                else:
                    logger.info(f"No context found in ChromaDB for user {user_id}, query '{message[:30]}...'")
            
            # Generate response based on context or fallback
            if context:
                response = await self._generate_contextual_response(message, context)
                model_used = "ChromaDB RAG"
            else:
                response = await self._generate_fallback_response(message)
                model_used = "Fallback (No Context)"
            
            # Polish with OpenAI if requested and available
            if use_openai_polish and self.openai_client and context:
                try:
                    polished_response = await self._polish_with_openai(response, message, context)
                    response = polished_response
                    model_used += " + OpenAI Polish"
                    logger.info("✨ Response polished with OpenAI.")
                except Exception as e:
                    logger.warning(f"⚠️ OpenAI polishing failed for user {user_id}: {str(e)}. Returning unpolished response.")
            
            return response, model_used, context, similarity_score, sources
            
        except Exception as e:
            logger.error(f"❌ Critical error in chat for user {user_id}: {str(e)}")
            return "I apologize, but I encountered an internal error processing your message. Please try again later.", "Error", None, 0.0, []
    
    async def _generate_contextual_response(self, message: str, context: str) -> str:
        """
        Generates a basic response heavily relying on the retrieved context.
        This function should ensure answers come *strictly* from the ingested data.
        For simplicity, it directly uses parts of the context. In a real LLM setup,
        you'd pass this context to an LLM with specific instructions.
        """
        # A simple approach to use context directly to prevent hallucination.
        # If the context is empty or too short, fallback will be used.

        if not context.strip():
            return self._generate_fallback_response(message)

        response_prefix = "Based on the information I have from your data, "
        
        # Simple heuristic: look for parts of the query in the context
        query_keywords = set(re.findall(r'\b\w{3,}\b', message.lower())) # Keywords of 3+ chars
        
        best_sentence = ""
        max_keyword_matches = 0

        context_sentences = re.split(r'(?<=[.!?])\s+', context) # Split by sentence-ending punctuation

        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            current_matches = sum(1 for kw in query_keywords if kw in sentence_lower)
            if current_matches > max_keyword_matches:
                max_keyword_matches = current_matches
                best_sentence = sentence

        if best_sentence:
            # Prevent direct repetition of the prompt in the response
            if message.lower() in best_sentence.lower():
                return response_prefix + "here's what I found: " + best_sentence.replace(message, "").strip()
            return response_prefix + best_sentence
        else:
            # If no directly relevant sentence, use the first few lines of context as a summary
            summary_context = context.split('\n\n')[0] # Take first paragraph
            if len(summary_context) > 200:
                summary_context = summary_context[:200] + "..."
            return response_prefix + f"I found the following in your data: {summary_context}"


    async def _generate_fallback_response(self, message: str) -> str:
        """Generates a generic fallback response when no relevant context is found."""
        fallback_responses = [
            "I don't have specific information about that in my training data. Could you provide more context?",
            "That's an interesting question, but I don't have relevant information in my knowledge base to answer it properly.",
            "I'd need more information from your training data to provide a meaningful response to that.",
            "Based on my current training data, I don't have enough context to answer that question accurately.",
            "My apologies, I couldn't find a direct answer within the provided training data."
        ]
        import random
        return random.choice(fallback_responses)
    
    async def _polish_with_openai(self, response: str, original_message: str, context: str) -> str:
        """Polishes a given response using OpenAI while strictly maintaining factual accuracy."""
        if not self.openai_client:
            logger.warning("OpenAI client not initialized. Cannot polish response.")
            return response

        try:
            system_prompt = f"""
You are an AI assistant designed to rephrase and improve the clarity, grammar, and tone of a given response.
The response *must* remain factually identical to the original content provided.
The original response was generated based on the following context:
---
{context[:1000]} # Limit context sent to OpenAI to avoid token limits
---

Instructions:
1. Improve readability and flow.
2. Correct any grammatical errors or typos.
3. Make the tone more natural and conversational.
4. **ABSOLUTELY DO NOT ADD, REMOVE, OR ALTER ANY FACTUAL INFORMATION.**
5. **DO NOT introduce new concepts or ideas not present in the original response/context.**
6. Keep the response concise, similar in length to the original.
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original question: {original_message}\n\nResponse to polish: {response}"}
            ]
            
            # Use asyncio.to_thread for blocking OpenAI API call to prevent blocking FastAPI event loop
            response_obj = await asyncio.to_thread(
                self.openai_client.chat.completions.create, # Modern OpenAI client
                model=settings.OPENAI_MODEL,
                messages=messages,
                max_tokens=200, # Limit response length for polish
                temperature=0.3 # Low temperature for factual consistency
            )
            
            polished_text = response_obj.choices[0].message.content.strip()
            logger.debug(f"Polished response: {polished_text[:100]}...")
            return polished_text
            
        except openai.APIErrors.AuthenticationError:
            logger.error("❌ OpenAI API authentication failed. Check your API key.")
            return response # Return original if auth fails
        except openai.APIErrors.RateLimitError:
            logger.warning("⚠️ OpenAI API rate limit exceeded. Retrying later or consider increasing quota.")
            return response # Return original if rate-limited
        except Exception as e:
            logger.error(f"❌ Error polishing response with OpenAI: {str(e)}")
            return response # Return original response if polishing fails

# --- Initialize Global Instances ---
embedding_manager = ChromaEmbeddingManager()
social_extractor = SocialMediaExtractor()
document_processor = DocumentProcessor()
chat_manager = ChatManager(embedding_manager)

# In-memory storage for training status
training_status: Dict[str, Dict[str, Any]] = {}

# --- FastAPI App ---
app = FastAPI(
    title="Human Clone AI Training Platform",
    description="Complete AI clone training platform with social media integration and ChromaDB embeddings",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initializes services on application startup."""
    logger.info("🚀 Starting Human Clone AI Training Platform...")
    await embedding_manager.initialize()
    logger.info("✅ All services initialized successfully")

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Human Clone AI Training Platform",
        "version": "1.0.0",
        "features": [
            "Social Media Data Extraction (LinkedIn, Facebook, YouTube)",
            "Document Processing (PDF, DOCX, TXT)",
            "ChromaDB Vector Storage",
            "Sentence Transformers Embeddings",
            "Context-based Chat with RAG",
            "Optional OpenAI Response Polishing"
        ],
        "endpoints": {
            "upload": "/api/upload",
            "social_extract": "/api/social/extract",
            "training_start": "/api/training/start",
            "training_status": "/api/training/status/{user_id}",
            "chat": "/api/chat",
            "search": "/api/embeddings/search/{user_id}",
            "stats": "/api/stats/{user_id}"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Human Clone AI Training Platform",
        "version": "1.0.0"
    }

@app.post("/api/users")
async def create_user():
    """Creates a new user and returns a unique user ID."""
    user_id = str(uuid.uuid4())
    # Initialize a basic training status entry for the new user
    training_status[user_id] = {
        "status": "not_started",
        "progress": 0.0,
        "message": "User created. No training initiated yet.",
        "social_data_files": [],
        "uploaded_files": [],
        "total_chunks": 0,
        "embedding_model": settings.EMBEDDING_MODEL
    }
    logger.info(f"👤 Created new user: {user_id}")
    return {"user_id": user_id}

@app.post("/api/upload")
async def upload_files(
    user_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Uploads files for training data ingestion."""
    try:
        if user_id not in training_status:
            raise HTTPException(status_code=404, detail="User not found. Please create a user first.")

        logger.info(f"📤 Uploading {len(files)} files for user {user_id}")
        
        uploaded_files_info = []
        user_upload_dir = os.path.join(settings.UPLOADS_DIR, user_id)
        os.makedirs(user_upload_dir, exist_ok=True)
        
        for file in files:
            # Ensure safe filename (basic sanitization)
            filename = file.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
            file_path = os.path.join(user_upload_dir, filename)
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            uploaded_files_info.append({
                "filename": filename,
                "file_path": file_path,
                "file_size": len(content),
                "mimetype": file.content_type
            })
            
            logger.info(f"💾 Saved file: {filename} ({len(content)} bytes) for user {user_id}")
        
        # Update user's training status with uploaded file info
        current_user_status = training_status.get(user_id, {})
        current_user_status["uploaded_files"] = uploaded_files_info
        current_user_status["message"] = f"Successfully uploaded {len(files)} files. Ready for training."
        training_status[user_id] = current_user_status # Ensure update is stored

        return {
            "message": f"Successfully uploaded {len(files)} files",
            "files": uploaded_files_info
        }
        
    except HTTPException as e:
        raise e # Re-raise HTTPException directly
    except Exception as e:
        logger.error(f"❌ Error uploading files for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")

@app.post("/api/social/extract")
async def extract_social_media(request: SocialMediaRequest):
    """
    Extracts social media data based on the provided request.
    This endpoint allows explicit triggering of social media extraction.
    """
    try:
        if request.user_id not in training_status:
            raise HTTPException(status_code=404, detail="User not found. Please create a user first.")

        logger.info(f"🌐 Extracting social media data for user {request.user_id}")
        
        extracted_files = []
        
        # Extract LinkedIn data
        if request.linkedin_export_file:
            try:
                linkedin_result = await social_extractor.extract_linkedin_data(
                    request.user_id, request.linkedin_export_file
                )
                extracted_files.append(linkedin_result)
            except Exception as e:
                logger.error(f"❌ Failed to extract LinkedIn data: {str(e)}")
                # Optionally, re-raise or add to a list of errors for the response
        
        # Extract Facebook data
        if request.facebook_access_token:
            try:
                facebook_result = await social_extractor.extract_facebook_data(
                    request.user_id, request.facebook_access_token
                )
                extracted_files.append(facebook_result)
            except Exception as e:
                logger.error(f"❌ Failed to extract Facebook data: {str(e)}")

        # Extract YouTube data
        if request.youtube_channel_id:
            try:
                youtube_result = await social_extractor.extract_youtube_data(
                    request.user_id, request.youtube_channel_id
                )
                extracted_files.append(youtube_result)
            except Exception as e:
                logger.error(f"❌ Failed to extract YouTube data: {str(e)}")
        
        # Update user's training status with social data info
        current_user_status = training_status.get(request.user_id, {})
        current_user_status["social_data_files"].extend(extracted_files)
        current_user_status["message"] = f"Extracted data from {len(extracted_files)} social platforms."
        training_status[request.user_id] = current_user_status # Ensure update is stored

        logger.info(f"✅ Extracted data from {len(extracted_files)} social media platforms for user {request.user_id}")
        
        return {
            "message": f"Successfully extracted data from {len(extracted_files)} platforms",
            "extracted_files": extracted_files
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Error extracting social media data for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract social media data: {str(e)}")

@app.post("/api/training/start")
async def start_training(request: StartTrainingRequest, background_tasks: BackgroundTasks):
    """Starts the training process in a background task."""
    try:
        if request.user_id not in training_status:
            raise HTTPException(status_code=404, detail="User not found. Please create a user first.")
        
        # Check if a training is already in progress
        current_status = training_status[request.user_id].get("status")
        if current_status in ["starting", "processing_files", "processing_social_media", "creating_embeddings", "completed"]:
            # If training is ongoing or completed, prevent new start unless reset
            if current_status == "completed":
                 return {
                    "message": "Training already completed for this user. Please reset if you wish to retrain.",
                    "user_id": request.user_id,
                    "status": current_status
                }
            else:
                raise HTTPException(status_code=400, detail=f"Training for user {request.user_id} is already in progress ({current_status}).")

        logger.info(f"🎯 Starting training for user {request.user_id}")
        
        # Initialize/Reset training status for a new run
        training_status[request.user_id] = {
            "status": "starting",
            "progress": 0.0,
            "message": "Initializing training process...",
            "social_data_files": [], # Reset for new training run
            "uploaded_files": [],    # Reset for new training run
            "total_chunks": 0,
            "embedding_model": settings.EMBEDDING_MODEL,
            "start_time": datetime.now().isoformat(),
            "model_name": request.model_name or "default" # Store the chosen model name
        }
        
        # Start training in background
        background_tasks.add_task(
            train_model_background, 
            request.user_id, 
            request.include_social_media,
            request.model_name # Pass model_name to background task
        )
        
        return {
            "message": "Training started successfully",
            "user_id": request.user_id,
            "status": "started"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Error starting training for user {request.user_id}: {str(e)}")
        # Update status to failed immediately on start error
        if request.user_id in training_status:
             training_status[request.user_id].update({
                "status": "failed",
                "progress": 0.0,
                "message": f"Training failed to start: {str(e)}"
            })
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/api/training/status/{user_id}", response_model=TrainingStatus)
async def get_training_status(user_id: str):
    """Gets the training status for a specific user."""
    if user_id not in training_status:
        # It's okay if a user_id doesn't have a status if they haven't started training
        # return a default "not_started" status rather than 404
        return TrainingStatus(
            user_id=user_id,
            status="not_started",
            progress=0.0,
            message="No training session found for this user.",
            social_data_files=[],
            uploaded_files=[],
            total_chunks=0,
            embedding_model=settings.EMBEDDING_MODEL
        )
    
    status_data = training_status[user_id]
    return TrainingStatus(
        user_id=user_id,
        status=status_data["status"],
        progress=status_data["progress"],
        message=status_data["message"],
        social_data_files=status_data.get("social_data_files", []),
        uploaded_files=status_data.get("uploaded_files", []),
        total_chunks=status_data.get("total_chunks", 0),
        embedding_model=status_data.get("embedding_model", settings.EMBEDDING_MODEL)
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_clone_endpoint(message: ChatMessage): # Renamed to avoid conflict with class method
    """Endpoint for chatting with the AI clone."""
    try:
        if message.user_id not in training_status:
            raise HTTPException(status_code=404, detail="User not found or model not trained. Please create a user and complete training.")
        
        # Check if training is completed before allowing chat
        if training_status[message.user_id]["status"] != "completed":
            raise HTTPException(status_code=400, detail="Training is not yet completed for this user. Please wait or start training.")

        logger.info(f"💬 Chat request from user {message.user_id}: {message.message[:50]}...")
        
        response, model_used, context, similarity_score, sources = await chat_manager.chat_with_clone(
            message.user_id,
            message.message,
            message.use_context,
            message.use_openai_polish
        )
        
        # Truncate context for response if too long, for readability
        display_context = context[:500] + "..." if context and len(context) > 500 else context

        return ChatResponse(
            response=response,
            model_used=model_used,
            context_used=display_context,
            similarity_score=similarity_score,
            sources=sources
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Error in chat for user {message.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")

@app.get("/api/embeddings/search/{user_id}")
async def search_embeddings(user_id: str, query: str, limit: int = 5):
    """Searches user's embeddings for a query."""
    try:
        if user_id not in training_status or training_status[user_id]["status"] != "completed":
            raise HTTPException(status_code=400, detail="User not found or training not completed. Embeddings not available.")

        results = await embedding_manager.similarity_search(user_id, query, limit)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Error searching embeddings for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search embeddings: {str(e)}")

@app.get("/api/stats/{user_id}")
async def get_user_stats(user_id: str):
    """Gets user statistics, including training status and data counts."""
    try:
        if user_id not in training_status:
            # If user_id is not in training_status, return a default/empty stat response
            # instead of a 404 to match frontend's expectation of some data.
            return {
                "user_id": user_id,
                "total_documents_in_chroma": 0,
                "embedding_model": settings.EMBEDDING_MODEL,
                "training_status": "not_found",
                "training_progress": 0.0,
                "total_chunks_processed": 0,
                "uploaded_files_count": 0,
                "social_data_files_count": 0,
                "last_training_message": "User or training session not found.",
                "start_time": "N/A",
                "model_name": "N/A"
            }

        collection_count = await embedding_manager.get_collection_count(user_id)
        
        user_current_status = training_status.get(user_id, {})

        return {
            "user_id": user_id,
            "total_documents_in_chroma": collection_count,
            "embedding_model": settings.EMBEDDING_MODEL,
            "training_status": user_current_status.get("status", "not_started"),
            "training_progress": user_current_status.get("progress", 0.0),
            "total_chunks_processed": user_current_status.get("total_chunks", 0),
            "uploaded_files_count": len(user_current_status.get("uploaded_files", [])),
            "social_data_files_count": len(user_current_status.get("social_data_files", [])),
            "last_training_message": user_current_status.get("message", "No training message."),
            "start_time": user_current_status.get("start_time", "N/A"),
            "model_name": user_current_status.get("model_name", "N/A")
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Error getting user stats for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user stats: {str(e)}")

# --- Background Training Function ---
async def train_model_background(user_id: str, include_social_media: bool, model_name: str):
    """
    Background process for training the AI clone.
    This function performs data processing, embedding generation, and ChromaDB ingestion.
    """
    try:
        logger.info(f"🎯 Starting background training for user {user_id}")
        
        # Ensure initial status is set
        training_status[user_id].update({
            "status": "processing_files",
            "progress": 10.0,
            "message": "Processing uploaded files...",
            "model_name": model_name # Ensure model name is carried through
        })
        
        all_chunks = []
        all_metadatas = []
        uploaded_files_processed = []
        social_data_files_info = []
        total_chunks_count = 0

        # --- Step 1: Process Uploaded Files ---
        user_upload_dir = os.path.join(settings.UPLOADS_DIR, user_id)
        if os.path.exists(user_upload_dir):
            files_in_dir = [f for f in os.listdir(user_upload_dir) if os.path.isfile(os.path.join(user_upload_dir, f))]
            
            for i, filename in enumerate(files_in_dir):
                file_path = os.path.join(user_upload_dir, filename)
                try:
                    chunks, chunk_count = await document_processor.process_file(file_path, filename)
                    
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            "source": filename,
                            "type": "uploaded_file",
                            "user_id": user_id,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    uploaded_files_processed.append(filename)
                    total_chunks_count += chunk_count
                    logger.info(f"📄 Processed uploaded file '{filename}': {chunk_count} chunks")
                    
                    # Update progress
                    current_progress = 10.0 + (i + 1) / len(files_in_dir) * 30.0 # 10% to 40% for file processing
                    training_status[user_id].update({
                        "progress": min(current_progress, 40.0),
                        "message": f"Processed uploaded file {i+1}/{len(files_in_dir)}: {filename}. Total chunks: {total_chunks_count}",
                        "uploaded_files": uploaded_files_processed,
                        "total_chunks": total_chunks_count
                    })
                    await asyncio.sleep(0.1) # Yield to event loop

                except Exception as e:
                    logger.error(f"❌ Error processing uploaded file '{filename}' for user {user_id}: {str(e)}")
                    # Continue with other files even if one fails
        else:
            logger.info(f"ℹ️ No uploaded files found for user {user_id}.")
        
        # --- Step 2: Process Social Media Data (if included) ---
        if include_social_media:
            training_status[user_id].update({
                "status": "processing_social_media",
                "progress": 45.0,
                "message": "Processing social media data..."
            })
            await asyncio.sleep(0.5)

            social_export_path = Path(settings.SOCIAL_EXPORTS_DIR)
            
            # Check for previously extracted social media files in the social_exports directory
            # These would have been created by explicit calls to /api/social/extract or previous runs
            social_files_found = [
                f for f in os.listdir(social_export_path) 
                if f.startswith(f"{user_id}_") and f.endswith("_data.txt")
            ]

            for i, filename in enumerate(social_files_found):
                file_path = os.path.join(social_export_path, filename)
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                    
                    chunks = document_processor._create_chunks(content)
                    
                    # Determine platform from filename (e.g., user_id_linkedin_data.txt -> LinkedIn)
                    platform = filename.replace(f"{user_id}_", "").replace("_data.txt", "").capitalize()
                    
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            "source": f"{platform} Data",
                            "type": "social_media",
                            "platform": platform,
                            "user_id": user_id,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    file_size = os.path.getsize(file_path)
                    social_data_files_info.append({
                        "platform": platform,
                        "filename": filename,
                        "file_size": file_size,
                        "chunks": len(chunks)
                    })
                    total_chunks_count += len(chunks)
                    logger.info(f"🌐 Processed {platform} data from '{filename}': {len(chunks)} chunks ({file_size} bytes)")

                    # Update progress
                    current_progress = 45.0 + (i + 1) / len(social_files_found) * 20.0 # 45% to 65% for social media
                    training_status[user_id].update({
                        "progress": min(current_progress, 65.0),
                        "message": f"Processed social media data from {platform}. Total chunks: {total_chunks_count}",
                        "social_data_files": social_data_files_info,
                        "total_chunks": total_chunks_count
                    })
                    await asyncio.sleep(0.1) # Yield to event loop

                except Exception as e:
                    logger.error(f"❌ Error processing social media file '{filename}' for user {user_id}: {str(e)}")
        else:
            logger.info(f"ℹ️ Social media data inclusion skipped for user {user_id}.")

        # --- Step 3: Create Embeddings and Add to ChromaDB ---
        training_status[user_id].update({
            "status": "creating_embeddings",
            "progress": 70.0,
            "message": f"Creating embeddings for {len(all_chunks)} total chunks...",
            "social_data_files": social_data_files_info,
            "uploaded_files": uploaded_files_processed,
            "total_chunks": total_chunks_count
        })
        await asyncio.sleep(0.5)

        if all_chunks:
            try:
                await embedding_manager.add_documents(
                    user_id, all_chunks, all_metadatas, "training_data"
                )
                logger.info(f"✅ Added {len(all_chunks)} chunks to ChromaDB for user {user_id}")
            except Exception as e:
                logger.error(f"❌ Failed to add documents to ChromaDB for user {user_id}: {str(e)}")
                raise # Critical failure, re-raise to fail training

        else:
            logger.warning(f"⚠️ No chunks generated for user {user_id}. Skipping embedding creation.")
            training_status[user_id].update({
                "message": "No data processed, training skipped.",
                "progress": 0.0,
                "status": "failed"
            })
            return # Exit if no data to process

        # Final status update
        training_status[user_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": (f"Training completed successfully! Processed {len(all_chunks)} chunks "
                        f"from {len(uploaded_files_processed)} uploaded files "
                        f"and {len(social_data_files_info)} social media sources. "
                        f"Model: {model_name}. Vector store ready for chat retrieval."),
            "end_time": datetime.now().isoformat()
        })
        
        logger.info(f"🎉 Training completed for user {user_id}")
        logger.info(f"📊 Summary for {user_id}:")
        logger.info(f"   - Social media data files: {len(social_data_files_info)}")
        logger.info(f"   - Uploaded files: {len(uploaded_files_processed)}")
        logger.info(f"   - Total chunks processed: {total_chunks_count}")
        logger.info(f"   - Embedding model used: {settings.EMBEDDING_MODEL}")
        logger.info(f"   - Selected AI Model (for metadata): {model_name}")
        logger.info(f"   - Vector store ready for chat retrieval")
        
    except Exception as e:
        logger.error(f"❌ Training failed for user {user_id}: {str(e)}", exc_info=True) # Log full traceback
        training_status[user_id].update({
            "status": "failed",
            "progress": 0.0,
            "message": f"Training failed: {str(e)}"
        })

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom handler for FastAPI's HTTPException."""
    logger.error(f"HTTP Exception caught: Status {exc.status_code}, Detail: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Custom handler for general unhandled exceptions."""
    logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True) # Log full traceback
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500, "detail": str(exc)}
    )

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Human Clone AI Training Platform...")
    logger.info(f"🔍 Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"📂 Models Directory: {os.path.abspath(settings.MODELS_DIR)}")
    logger.info(f"📁 Uploads Directory: {os.path.abspath(settings.UPLOADS_DIR)}")
    logger.info(f"📊 ChromaDB Directory: {os.path.abspath(settings.CHROMA_PERSIST_DIRECTORY)}")
    logger.info("💡 Features: Social Media + ChromaDB + RAG + Optional OpenAI Integration")
    
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
