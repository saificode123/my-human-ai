# filename: backend_service.py

import requests
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import asyncio
import os

# Third-party libraries for NLP and data processing
from transformers import pipeline
import spacy
from bs4 import BeautifulSoup  # Although not directly used in the final version, it's a good practice to keep it if web scraping is a potential feature.

# The `textblob` library was imported in the original code but not used.
# It has been removed to reduce unnecessary dependencies.

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================
# DataFetcher Class
# ================================
class DataFetcher:
    """
    Handles data ingestion and preprocessing from various sources
    like local files and simulated social media APIs.
    """
    def __init__(self):
        """Initializes the data fetcher with NLP models."""
        
        # Load spaCy model for linguistic analysis (e.g., named entity recognition)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy 'en_core_web_sm' model loaded successfully.")
        except OSError:
            logger.warning("⚠️ spaCy model not found. Run 'python -m spacy download en_core_web_sm'.")
            self.nlp = None

        # Initialize sentiment analysis pipeline from Hugging Face Transformers
        try:
            # Use a robust, pre-trained sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("✅ Transformer sentiment-analysis pipeline initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to load advanced sentiment model. Falling back to a simpler one. Error: {e}")
            # Fallback to a lighter model if the primary one fails
            self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    async def process_uploaded_file(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes an uploaded training data file based on its extension.

        Args:
            file_path (str): The local path to the uploaded file.
            config (Dict[str, Any]): Configuration for processing (e.g., 'text_column' for CSV).

        Returns:
            Dict[str, Any]: A dictionary containing processed text content and metadata.
        """
        logger.info(f"Processing uploaded file: {file_path}")
        
        file_extension = file_path.split('.')[-1].lower()
        
        try:
            if file_extension == 'txt':
                processed_data = await self._process_text_file(file_path, config)
            elif file_extension == 'csv':
                processed_data = await self._process_csv_file(file_path, config)
            elif file_extension == 'json':
                processed_data = await self._process_json_file(file_path, config)
            elif file_extension == 'pdf':
                processed_data = await self._process_pdf_file(file_path, config)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            logger.info(f"Processed {processed_data['metadata']['total_items']} items from {file_path}")
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error processing file {file_path}: {str(e)}")
            # Re-raise the exception after logging to ensure the caller knows it failed
            raise
    
    async def _process_text_file(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a plain text file, splitting it by lines or paragraphs."""
        def read_file():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        content = await asyncio.to_thread(read_file)
        
        # Split by lines or paragraphs based on configuration
        if config.get('split_by', 'lines') == 'paragraphs':
            texts = [p.strip() for p in content.split('\n\n') if p.strip()]
        else:
            texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Clean and filter texts based on minimum length
        cleaned_texts = [
            self._clean_text(text)
            for text in texts
            if len(self._clean_text(text)) > config.get('min_length', 10)
        ]
        
        return {
            "text_content": cleaned_texts,
            "metadata": {
                "file_type": "txt",
                "config": config,
                "total_items": len(cleaned_texts),
                "content_types": ["text"]
            }
        }
    
    async def _process_csv_file(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a CSV file, extracting text from a specified column."""
        def read_csv():
            return pd.read_csv(file_path)

        df = await asyncio.to_thread(read_csv)
        
        text_column = config.get('text_column', 'text')
        
        # Fallback logic to find a suitable text column if the specified one doesn't exist
        if text_column not in df.columns:
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
            if text_columns:
                text_column = text_columns[0]
            else:
                text_column = df.columns[0]
        
        texts = []
        for text in df[text_column].astype(str).tolist():
            cleaned = self._clean_text(text)
            if len(cleaned) > config.get('min_length', 10):
                texts.append(cleaned)
        
        return {
            "text_content": texts,
            "metadata": {
                "file_type": "csv",
                "config": config,
                "total_items": len(texts),
                "columns": list(df.columns),
                "text_column": text_column,
                "content_types": ["structured_text"]
            }
        }
    
    async def _process_json_file(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a JSON file, extracting text based on a configuration."""
        def read_json():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        data = await asyncio.to_thread(read_json)
        texts = []
        
        if isinstance(data, list):
            for item in data:
                text = self._extract_text_from_json_item(item, config)
                if text:
                    cleaned = self._clean_text(text)
                    if len(cleaned) > config.get('min_length', 10):
                        texts.append(cleaned)
        elif isinstance(data, dict):
            text = self._extract_text_from_json_item(data, config)
            if text:
                cleaned = self._clean_text(text)
                if len(cleaned) > config.get('min_length', 10):
                    texts.append(cleaned)
        
        return {
            "text_content": texts,
            "metadata": {
                "file_type": "json",
                "config": config,
                "total_items": len(texts),
                "content_types": ["structured_json"]
            }
        }
    
    async def _process_pdf_file(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates PDF file processing.
        NOTE: This is a mock implementation. In a real-world scenario, you would
        use libraries like PyPDF2 or pdfplumber to extract text.
        """
        logger.warning("Simulating PDF text extraction. Actual implementation requires a library like PyPDF2.")
        
        simulated_text = f"""
        This is simulated text extracted from PDF: {os.path.basename(file_path)}.
        In a real implementation, this would be the actual text content extracted from the PDF file.
        The text would be processed paragraph by paragraph or page by page depending on the configuration.
        """
        
        paragraphs = [p.strip() for p in simulated_text.split('\n\n') if p.strip()]
        
        cleaned_texts = [
            self._clean_text(text)
            for text in paragraphs
            if len(self._clean_text(text)) > config.get('min_length', 10)
        ]
        
        return {
            "text_content": cleaned_texts,
            "metadata": {
                "file_type": "pdf",
                "config": config,
                "total_items": len(cleaned_texts),
                "content_types": ["pdf_text"]
            }
        }
    
    def _extract_text_from_json_item(self, item: Any, config: Dict[str, Any]) -> Optional[str]:
        """
        Recursively extracts text from a JSON item based on configured fields.
        """
        text_fields = config.get('text_fields', ['text', 'content', 'message', 'description'])
        
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            for field in text_fields:
                if field in item and isinstance(item[field], str):
                    return item[field]
            # Fallback: concatenate all non-empty string values
            text_parts = [str(v) for v in item.values() if isinstance(v, str) and len(str(v)) > 10]
            return ' '.join(text_parts) if text_parts else None
        else:
            return str(item) if item else None
    
    def _clean_text(self, text: str) -> str:
        """Cleans and normalizes a string for training."""
        if not text:
            return ""
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', text)
        text = ' '.join(text.split()).strip()
        
        return text
    
    # --- Simulated Social Media Data Fetching (for demonstration) ---
    async def fetch_linkedin_profile(self, url: str) -> Dict[str, Any]:
        """Simulates fetching LinkedIn profile data."""
        logger.info(f"Simulating LinkedIn data fetch for: {url}")
        
        simulated_data = {
            "platform": "linkedin",
            "posts": [
                "Excited to announce our new AI project!",
                "Just completed a fascinating machine learning course."
            ],
            "about": "Passionate about artificial intelligence and machine learning."
        }
        await asyncio.sleep(1)
        return simulated_data
    
    async def export_facebook_data(self, url: str) -> Dict[str, Any]:
        """Simulates fetching Facebook data."""
        logger.info(f"Simulating Facebook data export for: {url}")
        
        simulated_data = {
            "platform": "facebook",
            "posts": [
                "Had an amazing weekend hiking with friends!",
                "Cooking a new recipe tonight. Love experimenting."
            ]
        }
        await asyncio.sleep(1)
        return simulated_data
    
    async def fetch_youtube_transcripts(self, channel_id: str) -> Dict[str, Any]:
        """Simulates fetching YouTube transcripts."""
        logger.info(f"Simulating YouTube transcript fetch for channel: {channel_id}")
        
        simulated_data = {
            "platform": "youtube",
            "transcripts": [
                "Welcome back! Today we're going to talk about the latest developments in artificial intelligence.",
                "In this tutorial, I'll show you how to build a simple machine learning model from scratch."
            ]
        }
        await asyncio.sleep(1)
        return simulated_data
    
    async def fetch_all_data(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches data from all specified simulated platforms asynchronously."""
        all_data = {}
        tasks = []
        
        if profile_data.get('linkedin_url'):
            tasks.append(self.fetch_linkedin_profile(str(profile_data['linkedin_url'])))
        if profile_data.get('facebook_url'):
            tasks.append(self.export_facebook_data(str(profile_data['facebook_url'])))
        if profile_data.get('youtube_channel_id'):
            tasks.append(self.fetch_youtube_transcripts(profile_data['youtube_channel_id']))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching data from a platform: {result}")
            elif result:
                platform = result['platform']
                all_data[platform] = result
        
        logger.info(f"Fetched data from {len(all_data)} platforms")
        return all_data
    
    async def preprocess_data(self, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cleans and combines collected data from all platforms."""
        processed_data = {
            "text_content": [],
            "metadata": {
                "total_items": 0,
                "platforms": list(all_data.keys()),
                "content_types": []
            }
        }
        
        for platform, data in all_data.items():
            logger.info(f"Processing {platform} data")
            
            texts = []
            if platform == "linkedin":
                texts.extend(data.get('posts', []))
                texts.append(data.get('about', ''))
                texts.extend(data.get('experience', []))
                processed_data['metadata']['content_types'].append('professional')
            elif platform == "facebook":
                texts.extend(data.get('posts', []))
                texts.extend(data.get('comments', []))
                processed_data['metadata']['content_types'].append('personal')
            elif platform == "youtube":
                texts.extend(data.get('transcripts', []))
                texts.extend(data.get('comments', []))
                processed_data['metadata']['content_types'].append('educational')
            
            cleaned_texts = [
                self._clean_text(text)
                for text in texts
                if text and isinstance(text, str) and len(self._clean_text(text)) > 10
            ]
            
            processed_data['text_content'].extend(cleaned_texts)
            processed_data['metadata']['total_items'] += len(cleaned_texts)
        
        logger.info(f"Processed {processed_data['metadata']['total_items']} text items.")
        return processed_data

# ================================
# APIIntegration Class (OpenAI Only)
# ================================
class APIIntegration:
    """
    Manages all API integrations, now exclusively for OpenAI.
    """
    def __init__(self):
        """Initializes the OpenAI client."""
        # Initialize OpenAI client
        self.openai_client = None
        
        # Set up OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            # The openai library has been updated. The client is now a class instance.
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=openai_key)
            logger.info("✅ OpenAI client initialized.")
        else:
            logger.warning("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    async def check_api_status(self) -> Dict[str, bool]:
        """Checks if the OpenAI API service is available."""
        status = {"openai": False}
        if self.openai_client:
            try:
                # Make a simple test call to the API
                await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                status["openai"] = True
            except Exception as e:
                logger.error(f"❌ OpenAI API check failed: {str(e)}")
        
        return status
    
    async def chat_with_openai(
        self, 
        message: str, 
        model: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Sends a chat message to the OpenAI API and returns the response.

        Args:
            message (str): The user message to send.
            model (str): The OpenAI model to use.
            system_prompt (Optional[str]): A system prompt to guide the model's behavior.

        Returns:
            str: The text content of the model's response.
        """
        if not self.openai_client:
            raise Exception("OpenAI client not initialized. Please check your API key.")
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": message})
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {str(e)}")
            raise
    
    async def generate_training_data(self, topic: str, count: int = 10) -> List[str]:
        """
        Generates training data examples using the OpenAI API.

        Args:
            topic (str): The topic for which to generate examples.
            count (int): The number of examples to generate.

        Returns:
            List[str]: A list of generated training examples.
        """
        system_prompt = f"""Generate training examples for the topic: {topic}. 
Create realistic and diverse examples that would be useful for training an AI model.
Format each example as a simple text that represents the topic naturally."""
        
        prompt = f"Generate {count} diverse training examples about {topic}. Each example should be on a new line."
        
        try:
            response = await self.chat_with_openai(prompt, system_prompt=system_prompt)
            examples = [ex.strip() for ex in response.split('\n') if ex.strip()]
            
            logger.info(f"Generated {len(examples)} training examples using OpenAI.")
            return examples
            
        except Exception as e:
            logger.error(f"❌ Error generating training data: {str(e)}")
            raise
    
    async def enhance_dataset(self, original_data: List[str], enhancement_type: str = "paraphrase") -> List[str]:
        """
        Enhances an existing dataset using the OpenAI API.
        
        Args:
            original_data (List[str]): The original data to enhance.
            enhancement_type (str): The type of enhancement ('paraphrase', 'expand', 'style_transfer').

        Returns:
            List[str]: The enhanced data items.
        """
        enhanced_data = []
        
        system_prompts = {
            "paraphrase": "Paraphrase the following text while maintaining its meaning and context.",
            "expand": "Expand the following text with more detail while keeping the core message.",
            "style_transfer": "Rewrite the following text in a different style while preserving the meaning."
        }
        
        system_prompt = system_prompts.get(enhancement_type, system_prompts["paraphrase"])
        
        try:
            # Limiting to 10 items to prevent high API costs for large datasets
            for item in original_data[:10]:
                enhanced = await self.chat_with_openai(item, system_prompt=system_prompt)
                enhanced_data.append(enhanced)
                await asyncio.sleep(0.1) # Small delay to respect rate limits
            
            logger.info(f"Enhanced {len(enhanced_data)} data items using OpenAI.")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Error enhancing dataset: {str(e)}")
            raise
    
    async def analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Analyzes text quality using the OpenAI API, requesting a structured JSON response.

        Args:
            text (str): The text content to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results.
        """
        system_prompt = """Analyze the quality of the following text and provide a JSON response with:
        - `readability_score` (1-10)
        - `coherence_score` (1-10)
        - `sentiment` (positive/negative/neutral)
        - `key_topics` (list of main topics)
        - `suggestions` (list of improvement suggestions)"""
        
        try:
            response = await self.chat_with_openai(text, system_prompt=system_prompt)
            
            # The model might not always return perfect JSON, so we handle potential errors.
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Received non-JSON response from API. Returning raw text.")
                return {"analysis": response, "format": "text"}
            
        except Exception as e:
            logger.error(f"❌ Error analyzing text quality: {str(e)}")
            raise
