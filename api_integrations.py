# # api_integrations.py - A complete and corrected version

# import os
# import logging
# import json
# from typing import List, Dict, Any, Optional

# from groq import Groq, APIStatusError as GroqAPIStatusError
# from openai import OpenAI, APIStatusError as OpenAIAPIStatusError
# from fastapi import HTTPException
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class APIIntegration:
#     """
#     Manages all third-party API integrations (e.g., OpenAI, Groq).
#     Provides a consistent, asynchronous interface for the backend application.
#     """
#     def __init__(self):
#         """Initializes API clients for Groq and OpenAI based on environment variables."""
#         self.openai_client: Optional[OpenAI] = None
#         self.groq_client: Optional[Groq] = None
#         self.groq_model: str = "mixtral-8x7b-32768"  # Default Groq model
#         self.openai_model: str = "gpt-3.5-turbo"    # Default OpenAI model

#         # Initialize OpenAI client
#         openai_key = os.getenv("OPENAI_API_KEY")
#         if openai_key:
#             try:
#                 # Use the synchronous client which is now thread-safe.
#                 # The 'proxies' argument is not supported in the standard constructor.
#                 self.openai_client = OpenAI(api_key=openai_key)
#                 logger.info("✅ OpenAI client initialized successfully.")
#             except Exception as e:
#                 logger.error(f"❌ Error initializing OpenAI client: {e}")
#                 self.openai_client = None
#         else:
#             logger.warning("⚠️ OPENAI_API_KEY is not set. OpenAI integration disabled.")

#         # Initialize Groq client
#         groq_key = os.getenv("GROQ_API_KEY")
#         if groq_key:
#             try:
#                 # Use the synchronous client which is now thread-safe.
#                 # The 'proxies' argument is not supported in the standard constructor.
#                 self.groq_client = Groq(api_key=groq_key)
#                 logger.info("✅ Groq client initialized successfully.")
#             except Exception as e:
#                 logger.error(f"❌ Error initializing Groq client: {e}")
#                 self.groq_client = None
#         else:
#             logger.warning("⚠️ GROQ_API_KEY is not set. Groq integration disabled.")

#     async def check_api_status(self) -> Dict[str, Any]:
#         """
#         Checks if API services are available and responsive with simple test calls.
#         This now returns a more detailed status dictionary.
#         """
#         status = {
#             "openai": {"status": "disabled", "message": "No API key found."},
#             "groq": {"status": "disabled", "message": "No API key found."}
#         }
        
#         # Check OpenAI
#         if self.openai_client:
#             try:
#                 # A simple, cheap call to check connectivity
#                 await self.openai_client.with_raw_response.chat.completions.create(
#                     model=self.openai_model,
#                     messages=[{"role": "user", "content": "ping"}],
#                     max_tokens=1
#                 )
#                 status["openai"] = {"status": "ok", "message": "Client is ready."}
#             except Exception as e:
#                 logger.error(f"❌ OpenAI API check failed: {str(e)}")
#                 status["openai"] = {"status": "error", "message": str(e)}

#         # Check Groq
#         if self.groq_client:
#             try:
#                 # A simple, cheap call to check connectivity
#                 await self.groq_client.with_raw_response.chat.completions.create(
#                     messages=[{"role": "user", "content": "ping"}],
#                     model=self.groq_model,
#                     max_tokens=1
#                 )
#                 status["groq"] = {"status": "ok", "message": "Client is ready."}
#             except Exception as e:
#                 logger.error(f"❌ Groq API check failed: {str(e)}")
#                 status["groq"] = {"status": "error", "message": str(e)}

#         return status

#     async def chat_with_openai(
#         self, 
#         message: str, 
#         model: str = "gpt-3.5-turbo",
#         system_prompt: Optional[str] = None
#     ) -> str:
#         """Chat with OpenAI GPT models asynchronously."""
#         if not self.openai_client:
#             raise HTTPException(status_code=503, detail="OpenAI client not initialized. Please check your API key.")
        
#         try:
#             messages = []
#             if system_prompt:
#                 messages.append({"role": "system", "content": system_prompt})
            
#             messages.append({"role": "user", "content": message})
            
#             # Use the native async client call
#             response = await self.openai_client.chat.completions.create(
#                 model=model,
#                 messages=messages,
#                 max_tokens=1000,
#                 temperature=0.7
#             )
            
#             return response.choices[0].message.content.strip()
            
#         except OpenAIAPIStatusError as e:
#             logger.error(f"❌ OpenAI API error: {e.status_code} - {e.response.json()}")
#             raise HTTPException(status_code=e.status_code, detail=f"OpenAI API Error: {e.message}")
#         except Exception as e:
#             logger.error(f"❌ An unexpected error occurred with OpenAI: {e}")
#             raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    
#     async def chat_with_groq(
#         self, 
#         message: str, 
#         model: str = "mixtral-8x7b-32768",
#         system_prompt: Optional[str] = None
#     ) -> str:
#         """Chat with Groq models asynchronously."""
#         if not self.groq_client:
#             raise HTTPException(status_code=503, detail="Groq client not initialized. Please check your API key.")
        
#         try:
#             messages = []
#             if system_prompt:
#                 messages.append({"role": "system", "content": system_prompt})
            
#             messages.append({"role": "user", "content": message})
            
#             # Use the native async client call
#             response = await self.groq_client.chat.completions.create(
#                 messages=messages,
#                 model=model,
#                 max_tokens=1000,
#                 temperature=0.7
#             )
            
#             return response.choices[0].message.content.strip()
            
#         except GroqAPIStatusError as e:
#             logger.error(f"❌ Groq API error: {e.status_code} - {e.response.json()}")
#             raise HTTPException(status_code=e.status_code, detail=f"Groq API Error: {e.message}")
#         except Exception as e:
#             logger.error(f"❌ An unexpected error occurred with Groq: {e}")
#             raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    
#     async def generate_training_data(
#         self, 
#         topic: str, 
#         count: int = 10,
#         api_provider: str = "openai"
#     ) -> list:
#         """Generate training data using API."""
#         training_data = []
        
#         system_prompt = f"""Generate {count} training examples for the topic: {topic}. 
#         Create realistic and diverse examples that would be useful for training an AI model.
#         Format each example as a simple text that represents the topic naturally, with each example on a new line."""
        
#         try:
#             if api_provider == "openai":
#                 response = await self.chat_with_openai(
#                     message=f"Generate {count} diverse training examples about {topic}.", 
#                     system_prompt=system_prompt,
#                     model=self.openai_model
#                 )
#             elif api_provider == "groq":
#                 response = await self.chat_with_groq(
#                     message=f"Generate {count} diverse training examples about {topic}.", 
#                     system_prompt=system_prompt,
#                     model=self.groq_model
#                 )
#             else:
#                 raise ValueError("Invalid API provider")
            
#             # Split response into individual examples
#             examples = response.split('\n')
#             training_data = [ex.strip() for ex in examples if ex.strip()]
            
#             logger.info(f"Generated {len(training_data)} training examples using {api_provider}")
#             return training_data
            
#         except HTTPException:
#             raise
#         except Exception as e:
#             logger.error(f"❌ Error generating training data: {str(e)}")
#             raise
    
#     async def enhance_dataset(
#         self, 
#         original_data: list, 
#         enhancement_type: str = "paraphrase",
#         api_provider: str = "openai"
#     ) -> list:
#         """Enhance existing dataset using API."""
#         enhanced_data = []
        
#         system_prompts = {
#             "paraphrase": "Paraphrase the following text while maintaining its meaning and context.",
#             "expand": "Expand the following text with more detail while keeping the core message.",
#             "style_transfer": "Rewrite the following text in a different style while preserving the meaning."
#         }
        
#         system_prompt = system_prompts.get(enhancement_type, system_prompts["paraphrase"])
        
#         try:
#             for item in original_data[:10]:  # Limit to avoid API costs
#                 if api_provider == "openai":
#                     enhanced = await self.chat_with_openai(item, system_prompt=system_prompt)
#                 elif api_provider == "groq":
#                     enhanced = await self.chat_with_groq(item, system_prompt=system_prompt)
#                 else:
#                     continue
                
#                 enhanced_data.append(enhanced)
            
#             logger.info(f"Enhanced {len(enhanced_data)} data items using {api_provider}")
#             return enhanced_data
            
#         except HTTPException:
#             raise
#         except Exception as e:
#             logger.error(f"❌ Error enhancing dataset: {str(e)}")
#             raise
    
#     async def analyze_text_quality(
#         self, 
#         text: str, 
#         api_provider: str = "openai"
#     ) -> Dict[str, Any]:
#         """Analyze text quality using API."""
#         system_prompt = """Analyze the quality of the following text and provide a JSON response with:
#         - readability_score (1-10)
#         - coherence_score (1-10)
#         - sentiment (positive/negative/neutral)
#         - key_topics (list of main topics)
#         - suggestions (list of improvement suggestions)"""
        
#         try:
#             if api_provider == "openai":
#                 response = await self.chat_with_openai(text, system_prompt=system_prompt)
#             elif api_provider == "groq":
#                 response = await self.chat_with_groq(text, system_prompt=system_prompt)
#             else:
#                 raise ValueError("Invalid API provider")
            
#             # Try to parse as JSON, fallback to text analysis
#             try:
#                 # Clean up the response to ensure it's valid JSON before parsing
#                 json_string = response.strip().strip('`').strip()
#                 if json_string.startswith("json"):
#                     json_string = json_string[4:].strip()
#                 return json.loads(json_string)
#             except json.JSONDecodeError:
#                 return {"analysis": response, "format": "text"}
            
#         except HTTPException:
#             raise
#         except Exception as e:
#             logger.error(f"❌ Error analyzing text quality: {str(e)}")
#             raise
