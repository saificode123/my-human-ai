import os
import json
import logging
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import asyncio

logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self):
        self.models_dir = "models"
        self.loaded_models = {}  # Cache for loaded models
        self.max_cache_size = 3  # Maximum number of models to keep in memory
    
    async def generate_response(self, user_id: str, message: str) -> str:
        """Generate response using user's trained model"""
        logger.info(f"Generating response for user {user_id}")
        
        try:
            # Load or get cached model
            model, tokenizer = await self._get_user_model(user_id)
            
            if not model or not tokenizer:
                return "I'm sorry, but your personalized model isn't ready yet. Please complete the training process first."
            
            # Prepare input
            prompt = f"Human: {message} Assistant:"
            
            # Generate response
            response = await self._generate_text(model, tokenizer, prompt)
            
            # Clean up response
            cleaned_response = self._clean_response(response, message)
            
            logger.info(f"Response generated successfully for user {user_id}")
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."
    
    async def _get_user_model(self, user_id: str):
        """Load or retrieve cached user model"""
        model_path = os.path.join(self.models_dir, user_id)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found for user {user_id}")
            return None, None
        
        # Check if model is already loaded
        if user_id in self.loaded_models:
            logger.info(f"Using cached model for user {user_id}")
            return self.loaded_models[user_id]
        
        # Load model
        try:
            logger.info(f"Loading model for user {user_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Cache management
            if len(self.loaded_models) >= self.max_cache_size:
                # Remove oldest model
                oldest_user = next(iter(self.loaded_models))
                del self.loaded_models[oldest_user]
                logger.info(f"Removed cached model for user {oldest_user}")
            
            # Cache the model
            self.loaded_models[user_id] = (model, tokenizer)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model for user {user_id}: {str(e)}")
            return None, None
    
    async def _generate_text(self, model, tokenizer, prompt: str, max_length: int = 150) -> str:
        """Generate text using the model"""
        try:
            # Tokenize input
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return "I'm having trouble generating a response right now."
    
    def _clean_response(self, response: str, original_message: str) -> str:
        """Clean and format the generated response"""
        # Remove potential repetitions of the input
        if original_message.lower() in response.lower():
            response = response.replace(original_message, "").strip()
        
        # Remove common artifacts
        response = response.replace("Human:", "").replace("Assistant:", "").strip()
        
        # Ensure response doesn't end mid-sentence
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Fallback responses for empty or too short responses
        if len(response.strip()) < 10:
            fallbacks = [
                "That's an interesting point. Could you tell me more about it?",
                "I'd love to hear your thoughts on that topic.",
                "That's something I'd like to explore further with you.",
                "Could you elaborate on that? I find it quite intriguing."
            ]
            import random
            response = random.choice(fallbacks)
        
        return response[:500]  # Limit response length
    
    async def get_model_info(self, user_id: str) -> Optional[Dict]:
        """Get information about user's trained model"""
        model_path = os.path.join(self.models_dir, user_id)
        metadata_path = os.path.join(model_path, "training_metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                "model_exists": True,
                "base_model": metadata.get("base_model", "unknown"),
                "training_data_size": metadata.get("training_data_size", 0),
                "model_path": model_path,
                "training_completed": metadata.get("training_completed", False)
            }
            
        except Exception as e:
            logger.error(f"Error reading model metadata: {str(e)}")
            return None
    
    def clear_model_cache(self, user_id: Optional[str] = None):
        """Clear model cache"""
        if user_id:
            if user_id in self.loaded_models:
                del self.loaded_models[user_id]
                logger.info(f"Cleared cache for user {user_id}")
        else:
            self.loaded_models.clear()
            logger.info("Cleared all model cache")
