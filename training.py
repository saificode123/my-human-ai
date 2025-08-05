import os
import json
import logging
from typing import Dict, List, Any, Optional
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import asyncio

# Set up logging for this module
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Supported models (CPU-friendly and more realistic for limited RAM)
        # IMPORTANT: A 30B parameter model like Qwen/Qwen3-30B-A3B-Thinking-2507
        # is NOT feasible for training or efficient inference on 4GB RAM.
        # Even highly quantized versions require significantly more memory (15GB+).
        # Training typically requires 4x the model size in memory.
        # We are using smaller, more memory-efficient models for this constraint.
        self.supported_models = {
            "distilgpt2": "distilgpt2",
            "gpt2": "gpt2", 
            "microsoft/DialoGPT-small": "microsoft/DialoGPT-small",
            # Adding TinyLlama as a slightly larger, but still relatively small, option
            # for conversational tasks. Still challenging for 4GB RAM for training.
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        }
        
        logger.info(f"🤖 ModelTrainer initialized with supported models: {list(self.supported_models.keys())}")
    
    async def train_model(self, user_id: str, processed_data: Dict[str, Any], model_name: str = "distilgpt2") -> str:
        """
        Train a personalized model for the user.
        This function is designed for CPU-only training to be compatible with environments
        without dedicated GPUs, keeping memory usage in mind.
        """
        logger.info(f"🎯 Starting model training for user {user_id} with {model_name}")
        
        if model_name not in self.supported_models:
            logger.error(f"❌ Model '{model_name}' is not supported. Please choose from {list(self.supported_models.keys())}.")
            raise ValueError(f"Model '{model_name}' is not supported.")

        try:
            # Prepare dataset
            dataset = self._prepare_dataset(processed_data)
            if not dataset:
                raise ValueError("Prepared dataset is empty. Cannot train model.")
            logger.info(f"📊 Dataset prepared with {len(dataset)} examples")
            
            # Initialize model and tokenizer
            model_checkpoint = self.supported_models[model_name]
            logger.info(f"🔧 Loading model: {model_checkpoint}")
            
            # Use asyncio.to_thread for synchronous Hugging Face model/tokenizer loading
            # to prevent blocking the event loop.
            tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, model_checkpoint)
            model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_checkpoint)
            
            # Add pad token if not present, essential for batching
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("🔧 Added pad token to tokenizer")
            
            # Tokenize dataset
            logger.info("🔤 Tokenizing dataset...")
            # Using asyncio.to_thread to run the potentially long .map operation in a separate thread
            tokenized_dataset = await asyncio.to_thread(
                dataset.map,
                lambda examples: self._tokenize_function(examples, tokenizer),
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Set up training arguments
            model_path = os.path.join(self.models_dir, user_id)
            os.makedirs(model_path, exist_ok=True)
            
            training_args = TrainingArguments(
                output_dir=model_path,
                overwrite_output_dir=True,
                num_train_epochs=2,  # Reduced for faster training on CPU
                per_device_train_batch_size=2,  # Small batch size for CPU memory
                per_device_eval_batch_size=2,
                warmup_steps=50,
                logging_steps=25,
                save_steps=200,
                eval_steps=200,
                evaluation_strategy="steps",
                save_total_limit=2, # Limit saved checkpoints to save disk space
                prediction_loss_only=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False, # Disable for CPU to save memory
                no_cuda=True,  # Force CPU training - CRITICAL for 4GB RAM
                fp16=False,  # Disable mixed precision for CPU training
                logging_dir=f"{model_path}/logs",
                report_to=None,  # Disable external reporting (wandb/tensorboard)
                load_best_model_at_end=False,  # Disable to save memory during training
            )
            
            # Data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM (predict next token), not masked LM
            )
            
            # Split dataset for training and validation (if large enough)
            if len(tokenized_dataset) > 10:
                train_size = max(int(0.8 * len(tokenized_dataset)), 1)
                train_dataset = tokenized_dataset.select(range(train_size))
                eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
                logger.info(f"📊 Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
            else:
                train_dataset = tokenized_dataset
                eval_dataset = None
                logger.warning(f"📊 Dataset too small for split. Using full dataset for training: {len(train_dataset)} examples")
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer, # Pass tokenizer to trainer for proper handling
            )
            
            # Train model
            logger.info("🚀 Starting model training...")
            # Run trainer.train() in a separate thread to keep the main event loop free
            await asyncio.to_thread(trainer.train)
            
            # Save model and tokenizer
            logger.info("💾 Saving trained model...")
            await asyncio.to_thread(trainer.save_model)
            await asyncio.to_thread(tokenizer.save_pretrained, model_path)
            
            # Save training metadata
            metadata = {
                "user_id": user_id,
                "base_model": model_checkpoint,
                "training_data_size": len(dataset),
                "model_path": model_path,
                "training_completed": True,
                "model_name": model_name,
                "training_epochs": training_args.num_train_epochs,
                "created_at": str(datetime.now()) # Use datetime for consistency
            }
            
            metadata_file_path = os.path.join(model_path, "training_metadata.json")
            await asyncio.to_thread(
                lambda: json.dump(metadata, open(metadata_file_path, "w"), indent=2)
            )
            
            logger.info(f"✅ Model training completed successfully for user {user_id}")
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Error in model training for user {user_id}: {str(e)}", exc_info=True)
            raise # Re-raise the exception after logging for proper error propagation
    
    def _prepare_dataset(self, processed_data: Dict[str, Any]) -> Optional[Dataset]:
        """
        Prepare dataset from processed data, formatting it for conversational fine-tuning.
        Adds personality context if available.
        """
        texts = processed_data.get('text_content', [])
        
        if not texts:
            logger.warning("No text content available for training. Returning empty dataset.")
            return None
        
        formatted_texts = []
        
        # Add personality context if available
        personality_context = ""
        if 'personality_traits' in processed_data and isinstance(processed_data['personality_traits'], dict):
            traits = processed_data['personality_traits']
            sentiment_info = traits.get('sentiment_analysis', {})
            style_info = traits.get('communication_style', {})
            
            if sentiment_info.get('overall_tone'):
                personality_context += f"This person has a {sentiment_info['overall_tone']} communication style. "
            
            if style_info.get('communication_patterns'):
                patterns = style_info['communication_patterns']
                personality_context += f"They tend to be {patterns.get('enthusiasm', 'moderate')} in enthusiasm and {patterns.get('formality', 'moderate')} in formality. "
        
        # Format texts for training
        for i, text in enumerate(texts):
            cleaned_text = text.strip()
            if len(cleaned_text) > 20:  # Filter out very short texts
                # Create different conversation formats to make the model more robust
                formats = [
                    f"Human: Tell me about yourself. Assistant: {cleaned_text}",
                    f"Human: What do you think about this topic? Assistant: {cleaned_text}",
                    f"Human: Can you share your thoughts? Assistant: {cleaned_text}",
                    f"Human: How would you describe this? Assistant: {cleaned_text}"
                ]
                
                # Choose format based on text index to vary prompts
                format_choice = formats[i % len(formats)]
                
                # Add personality context occasionally to infuse persona
                if personality_context and i % 5 == 0: # Add context every 5 examples
                    format_choice = f"Context: {personality_context}\n{format_choice}"
                
                formatted_texts.append(format_choice)
        
        # Ensure we have at least a minimum number of training data examples
        # Training with very few examples can lead to poor results or errors.
        min_examples = 10 # Increased minimum for better stability
        if len(formatted_texts) < min_examples:
            logger.warning(f"Not enough unique formatted texts ({len(formatted_texts)}). Duplicating data to reach {min_examples} examples.")
            # Duplicate data if we have too little to meet minimum requirement
            # Use `max` to ensure we don't divide by zero if formatted_texts is empty
            if formatted_texts:
                formatted_texts = formatted_texts * (min_examples // len(formatted_texts) + 1)
            else: # If still empty, create a placeholder to prevent crash
                logger.error("No valid text content after filtering and formatting. Cannot create dataset.")
                return None
        
        # Create dataset from the list of formatted texts
        dataset = Dataset.from_dict({"text": formatted_texts})
        logger.info(f"📊 Prepared dataset with {len(dataset)} training examples")
        
        return dataset
    
    def _tokenize_function(self, examples: Dict[str, List], tokenizer) -> Dict[str, List]:
        """
        Tokenizes text examples for model training.
        """
        # Tokenize with appropriate settings for the model
        # max_length is kept conservative for memory efficiency on CPU
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length", # Use "max_length" for consistent tensor shapes
            max_length=256,       # Reduced for faster training and lower memory footprint
            return_tensors="pt"
        )
        
        # For causal LM, labels are typically the same as input_ids
        # This means the model tries to predict the next token given the previous ones.
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return {
            "input_ids": tokenized["input_ids"].tolist(),
            "attention_mask": tokenized["attention_mask"].tolist(),
            "labels": tokenized["labels"].tolist()
        }
    
    async def retrain_with_conversations(self, user_id: str, conversations: List[Dict[str, Any]]):
        """
        Retrain an existing personalized model with new conversation data.
        This is a lightweight fine-tuning step to adapt the model to recent interactions.
        """
        logger.info(f"🔄 Starting retraining for user {user_id} with {len(conversations)} conversations")
        
        try:
            model_path = os.path.join(self.models_dir, user_id)
            
            # Check if a base model exists to retrain
            if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
                logger.error(f"❌ No existing model found for user {user_id} at {model_path}. Cannot retrain.")
                raise FileNotFoundError(f"No existing model found for user {user_id}. Please train a base model first.")
            
            # Load existing model and tokenizer
            logger.info("📥 Loading existing model for retraining...")
            tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, model_path)
            model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_path)
            
            # Prepare conversation dataset from recent interactions
            conv_texts = []
            # Using only recent conversations (e.g., last 20) to keep retraining fast and focused
            for conv in conversations[-20:]: 
                user_msg = conv.get('user_message', '').strip()
                bot_resp = conv.get('bot_response', '').strip()
                if user_msg and bot_resp:
                    conv_text = f"Human: {user_msg} Assistant: {bot_resp}"
                    conv_texts.append(conv_text)
            
            if not conv_texts:
                logger.info("ℹ️ No valid conversation data available for retraining. Skipping.")
                return
            
            # Create dataset from conversation texts
            dataset = Dataset.from_dict({"text": conv_texts})
            
            # Tokenize the conversation dataset
            tokenized_dataset = await asyncio.to_thread(
                dataset.map,
                lambda examples: self._tokenize_function(examples, tokenizer),
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Quick fine-tuning arguments for retraining
            training_args = TrainingArguments(
                output_dir=model_path,
                overwrite_output_dir=True,
                num_train_epochs=1, # Very few epochs for fast adaptation
                per_device_train_batch_size=1, # Smallest batch size for memory
                warmup_steps=10,
                logging_steps=10,
                save_steps=50,
                save_total_limit=1,
                no_cuda=True, # Force CPU
                fp16=False,
                logging_dir=f"{model_path}/retrain_logs",
                report_to=None,
                load_best_model_at_end=False,
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # Retrain the model
            logger.info("🔄 Retraining model...")
            await asyncio.to_thread(trainer.train)
            await asyncio.to_thread(trainer.save_model)
            
            # Update metadata with retraining information
            metadata_path = os.path.join(model_path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['last_retrain'] = str(datetime.now())
                metadata['retrain_conversations'] = len(conversations)
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Model retrained successfully for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Error in retraining model for user {user_id}: {str(e)}", exc_info=True)
            raise 
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported model names"""
        return list(self.supported_models.keys())
    
    def model_exists(self, user_id: str) -> bool:
        """Check if a trained model exists for the user"""
        model_path = os.path.join(self.models_dir, user_id)
        # Check for both the directory and a key configuration file within it
        return os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json"))

