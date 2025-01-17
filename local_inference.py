"""Local inference module for running quantized language models on IBM Power architecture.

This module provides a drop-in replacement for OpenAI's chat completion API,
using GGUF quantized models for efficient inference.
"""

import os
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from llama_cpp import Llama
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class Message:
    role: str
    content: str

class LocalLLMChat:
    """Local LLM chat implementation using GGUF models."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "mistral",
        temperature: float = 0.0,
        max_length: int = 2048,
        streaming: bool = True,
        n_ctx: int = 2048,  # Context window size
        n_threads: Optional[int] = None  # Number of threads to use
    ):
        """Initialize local LLM for chat."""
        self.temperature = temperature
        self.max_length = max_length
        self.streaming = streaming
        
        # Verify model path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
            
        logger.info(f"Loading model from {model_path}")
        try:
            # Initialize the GGUF model
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads or os.cpu_count(),
                verbose=False
            )
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _format_chat_prompt(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Format chat messages for the model."""
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.role,
                "content": msg.content
            })
        return formatted

    async def ainvoke(
        self, 
        messages: List[Dict[str, str]], 
        structured_output: Optional[Any] = None
    ) -> Union[Dict[str, str], Any]:
        """Async interface for chat completion."""
        msgs = [Message(**msg) for msg in messages]
        formatted_msgs = self._format_chat_prompt(msgs)
        
        try:
            # Generate response
            completion = self.model.create_chat_completion(
                messages=formatted_msgs,
                temperature=self.temperature,
                max_tokens=self.max_length,
                stream=self.streaming
            )
            
            if self.streaming:
                # Handle streaming response
                generated_text = ""
                for chunk in completion:
                    if "content" in chunk["choices"][0]["delta"]:
                        new_text = chunk["choices"][0]["delta"]["content"]
                        generated_text += new_text
            else:
                # Handle non-streaming response
                generated_text = completion["choices"][0]["message"]["content"]
                
            return {"role": "assistant", "content": generated_text}
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

def create_local_llm(model_config: Dict[str, Any]) -> LocalLLMChat:
    """Create local LLM instance based on config."""
    return LocalLLMChat(
        model_path=model_config["path"],
        model_type=model_config.get("type", "mistral"),
        temperature=model_config.get("temperature", 0.0),
        streaming=model_config.get("streaming", True)
    )
