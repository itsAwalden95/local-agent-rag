"""Local inference module for running language models on IBM Power architecture.

This module provides a drop-in replacement for OpenAI's chat completion API,
allowing for local inference on IBM Power servers.
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str

class LocalLLMChat:
    """Local LLM chat implementation compatible with ChatOpenAI interface."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "mistral",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.0,
        max_length: int = 2048,
        streaming: bool = True
    ):
        """Initialize local LLM for chat.
        
        Args:
            model_path: Path to the model weights
            model_type: Type of model architecture
            device: Device to run inference on
            temperature: Sampling temperature
            max_length: Maximum sequence length
            streaming: Whether to stream output tokens
        """
        self.device = device
        self.temperature = temperature
        self.max_length = max_length
        self.streaming = streaming
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(device)
        
        # Set up streamer if needed
        self.streamer = TextIteratorStreamer(self.tokenizer) if streaming else None

    def _format_chat_prompt(self, messages: List[Message]) -> str:
        """Format chat messages into model prompt."""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"<|system|>\n{msg.content}</s>")
            elif msg.role == "user":
                formatted.append(f"<|user|>\n{msg.content}</s>") 
            elif msg.role == "assistant":
                formatted.append(f"<|assistant|>\n{msg.content}</s>")
        return "\n".join(formatted) + "\n<|assistant|>\n"

    async def ainvoke(
        self, 
        messages: List[Dict[str, str]], 
        structured_output: Optional[Any] = None
    ) -> Union[Dict[str, str], Any]:
        """Async interface for chat completion.
        
        Args:
            messages: List of message dictionaries
            structured_output: Optional type for structured output parsing
            
        Returns:
            Generated response
        """
        # Convert dict messages to dataclass
        msgs = [Message(**msg) for msg in messages]
        prompt = self._format_chat_prompt(msgs)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        if self.streaming:
            # Run in separate thread if streaming
            thread = Thread(target=self._generate, args=(inputs, self.streamer))
            thread.start()
            
            # Collect streamed output
            generated_text = ""
            for new_text in self.streamer:
                generated_text += new_text
                
        else:
            # Generate full response at once
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0
                )
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
        # Parse structured output if needed
        if structured_output is not None:
            # Add structured output parsing logic here
            pass
            
        return {"role": "assistant", "content": generated_text}
        
    def _generate(self, inputs: Dict[str, torch.Tensor], streamer: TextIteratorStreamer):
        """Run generation in thread for streaming."""
        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                streamer=streamer
            )

# Helper function to create model instances
def create_local_llm(model_config: Dict[str, Any]) -> LocalLLMChat:
    """Create local LLM instance based on config."""
    return LocalLLMChat(
        model_path=model_config["path"],
        model_type=model_config.get("type", "mistral"),
        temperature=model_config.get("temperature", 0.0),
        streaming=model_config.get("streaming", True)
    )
