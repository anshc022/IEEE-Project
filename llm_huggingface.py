# Alternative LLM Integration: Hugging Face Transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class HuggingFaceLLM:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize Hugging Face LLM
        
        Good local models to try:
        - "microsoft/DialoGPT-medium" (conversational)
        - "distilgpt2" (small, fast)
        - "gpt2" (classic)
        - "microsoft/DialoGPT-large" (better quality, slower)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            print(f"✅ Model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_response(self, prompt, max_length=100):
        """Generate response from the model"""
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        
        try:
            # Encode the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
        except Exception as e:
            return f"Error generating response: {e}"

# Example usage in your seed detection system:
def integrate_huggingface_llm():
    """
    Integration example for Hugging Face models
    """
    llm = HuggingFaceLLM("distilgpt2")  # Fast, small model
    
    # Example analysis
    seed_data = {
        'good_seeds': 15,
        'bad_seeds': 3,
        'total': 18
    }
    
    prompt = f"Analysis: Detected {seed_data['good_seeds']} good seeds and {seed_data['bad_seeds']} bad seeds. Quality assessment:"
    response = llm.generate_response(prompt)
    print(f"LLM Analysis: {response}")

if __name__ == "__main__":
    integrate_huggingface_llm()
