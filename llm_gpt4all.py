# GPT4All Integration for Completely Offline LLM
from gpt4all import GPT4All
import time
import json

class GPT4AllLLM:
    def __init__(self, model_name="orca-mini-3b-gguf2-q4_0.gguf"):
        """
        Initialize GPT4All LLM for completely offline usage
        
        Popular models:
        - "orca-mini-3b-gguf2-q4_0.gguf" (fast, good quality)
        - "gpt4all-falcon-q4_0.gguf" (falcon based)
        - "wizardlm-13b-v1.2.q4_0.gguf" (larger, better quality)
        - "nous-hermes-llama2-13b.q4_0.gguf" (instruction following)
        """
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load GPT4All model"""
        try:
            print(f"Loading GPT4All model: {self.model_name}")
            self.model = GPT4All(self.model_name)
            print("✅ GPT4All model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading GPT4All model: {e}")
            print("💡 Try downloading the model first:")
            print(f"   from gpt4all import GPT4All")
            print(f"   model = GPT4All('{self.model_name}')")
            return False
    
    def generate_response(self, prompt, max_tokens=100):
        """Generate response using GPT4All"""
        if not self.model:
            return "Model not loaded"
        
        try:
            response = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temp=0.7,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1
            )
            return response.strip()
        except Exception as e:
            return f"Error generating response: {e}"
    
    def analyze_seed_batch(self, good_seeds, bad_seeds, total_analyzed):
        """Specialized function for seed analysis"""
        quality_ratio = (good_seeds / max(total_analyzed, 1)) * 100
        
        prompt = f"""
        Seed Quality Analysis Report:
        - Good seeds: {good_seeds}
        - Bad seeds: {bad_seeds}
        - Total analyzed: {total_analyzed}
        - Quality ratio: {quality_ratio:.1f}%
        
        Please provide a brief assessment of this seed batch quality and any recommendations:
        """
        
        return self.generate_response(prompt, max_tokens=150)
    
    def generate_session_summary(self, session_data):
        """Generate a session summary"""
        prompt = f"""
        Seed Detection Session Summary:
        Duration: {session_data.get('duration', 'N/A')} minutes
        Total frames: {session_data.get('frames', 'N/A')}
        Average FPS: {session_data.get('fps', 'N/A')}
        Seeds detected: {session_data.get('total_seeds', 'N/A')}
        Good seeds: {session_data.get('good_seeds', 'N/A')}
        Bad seeds: {session_data.get('bad_seeds', 'N/A')}
        
        Provide a concise summary of this detection session:
        """
        
        return self.generate_response(prompt, max_tokens=120)

# Integration with your main app
def integrate_gpt4all_with_detection():
    """Example of integrating GPT4All with seed detection"""
    
    # Initialize LLM
    llm = GPT4AllLLM()
    
    # Example seed detection data
    seed_data = {
        'good_seeds': 25,
        'bad_seeds': 5,
        'total_analyzed': 30
    }
    
    # Get analysis
    analysis = llm.analyze_seed_batch(
        seed_data['good_seeds'],
        seed_data['bad_seeds'], 
        seed_data['total_analyzed']
    )
    
    print("🤖 GPT4All Analysis:")
    print(analysis)
    
    # Session summary example
    session_data = {
        'duration': 5.2,
        'frames': 780,
        'fps': 25.1,
        'total_seeds': 30,
        'good_seeds': 25,
        'bad_seeds': 5
    }
    
    summary = llm.generate_session_summary(session_data)
    print("\n📊 Session Summary:")
    print(summary)

if __name__ == "__main__":
    integrate_gpt4all_with_detection()
