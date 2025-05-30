#!/usr/bin/env python3
"""
LLM Demo for Seed Detection System
Demonstrates different LLM integration options without running the full camera system
"""

import time
import requests
from datetime import datetime

# Sample seed detection data for demonstration
SAMPLE_DATA = {
    'good_seeds': 23,
    'bad_seeds': 7,
    'total_analyzed': 30,
    'confidence_avg': 0.78,
    'session_time': 45.2
}

def demo_ollama():
    """Demo Ollama integration"""
    print("🦙 OLLAMA DEMO")
    print("=" * 40)
    
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not running. Start with: ollama serve")
            return
        
        # Demo prompts
        prompts = [
            f"Analyze this seed quality data: {SAMPLE_DATA['good_seeds']} good seeds, {SAMPLE_DATA['bad_seeds']} bad seeds detected. Overall quality assessment?",
            f"Based on {SAMPLE_DATA['total_analyzed']} seeds analyzed with average confidence {SAMPLE_DATA['confidence_avg']:.2f}, what recommendations would you give?",
            "What factors should be considered when sorting seeds for optimal crop yield?"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🤔 Query {i}: {prompt}")
            
            payload = {
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            }
            
            print("🤖 LLM thinking...")
            start_time = time.time()
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json().get('response', 'No response')
                print(f"✅ Response ({response_time:.1f}s): {result.strip()}")
            else:
                print(f"❌ Error: {response.status_code}")
            
            time.sleep(1)  # Small delay between requests
            
    except Exception as e:
        print(f"❌ Ollama demo failed: {e}")
        print("Make sure Ollama is installed and running: ollama serve")

def demo_huggingface():
    """Demo Hugging Face integration"""
    print("\n🤗 HUGGING FACE DEMO")
    print("=" * 40)
    
    try:
        from transformers import pipeline
        
        # Use a small, fast model for demo
        print("Loading model (this may take a moment on first run)...")
        generator = pipeline('text-generation', model='distilgpt2', max_length=100)
        
        prompts = [
            f"Seed analysis results: {SAMPLE_DATA['good_seeds']} good, {SAMPLE_DATA['bad_seeds']} bad seeds.",
            "Seed quality factors include"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🤔 Query {i}: {prompt}")
            
            start_time = time.time()
            result = generator(prompt, max_length=len(prompt.split()) + 30, num_return_sequences=1)
            response_time = time.time() - start_time
            
            generated_text = result[0]['generated_text']
            # Extract just the new part
            new_text = generated_text[len(prompt):].strip()
            print(f"✅ Response ({response_time:.1f}s): {prompt}{new_text}")
            
    except ImportError:
        print("❌ Transformers not installed. Run: pip install transformers")
    except Exception as e:
        print(f"❌ Hugging Face demo failed: {e}")

def demo_gpt4all():
    """Demo GPT4All integration"""
    print("\n🌐 GPT4ALL DEMO")
    print("=" * 40)
    
    try:
        from gpt4all import GPT4All
        
        print("Loading GPT4All model (downloads automatically on first run)...")
        model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
        
        prompts = [
            f"Analyze seed detection results: {SAMPLE_DATA['good_seeds']} good seeds, {SAMPLE_DATA['bad_seeds']} bad seeds from {SAMPLE_DATA['total_analyzed']} total. Assessment:",
            "What are key indicators of seed quality?"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🤔 Query {i}: {prompt}")
            
            start_time = time.time()
            with model.chat_session():
                response = model.generate(prompt, max_tokens=100, temp=0.7)
            response_time = time.time() - start_time
            
            print(f"✅ Response ({response_time:.1f}s): {response.strip()}")
            
    except ImportError:
        print("❌ GPT4All not installed. Run: pip install gpt4all")
    except Exception as e:
        print(f"❌ GPT4All demo failed: {e}")

def show_integration_comparison():
    """Show comparison of different LLM options"""
    print("\n📊 LLM INTEGRATION COMPARISON")
    print("=" * 50)
    
    comparison = [
        ("Feature", "Ollama", "Hugging Face", "GPT4All"),
        ("Setup Difficulty", "Easy", "Medium", "Easy"),
        ("Internet Required", "Initial only", "Initial only", "Initial only"),
        ("Model Variety", "Excellent", "Excellent", "Good"),
        ("Performance", "Fast", "Variable", "Medium"),
        ("Memory Usage", "Medium", "Variable", "Low"),
        ("Best For", "Production", "Custom models", "Offline use")
    ]
    
    for row in comparison:
        print(f"{row[0]:<17} | {row[1]:<10} | {row[2]:<13} | {row[3]:<10}")
        if row[0] == "Feature":
            print("-" * 60)

def main():
    print("🤖 LLM INTEGRATION DEMO")
    print("=" * 50)
    print(f"Sample Data: {SAMPLE_DATA}")
    
    # Demo all available integrations
    demo_ollama()
    demo_huggingface()
    demo_gpt4all()
    
    show_integration_comparison()
    
    print("\n🚀 NEXT STEPS:")
    print("1. Choose your preferred LLM option based on the demo results")
    print("2. For Ollama: Ensure it's running with your preferred model")
    print("3. Run the full system with: python app_with_llm.py")
    print("4. Use keyboard controls: 'l' for LLM mode, 'r' for reports")

if __name__ == "__main__":
    main()
