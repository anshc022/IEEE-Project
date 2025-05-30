#!/usr/bin/env python3
"""
LLM Setup Script for Seed Detection System
Helps configure and test different LLM options
"""

import subprocess
import sys
import requests
import os
from pathlib import Path

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✅ Ollama is running!")
            if models:
                print("Available models:")
                for model in models:
                    print(f"  - {model['name']}")
            else:
                print("⚠️  No models installed. Run: ollama pull llama3.2:3b")
            return True
    except:
        pass
    
    print("❌ Ollama not running or not installed")
    print("Install from: https://ollama.ai/download")
    return False

def install_python_deps():
    """Install required Python packages"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python dependencies installed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def test_huggingface():
    """Test Hugging Face transformers setup"""
    try:
        from transformers import AutoTokenizer
        print("✅ Hugging Face Transformers available")
        return True
    except ImportError:
        print("❌ Hugging Face Transformers not available")
        return False

def test_gpt4all():
    """Test GPT4All setup"""
    try:
        from gpt4all import GPT4All
        print("✅ GPT4All available")
        return True
    except ImportError:
        print("❌ GPT4All not available")
        return False

def recommend_setup():
    """Provide setup recommendations"""
    print("\n" + "="*50)
    print("LLM SETUP RECOMMENDATIONS")
    print("="*50)
    
    print("\n🚀 RECOMMENDED: Ollama (Easiest)")
    print("1. Download Ollama: https://ollama.ai/download")
    print("2. Install a model: ollama pull llama3.2:3b")
    print("3. Run: python app_with_llm.py")
    
    print("\n🔧 ALTERNATIVE: Hugging Face (More Control)")
    print("1. Models work offline after first download")
    print("2. Run: python llm_huggingface.py")
    
    print("\n💾 OFFLINE: GPT4All (Completely Offline)")
    print("1. Downloads models automatically")
    print("2. Run: python llm_gpt4all.py")
    
    print("\n📁 YOUR FILES:")
    print("- app_with_llm.py     (Main LLM-integrated system)")
    print("- llm_huggingface.py  (Hugging Face implementation)")
    print("- llm_gpt4all.py      (GPT4All implementation)")

def main():
    print("🤖 LLM Setup Checker for Seed Detection System")
    print("=" * 50)
    
    # Check current status
    print("\n📋 Checking current setup...")
    
    ollama_ok = check_ollama()
    python_deps_ok = install_python_deps()
    hf_ok = test_huggingface()
    gpt4all_ok = test_gpt4all()
    
    print(f"\n📊 STATUS SUMMARY:")
    print(f"Ollama:              {'✅' if ollama_ok else '❌'}")
    print(f"Python Dependencies: {'✅' if python_deps_ok else '❌'}")
    print(f"Hugging Face:        {'✅' if hf_ok else '❌'}")
    print(f"GPT4All:             {'✅' if gpt4all_ok else '❌'}")
    
    recommend_setup()
    
    # Quick test option
    if ollama_ok:
        test_ollama = input("\n🧪 Test Ollama integration? (y/n): ").lower().strip()
        if test_ollama == 'y':
            test_ollama_integration()

def test_ollama_integration():
    """Quick test of Ollama integration"""
    try:
        payload = {
            "model": "llama3.2:3b",
            "prompt": "Analyze this seed detection data: 15 good seeds, 3 bad seeds detected. Provide a brief quality assessment.",
            "stream": False
        }
        
        print("Testing Ollama integration...")
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json().get('response', 'No response')
            print(f"✅ LLM Response: {result[:100]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
