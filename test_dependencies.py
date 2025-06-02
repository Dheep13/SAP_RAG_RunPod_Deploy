#!/usr/bin/env python3
"""
Simple Dependency Test Script
Run this to test if all your packages are working together
"""

import sys
import subprocess
import importlib

def install_missing_packages():
    """Install missing packages"""
    packages_to_install = [
        "torch>=2.7.0",
        "transformers>=4.52.0", 
        "accelerate>=1.1.0",
        "langchain==0.3.16",
        "langchain-core==0.3.63", 
        "langchain-community==0.3.27",
        "langchain-huggingface==0.3.8",
        "sentence-transformers>=3.3.0",
        "supabase>=2.9.0",
        "python-dotenv>=1.0.1",
        "pydantic>=2.10.0",
        "huggingface-hub>=0.26.0",
        "datasets>=3.2.0"
    ]
    
    print("🔄 Installing/upgrading packages...")
    for package in packages_to_install:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade"])
            print(f"✅ {package}")
        except:
            print(f"❌ Failed: {package}")

def test_imports():
    """Test all package imports"""
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers', 
        'accelerate': 'Accelerate',
        'langchain': 'LangChain',
        'langchain_core': 'LangChain Core',
        'langchain_community': 'LangChain Community',
        'langchain_huggingface': 'LangChain HuggingFace',
        'sentence_transformers': 'Sentence Transformers',
        'supabase': 'Supabase',
        'numpy': 'NumPy',
        'pydantic': 'Pydantic',
        'huggingface_hub': 'HuggingFace Hub'
    }
    
    print("\n🔍 Testing imports...")
    print("=" * 40)
    
    results = {}
    for package, name in packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
            results[package] = True
        except ImportError as e:
            print(f"❌ {name}: {e}")
            results[package] = False
    
    return results

def test_versions():
    """Test package versions"""
    print("\n🔍 Testing versions...")
    print("=" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except:
        print("❌ PyTorch version check failed")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except:
        print("❌ Transformers version check failed")
    
    try:
        import langchain
        print(f"✅ LangChain: {langchain.__version__}")
    except:
        print("❌ LangChain version check failed")
    
    try:
        import sentence_transformers
        print(f"✅ Sentence Transformers: {sentence_transformers.__version__}")
    except:
        print("❌ Sentence Transformers version check failed")

def test_functionality():
    """Test basic functionality"""
    print("\n🔍 Testing functionality...")
    print("=" * 40)
    
    # Test PyTorch
    try:
        import torch
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        z = torch.mm(x, y)
        print("✅ PyTorch operations")
    except Exception as e:
        print(f"❌ PyTorch operations: {e}")
    
    # Test Transformers
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print("✅ Transformers model loading")
    except Exception as e:
        print(f"❌ Transformers model loading: {e}")
    
    # Test LangChain
    try:
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content="test")
        print("✅ LangChain basic functionality")
    except Exception as e:
        print(f"❌ LangChain basic functionality: {e}")
    
    # Test LangChain + HuggingFace
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ LangChain HuggingFace integration")
    except Exception as e:
        print(f"❌ LangChain HuggingFace integration: {e}")
    
    # Test Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers")
    except Exception as e:
        print(f"❌ Sentence Transformers: {e}")

def main():
    print("🚀 Dependency Test Script")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print("=" * 50)
    
    # Ask user if they want to install packages
    install = input("\n❓ Install/upgrade packages first? (y/n): ").lower().strip()
    if install == 'y':
        install_missing_packages()
    
    # Test imports
    results = test_imports()
    
    # Test versions
    test_versions()
    
    # Test functionality  
    test_functionality()
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 40)
    total = len(results)
    passed = sum(results.values())
    print(f"Packages tested: {total}")
    print(f"Successful: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} packages failed")
        print("Try running with package installation (y) option")

if __name__ == "__main__":
    main()