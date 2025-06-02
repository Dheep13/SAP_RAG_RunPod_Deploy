#!/usr/bin/env python3
"""
Test script for the RunPod handler
Run this locally to test your handler before deploying
"""

import sys
import os
import json
from typing import Dict, Any

# Add the parent directory to sys.path so we can import handler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from handler import handler, load_models
except ImportError as e:
    print(f"❌ Error importing handler: {e}")
    print("Make sure handler.py is in the same directory")
    sys.exit(1)

def test_embeddings():
    """Test embeddings generation"""
    print("🧪 Testing embeddings generation...")
    
    test_input = {
        "input": {
            "text": "This is a test sentence for generating embeddings.",
            "task": "embeddings"
        }
    }
    
    result = handler(test_input)
    
    if "error" in result:
        print(f"❌ Embeddings test failed: {result['error']}")
        return False
    
    embeddings = result.get("output", {}).get("embeddings", [])
    if embeddings and len(embeddings) > 0:
        print(f"✅ Embeddings generated: {len(embeddings)} dimensions")
        print(f"   First 5 values: {embeddings[:5]}")
        return True
    else:
        print("❌ No embeddings generated")
        return False

def test_tokenization():
    """Test tokenization"""
    print("\n🧪 Testing tokenization...")
    
    test_input = {
        "input": {
            "text": "Hello world, this is a tokenization test!",
            "task": "tokenize"
        }
    }
    
    result = handler(test_input)
    
    if "error" in result:
        print(f"❌ Tokenization test failed: {result['error']}")
        return False
    
    tokens = result.get("output", {}).get("tokens", [])
    if tokens:
        print(f"✅ Tokenization successful: {len(tokens)} tokens")
        print(f"   Tokens: {tokens}")
        return True
    else:
        print("❌ No tokens generated")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n🧪 Testing error handling...")
    
    # Test with empty input
    test_input = {
        "input": {
            "text": "",
            "task": "embeddings"
        }
    }
    
    result = handler(test_input)
    
    if "error" in result:
        print("✅ Error handling works correctly for empty input")
        return True
    else:
        print("❌ Error handling failed - should have returned error for empty input")
        return False

def test_invalid_task():
    """Test invalid task handling"""
    print("\n🧪 Testing invalid task handling...")
    
    test_input = {
        "input": {
            "text": "Test text",
            "task": "invalid_task"
        }
    }
    
    result = handler(test_input)
    
    if "error" in result:
        print("✅ Invalid task handling works correctly")
        return True
    else:
        print("❌ Invalid task handling failed")
        return False

def performance_test():
    """Test performance with larger text"""
    print("\n🧪 Testing performance with larger text...")
    
    large_text = "This is a performance test. " * 100  # 500+ words
    
    test_input = {
        "input": {
            "text": large_text,
            "task": "embeddings"
        }
    }
    
    import time
    start_time = time.time()
    result = handler(test_input)
    end_time = time.time()
    
    if "error" not in result:
        print(f"✅ Performance test completed in {end_time - start_time:.2f} seconds")
        return True
    else:
        print(f"❌ Performance test failed: {result['error']}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting RunPod Handler Tests")
    print("=" * 50)
    
    # First, try to load models
    print("📦 Loading models...")
    if not load_models():
        print("❌ Failed to load models. Cannot proceed with tests.")
        return
    
    print("✅ Models loaded successfully!")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Embeddings", test_embeddings),
        ("Tokenization", test_tokenization), 
        ("Error Handling", test_error_handling),
        ("Invalid Task", test_invalid_task),
        ("Performance", performance_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for deployment!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Fix issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)