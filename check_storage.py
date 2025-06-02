#!/usr/bin/env python3
"""
Simple storage check script for RunPod deployment
"""

import os
import shutil
import subprocess
import json

def check_all_storage():
    """Check all available storage locations"""
    print("🔍 Checking RunPod Storage Configuration...")
    print("=" * 50)
    
    # Check main filesystem
    print("\n📁 Main Filesystem (/)")
    try:
        total, used, free = shutil.disk_usage("/")
        print(f"  Total: {total / (1024**3):.2f} GB")
        print(f"  Used:  {used / (1024**3):.2f} GB")
        print(f"  Free:  {free / (1024**3):.2f} GB")
        print(f"  Usage: {(used / total) * 100:.1f}%")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Check runpod volume
    print("\n💾 RunPod Volume (/runpod-volume)")
    if os.path.exists("/runpod-volume"):
        try:
            total, used, free = shutil.disk_usage("/runpod-volume")
            print(f"  ✅ Volume mounted!")
            print(f"  Total: {total / (1024**3):.2f} GB")
            print(f"  Used:  {used / (1024**3):.2f} GB")
            print(f"  Free:  {free / (1024**3):.2f} GB")
            print(f"  Usage: {(used / total) * 100:.1f}%")
            
            # Check if we can write to it
            test_file = "/runpod-volume/test_write.txt"
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                print(f"  ✅ Write permissions: OK")
            except Exception as e:
                print(f"  ❌ Write permissions: {e}")
                
        except Exception as e:
            print(f"  ❌ Error accessing volume: {e}")
    else:
        print("  ❌ Volume NOT mounted!")
    
    # Check environment variables
    print("\n🔧 Environment Variables")
    env_vars = [
        "TRANSFORMERS_CACHE",
        "HF_HOME", 
        "MODEL_PATH",
        "SUPABASE_URL"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "Not set")
        if len(value) > 50:
            value = value[:47] + "..."
        print(f"  {var}: {value}")
    
    # Check disk usage with df
    print("\n📊 Disk Usage (df -h)")
    try:
        result = subprocess.run(['df', '-h'], capture_output=True, text=True, timeout=10)
        print(result.stdout)
    except Exception as e:
        print(f"  ❌ df command failed: {e}")
    
    # Check mount points
    print("\n🔗 Mount Points")
    try:
        result = subprocess.run(['mount'], capture_output=True, text=True, timeout=10)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'runpod' in line.lower() or '/dev/' in line:
                print(f"  {line}")
    except Exception as e:
        print(f"  ❌ mount command failed: {e}")
    
    # Check if models directory exists
    print("\n🤖 Model Cache Locations")
    cache_dirs = [
        "/runpod-volume/models--codellama--CodeLlama-13b-Instruct-hf",
        "/runpod-volume/models--sentence-transformers--all-mpnet-base-v2",
        "/cache",
        "/root/.cache/huggingface"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(cache_dir)
                          for filename in filenames)
                print(f"  ✅ {cache_dir}: {size / (1024**3):.2f} GB")
            except:
                print(f"  ✅ {cache_dir}: exists (size unknown)")
        else:
            print(f"  ❌ {cache_dir}: not found")

if __name__ == "__main__":
    check_all_storage()
