# test_api.py
import requests
import json

# Your RunPod endpoint URL
ENDPOINT_URL = "https://api.runpod.ai/v2//runsync"
API_KEY = "your-runpod-api-key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Test query
payload = {
    "input": {
        "query": "How do I create a secure CPI iFlow with error handling?",
        "component": "CPI"
    }
}

print("🧪 Testing SAP RAG endpoint...")
response = requests.post(ENDPOINT_URL, json=payload, headers=headers)

if response.status_code == 200:
    result = response.json()
    print("✅ Success!")
    print(f"Answer: {result.get('output', {}).get('answer', 'No answer')}")
    print(f"Confidence: {result.get('output', {}).get('confidence', 0)}")
    print(f"Sources: {len(result.get('output', {}).get('sources', []))}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)