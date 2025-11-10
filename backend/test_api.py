"""
Quick test script for the FastAPI backend
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("="*60)
    print("  🧪 TESTING ALPHAREPS API")
    print("="*60)
    print()
    
    # Test 1: Root endpoint
    print("1️⃣  Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API is running!")
            print(f"   Version: {data.get('version')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   Make sure the API is running: python main.py")
        return
    
    print()
    
    # Test 2: Health check
    print("2️⃣  Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print(f"   ✅ Health check passed")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 3: Login
    print("3️⃣  Testing login endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={"name": "Test User", "role": "user"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Login successful")
            print(f"   User: {data['user']['name']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 4: Get exercises
    print("4️⃣  Testing exercises endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/exercises")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found {data['total_count']} exercises")
            for ex in data['exercises']:
                print(f"      • {ex}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 5: Get user stats
    print("5️⃣  Testing user stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/user/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Stats retrieved")
            print(f"      Total workouts: {data['totalWorkouts']}")
            print(f"      Total reps: {data['totalReps']}")
            print(f"      Avg accuracy: {data['avgAccuracy']}%")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("="*60)
    print("  ✅ API TESTS COMPLETE")
    print("="*60)
    print()
    print("🚀 Next steps:")
    print("   1. Start frontend: cd frontend && npm run dev")
    print("   2. Open browser: http://localhost:3000")
    print()

if __name__ == "__main__":
    test_api()
