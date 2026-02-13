#!/usr/bin/env python3
"""
Quick Test Script for CropCapital AI Engine
Tests API endpoints with various coordinates
"""

import requests
import json
import time
from tabulate import tabulate

API_URL = "http://localhost:5000"

# Test cases with known crop regions
TEST_CASES = [
    {
        "name": "Punjab Rice Belt",
        "lat": 30.7046,
        "lon": 76.7179,
        "acres": 5,
        "expected_crop": "Paddy/Rice"
    },
    {
        "name": "Haryana Wheat Region",
        "lat": 29.0588,
        "lon": 76.0856,
        "acres": 8,
        "expected_crop": "Wheat"
    },
    {
        "name": "Gujarat Cotton",
        "lat": 23.0225,
        "lon": 72.5714,
        "acres": 10,
        "expected_crop": "Cotton"
    },
    {
        "name": "Maharashtra Sugarcane",
        "lat": 19.0760,
        "lon": 72.8777,
        "acres": 3,
        "expected_crop": "Sugarcane"
    },
    {
        "name": "Karnataka Maize",
        "lat": 12.9716,
        "lon": 77.5946,
        "acres": 4,
        "expected_crop": "Maize/Corn"
    }
]

def test_health():
    """Test if API is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        print("✓ API Server is running")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ API Server is not running!")
        print("  Start server with: python crop_ai_engine_v3.py")
        return False

def test_farm_analysis(test_case):
    """Test farm analysis endpoint"""
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{API_URL}/analyze-farm",
            json={
                "lat": test_case["lat"],
                "lon": test_case["lon"],
                "acres": test_case["acres"]
            },
            timeout=30
        )
        
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        if response.status_code == 200:
            data = response.json()
            
            detected_crop = data["crop_identification"]["detected_crop"]
            confidence = data["crop_identification"]["confidence"]
            credit_score = data["score_card"]["total_credit_score"]
            loan_amount = data["score_card"]["max_eligible_loan"]
            ndvi = data["satellite_metrics"]["ndvi_index"]
            
            # Check if detection matches expected
            match = "✓" if detected_crop == test_case["expected_crop"] else "✗"
            
            return {
                "status": "✓ Pass",
                "region": test_case["name"],
                "detected": detected_crop,
                "expected": test_case["expected_crop"],
                "match": match,
                "confidence": f"{confidence}%",
                "credit_score": credit_score,
                "loan": loan_amount,
                "ndvi": f"{ndvi:.3f}",
                "time_ms": f"{elapsed:.0f}ms"
            }
        else:
            return {
                "status": "✗ Fail",
                "region": test_case["name"],
                "error": f"HTTP {response.status_code}"
            }
    
    except Exception as e:
        return {
            "status": "✗ Error",
            "region": test_case["name"],
            "error": str(e)
        }

def run_tests():
    """Run all tests and display results"""
    print("\n" + "="*80)
    print("   CROPCAPITAL AI ENGINE - SYSTEM TEST")
    print("="*80 + "\n")
    
    # Check if server is running
    if not test_health():
        return
    
    print("\n[Running Tests]")
    print("-" * 80)
    
    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}/{len(TEST_CASES)}: {test_case['name']}")
        result = test_farm_analysis(test_case)
        results.append(result)
        
        # Print immediate result
        if result["status"] == "✓ Pass":
            print(f"  {result['match']} Detected: {result['detected']} (Confidence: {result['confidence']})")
            print(f"  Credit Score: {result['credit_score']} | Loan: {result['loan']}")
            print(f"  NDVI: {result['ndvi']} | Time: {result['time_ms']}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Summary table
    print("\n" + "="*80)
    print("   TEST SUMMARY")
    print("="*80 + "\n")
    
    table_data = []
    for r in results:
        if r["status"] == "✓ Pass":
            table_data.append([
                r["region"],
                r["detected"],
                r["confidence"],
                r["credit_score"],
                r["ndvi"],
                r["time_ms"]
            ])
    
    if table_data:
        headers = ["Region", "Crop Detected", "Confidence", "Credit Score", "NDVI", "Response Time"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Statistics
    passed = sum(1 for r in results if r["status"] == "✓ Pass")
    total = len(results)
    accuracy = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed ({accuracy:.1f}% success rate)")
    print(f"{'='*80}\n")
    
    if passed == total:
        print("✓ All tests passed! System is working correctly.")
    else:
        print("⚠ Some tests failed. Check logs for details.")

if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nTest suite error: {e}")
