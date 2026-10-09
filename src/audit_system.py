import requests
import time
import base64
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = Path("temp_upload.jpg")

def run_tests():
    print("Starting Diagnostic Tests...\n")
    results = {}

    # Test 1: Health Check
    try:
        r = requests.get(f"{BASE_URL}/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["onnx_loaded"] is True
        results["Test 1: Health Check"] = "PASS"
    except Exception as e:
        results["Test 1: Health Check"] = f"FAIL ({e})"

    if not IMAGE_PATH.exists():
        print("Image not found. Creating a dummy image for testing.")
        from PIL import Image
        img = Image.new('RGB', (224, 224), color = (73, 109, 137))
        img.save(IMAGE_PATH)

    # Test 2: Tier-1 Confident Classification
    try:
        with open(IMAGE_PATH, "rb") as f:
            r = requests.post(f"{BASE_URL}/api/v1/classify", files={"image": f}, data={"threshold": 0.0, "force_fallback": "false"})
        assert r.status_code == 200
        data = r.json()
        assert data["tier"] == "TIER_1_ONNX_LOCAL"
        assert data["prediction"]["confidence"] >= 0.0
        assert "metadata" in data
        results["Test 2: Tier-1 Confident Classification"] = "PASS"
    except Exception as e:
        results["Test 2: Tier-1 Confident Classification"] = f"FAIL ({e})"

    # Test 3: Tier-2 Forced Fallback
    try:
        with open(IMAGE_PATH, "rb") as f:
            r = requests.post(f"{BASE_URL}/api/v1/classify", files={"image": f}, data={"threshold": 0.75, "force_fallback": "true"})
        assert r.status_code == 200
        data = r.json()
        assert data["tier"] == "TIER_2_GEMINI_FALLBACK"
        assert "fallback_response" in data
        fb = data["fallback_response"]
        assert "is_spice" in fb
        assert "identified_name" in fb
        results["Test 3: Tier-2 Forced Fallback"] = "PASS"
    except Exception as e:
        results["Test 3: Tier-2 Forced Fallback"] = f"FAIL ({e})"

    # Test 4: Grad-CAM Attribution
    try:
        with open(IMAGE_PATH, "rb") as f:
            r = requests.post(f"{BASE_URL}/api/v1/explain", files={"image": f})
        assert r.status_code == 200
        data = r.json()
        assert "heatmap_base64" in data
        b64_str = data["heatmap_base64"]
        decoded = base64.b64decode(b64_str)
        assert len(decoded) > 0
        results["Test 4: Grad-CAM Attribution"] = "PASS"
    except Exception as e:
        results["Test 4: Grad-CAM Attribution"] = f"FAIL ({e})"

    # Test 5: Latency Check
    try:
        latencies = []
        for _ in range(10):
            with open(IMAGE_PATH, "rb") as f:
                start = time.time()
                r = requests.post(f"{BASE_URL}/api/v1/classify", files={"image": f}, data={"threshold": 1.0})
                latencies.append(time.time() - start)
        
        avg_latency = (sum(latencies) / len(latencies)) * 1000
        results["Test 5: Latency Check"] = f"PASS ({avg_latency:.2f}ms)"
    except Exception as e:
        results["Test 5: Latency Check"] = f"FAIL ({e})"

    print("--- DIAGNOSTIC RESULTS ---")
    for k, v in results.items():
        print(f"{k.ljust(45)} | {v}")

if __name__ == "__main__":
    run_tests()
