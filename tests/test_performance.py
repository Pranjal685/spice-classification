import psutil
import os
import time

def test_inference_latency_sla(client, valid_synthetic_image):
    latencies = []
    
    # Run 30 consecutive classifications
    for _ in range(30):
        files = {"image": ("test.jpg", valid_synthetic_image, "image/jpeg")}
        # Force Tier-1 by using 0.001 threshold so we strictly test ONNX compute latency
        data = {"threshold": 0.001}
        response = client.post("/api/v1/classify", files=files, data=data)
        
        assert response.status_code == 200
        res = response.json()
        assert res["tier"] == "TIER_1_ONNX_LOCAL"
        
        # Collect latency reported by API (should be just the compute time)
        # Or measure round-trip. The prompt asks for "average ONNX compute latency remains strictly below 25 ms"
        latencies.append(res.get("latency_ms", 0))
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage ONNX compute latency across 30 runs: {avg_latency:.2f} ms")
    
    # Assert SLA (adjust bounds slightly for GH Actions/CI if needed, but per prompt < 25ms)
    assert avg_latency < 25.0, f"Latency SLA failed: {avg_latency:.2f} ms >= 25 ms"


def test_grad_cam_memory_stability(client, valid_synthetic_image):
    process = psutil.Process(os.getpid())
    
    # Do a warmup run to load models and initialize caches
    files = {"image": ("test.jpg", valid_synthetic_image, "image/jpeg")}
    client.post("/api/v1/explain", files=files)
    
    # Get baseline memory
    baseline_memory = process.memory_info().rss
    
    # Run 25 consecutive explain calls
    for _ in range(25):
        files = {"file": ("test.jpg", valid_synthetic_image, "image/jpeg")}
        client.post("/api/v1/explain", files=files)
    
    # Get final memory
    final_memory = process.memory_info().rss
    
    # Delta in MB
    delta_mb = (final_memory - baseline_memory) / (1024 * 1024)
    print(f"\nMemory Delta after 25 Grad-CAM runs: {delta_mb:.2f} MB")
    
    # If the system isn't clearing autograd graphs, memory will grow linearly by hundreds of MBs
    # We assert that growth is minimal (e.g. less than 50MB allowance for python GC variations)
    assert delta_mb < 50.0, f"Memory leak detected: grew by {delta_mb:.2f} MB"
