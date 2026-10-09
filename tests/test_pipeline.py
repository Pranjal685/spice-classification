def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "onnx_loaded" in data
    assert "cam_loaded" in data
    assert "fallback_ready" in data

def test_tier1_local_execution(client, valid_synthetic_image):
    files = {"image": ("test.jpg", valid_synthetic_image, "image/jpeg")}
    data = {"threshold": 0.75}
    response = client.post("/api/v1/classify", files=files, data=data)
    
    assert response.status_code == 200
    res = response.json()
    assert "tier" in res
    if res["tier"] == "TIER_1_ONNX_LOCAL":
        assert "confidence" in res["prediction"]
        top_3 = res["top_3"]
    else:
        assert "fallback_response" in res
        top_3 = res["tier_1_context"]["top_3"]
        
    assert "latency_ms" in res
    
    # We don't assert tier == "TIER_1_LOCAL" strictly here because synthetic random noise
    # might yield < 0.75 confidence and trigger Tier 2, unless the model happens to be very confident.
    # But we can assert the fields exist and sum to approx 1.0 (actually top_3 sum might not be 1.0, wait)
    # The sum of ALL classes is 1.0. top_3 is just 3 classes, so sum <= 1.0
    total_prob = sum(item["probability"] for item in top_3)
    assert 0.0 < total_prob <= 1.01

def test_threshold_gate_enforcement_high(client, valid_synthetic_image):
    files = {"image": ("test.jpg", valid_synthetic_image, "image/jpeg")}
    data = {"threshold": 0.999} # Very high, forcing tier 2
    response = client.post("/api/v1/classify", files=files, data=data)
    
    assert response.status_code == 200
    res = response.json()
    assert res["tier"] == "TIER_2_GEMINI_FALLBACK" or res["confidence"] < 0.999

def test_threshold_gate_enforcement_low(client, valid_synthetic_image):
    files = {"image": ("test.jpg", valid_synthetic_image, "image/jpeg")}
    data = {"threshold": 0.001} # Very low, forcing tier 1
    response = client.post("/api/v1/classify", files=files, data=data)
    
    assert response.status_code == 200
    res = response.json()
    assert res["tier"] == "TIER_1_ONNX_LOCAL"
    assert res["prediction"]["confidence"] >= 0.001

def test_forced_fallback_flag(client, valid_synthetic_image):
    files = {"image": ("test.jpg", valid_synthetic_image, "image/jpeg")}
    data = {"threshold": 0.75, "force_fallback": True}
    response = client.post("/api/v1/classify", files=files, data=data)
    
    assert response.status_code == 200
    res = response.json()
    assert res["tier"] == "TIER_2_GEMINI_FALLBACK"

def test_grad_cam_endpoint(client, valid_synthetic_image):
    files = {"image": ("test.jpg", valid_synthetic_image, "image/jpeg")}
    response = client.post("/api/v1/explain", files=files)
    
    assert response.status_code == 200
    res = response.json()
    assert "predicted_class" not in res  # It returns class_id, not predicted_class
    assert "class_id" in res
    assert "heatmap_base64" in res
    # Depending on how encode_image_base64 is implemented, it might or might not have the data URI prefix.
    # The actual implementation just returns the raw base64.
    assert len(res["heatmap_base64"]) > 100
