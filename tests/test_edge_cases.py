def test_non_image_file_rejection(client):
    files = {"image": ("test.txt", b"This is a text file", "text/plain")}
    response = client.post("/api/v1/classify", files=files)
    
    # Assert that the API handles it gracefully instead of 500
    assert response.status_code == 400
    assert "error" in response.json() or "detail" in response.json()

def test_corrupted_payload_upload(client, corrupted_image_bytes):
    files = {"image": ("test.jpg", corrupted_image_bytes, "image/jpeg")}
    response = client.post("/api/v1/classify", files=files)
    
    # Handling PIL.UnidentifiedImageError or similar -> 400
    assert response.status_code == 400

def test_zero_byte_upload(client, zero_byte_image):
    files = {"image": ("test.jpg", zero_byte_image, "image/jpeg")}
    response = client.post("/api/v1/classify", files=files)
    
    assert response.status_code == 400

def test_solid_monochromatic_black(client, monochromatic_black):
    files = {"image": ("black.jpg", monochromatic_black, "image/jpeg")}
    response = client.post("/api/v1/classify", files=files)
    
    # Should not crash with divide-by-zero, should either route to low-confidence Tier-1 or Tier-2 fallback safely
    assert response.status_code == 200

def test_solid_monochromatic_white(client, monochromatic_white):
    files = {"image": ("white.jpg", monochromatic_white, "image/jpeg")}
    response = client.post("/api/v1/classify", files=files)
    
    # Should not crash with divide-by-zero, should either route to low-confidence Tier-1 or Tier-2 fallback safely
    assert response.status_code == 200

def test_extreme_aspect_ratio_wide(client, extreme_wide_image):
    files = {"image": ("wide.jpg", extreme_wide_image, "image/jpeg")}
    response = client.post("/api/v1/classify", files=files)
    
    # Resizing shouldn't collapse the dimension
    assert response.status_code == 200

def test_extreme_aspect_ratio_tall(client, extreme_tall_image):
    files = {"image": ("tall.jpg", extreme_tall_image, "image/jpeg")}
    response = client.post("/api/v1/classify", files=files)
    
    assert response.status_code == 200
