import io
import pytest
from PIL import Image
import random
from fastapi.testclient import TestClient
from src.api import app

@pytest.fixture(scope="session")
def client():
    # Properly initialized lifespan state is handled automatically by TestClient context manager
    with TestClient(app) as client:
        yield client

def generate_synthetic_image(width, height, color=None, random_pixels=False):
    if color:
        img = Image.new("RGB", (width, height), color)
    else:
        img = Image.new("RGB", (width, height))
        
    if random_pixels:
        pixels = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

@pytest.fixture
def valid_synthetic_image():
    return generate_synthetic_image(500, 500, random_pixels=True)

@pytest.fixture
def extreme_wide_image():
    return generate_synthetic_image(1200, 40, random_pixels=True)

@pytest.fixture
def extreme_tall_image():
    return generate_synthetic_image(40, 1200, random_pixels=True)

@pytest.fixture
def monochromatic_black():
    return generate_synthetic_image(500, 500, color=(0, 0, 0))

@pytest.fixture
def monochromatic_white():
    return generate_synthetic_image(500, 500, color=(255, 255, 255))

@pytest.fixture
def corrupted_image_bytes():
    return b"This is not a valid JPEG stream, just some corrupted payload."

@pytest.fixture
def zero_byte_image():
    return b""
