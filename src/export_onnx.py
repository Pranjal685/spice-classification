"""
src/export_onnx.py
------------------
Exports the fine-tuned PyTorch model to ONNX, verifies numerical parity,
and benchmarks CPU inference latency.
"""

import os
import time
import torch
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

from src.model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
ONNX_PATH = PROJECT_ROOT / "models" / "model.onnx"

def main():
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    # MobileNetV3-Small, 14 classes
    model = build_model("mobilenet_v3_small", 14, pretrained=False)
    
    # Load weights
    state_dict = torch.load(str(MODEL_PATH), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"Successfully exported ONNX model to {ONNX_PATH}")

    # Verify ONNX model
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    
    # Parity check
    print("\n--- Numerical Parity Verification ---")
    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    
    test_input = torch.randn(1, 3, 224, 224)
    ort_inputs = {"input": test_input.numpy()}
    
    with torch.no_grad():
        pt_out = model(test_input).numpy()
        
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"Maximum absolute difference between PT and ONNX: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("[SUCCESS] Parity verified!")
    else:
        print("[WARNING] Parity check failed (diff >= 1e-4)!")
        
    # Latency Benchmark
    print("\n--- Latency Benchmark (CPU) ---")
    iterations = 50
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)
        _ = ort_session.run(None, ort_inputs)
        
    # PT benchmark
    t0 = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(test_input)
    pt_time = (time.time() - t0) / iterations
    
    # ONNX benchmark
    t0 = time.time()
    for _ in range(iterations):
        _ = ort_session.run(None, ort_inputs)
    ort_time = (time.time() - t0) / iterations
    
    print(f"PyTorch CPU Latency:      {pt_time*1000:.2f} ms")
    print(f"ONNX Runtime CPU Latency: {ort_time*1000:.2f} ms")
    print(f"Speedup:                  {pt_time/ort_time:.2f}x")

if __name__ == "__main__":
    main()
