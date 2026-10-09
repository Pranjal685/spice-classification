import json
from pathlib import Path
import os

def test_class_mapping_symmetry():
    class_indices_path = Path("data/class_indices.json")
    assert class_indices_path.exists(), "class_indices.json must exist"
    
    with open(class_indices_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    class_to_idx = data["class_to_idx"]
    idx_to_class = data["idx_to_class"]
    
    # Assert exactly 14 unique classes
    assert len(class_to_idx) == 14, f"Expected 14 classes, found {len(class_to_idx)}"
    assert len(idx_to_class) == 14, f"Expected 14 indices, found {len(idx_to_class)}"
    
    # Assert bi-directional mapping without collisions
    for cls_name, idx in class_to_idx.items():
        assert str(idx) in idx_to_class, f"Index {idx} for {cls_name} not found in idx_to_class"
        assert idx_to_class[str(idx)] == cls_name, f"Mismatch: {idx_to_class[str(idx)]} != {cls_name}"

def test_knowledge_base_completeness():
    class_indices_path = Path("data/class_indices.json")
    knowledge_path = Path("data/spice_knowledge.json")
    
    assert knowledge_path.exists(), "spice_knowledge.json must exist"
    
    with open(class_indices_path, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)["class_to_idx"]
        
    with open(knowledge_path, 'r', encoding='utf-8') as f:
        knowledge = json.load(f)
        
    class_names = list(class_indices.keys())
    
    for class_name in class_names:
        assert class_name in knowledge, f"Class '{class_name}' is missing in spice_knowledge.json"
        entry = knowledge[class_name]
        
        # Assert required fields are present and not empty
        required_fields = ["display_name", "scientific_name", "flavor_profile", "common_uses", "key_visual_features"]
        for field in required_fields:
            assert field in entry, f"Missing field '{field}' for class '{class_name}'"
            assert entry[field] is not None and str(entry[field]).strip() != "", f"Empty field '{field}' for class '{class_name}'"

def test_model_artifact_availability():
    onnx_path = Path("models/model.onnx")
    pth_path = Path("models/best_model.pth")
    
    assert onnx_path.exists(), "models/model.onnx must exist"
    assert onnx_path.stat().st_size > 0, "models/model.onnx must not be empty"
    
    assert pth_path.exists(), "models/best_model.pth must exist"
    assert pth_path.stat().st_size > 0, "models/best_model.pth must not be empty"
