import os
from pathlib import Path

def main():
    print("Starting Pre-Flight Verification for Hugging Face Spaces...")
    
    project_root = Path(__file__).resolve().parent.parent
    
    required_artifacts = [
        "models/model.onnx",
        "models/best_model.pth",
        "data/class_indices.json",
        "data/spice_knowledge.json",
        "static/index.html",
        "static/styles.css",
        "static/app.js"
    ]
    
    missing = []
    for artifact in required_artifacts:
        path = project_root / artifact
        if not path.exists():
            missing.append(artifact)
        else:
            print(f" [PASS] Artifact found: {artifact}")
            
    if missing:
        print("\n [FAIL] Missing required deployment artifacts:")
        for m in missing:
            print(f"   - {m}")
        exit(1)
        
    dockerignore_path = project_root / ".dockerignore"
    if not dockerignore_path.exists():
        print(" [FAIL] .dockerignore not found!")
        exit(1)
        
    print(f" [PASS] Artifact found: .dockerignore")
        
    with open(dockerignore_path, "r") as f:
        dockerignore_content = f.read()
        
    required_ignores = ["data/raw/", "tests/", ".env"]
    missing_ignores = []
    for req in required_ignores:
        if req not in dockerignore_content:
            missing_ignores.append(req)
            
    if missing_ignores:
        print("\n [FAIL] .dockerignore is missing critical exclusions:")
        for m in missing_ignores:
            print(f"   - {m}")
        exit(1)
        
    print(" [PASS] .dockerignore excludes required paths.")
    
    # Check Dockerfile port
    dockerfile_path = project_root / "Dockerfile"
    if not dockerfile_path.exists():
        print(" [FAIL] Dockerfile not found!")
        exit(1)
        
    with open(dockerfile_path, "r") as f:
        dockerfile_content = f.read()
        
    if "EXPOSE 7860" not in dockerfile_content or "PORT=7860" not in dockerfile_content:
         print(" [FAIL] Dockerfile is not configured for port 7860!")
         exit(1)
         
    print(" [PASS] Dockerfile port configured to 7860.")
    print("\nAll Pre-Flight Checks Passed Successfully! Ready for Hugging Face Spaces deployment.")

if __name__ == "__main__":
    main()
