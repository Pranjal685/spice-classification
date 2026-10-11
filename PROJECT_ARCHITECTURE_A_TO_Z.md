# Indian Spice Classification & Knowledge Retrieval System
## Complete A-to-Z Technical Architecture, Implementation, and Operations Manual

---

### Executive Overview

The **Indian Spice Classification & Knowledge Retrieval System** is an enterprise-grade, edge-optimized Computer Vision and Multimodal AI system. It is specifically engineered to solve the challenging problem of **Fine-Grained Visual Categorization (FGVC)** for 14 botanically and commercially critical Indian spices—including morphologically similar cultivars such as four distinct varieties of red chillies.

The system is built on a **two-tier decoupled architecture**:
1. **Tier-1 (Edge ONNX Runtime)**: Deterministic, sub-15ms CPU inference using a fine-tuned `MobileNetV3-Small` convolutional neural network running entirely on the edge with zero cloud dependency.
2. **Tier-2 (Multimodal VLM Fallback)**: Intelligent arbitration via Google Gemini 2.5 Flash Vision, triggered only when local model confidence drops below a calibrated decision boundary ($\tau = 75\%$) or when an Out-of-Distribution (OOD) non-spice sample is suspected.
3. **Explainable AI (Grad-CAM)**: Real-time visual attribution backpropagated to the final convolutional block (`features[-1]`), synthesizing 256-color OpenCV JET thermal saliency maps to verify physical botanical contours against background artifacts.
4. **Modern Developer-Grade Client**: A zero-framework Vanilla JS/CSS web interface featuring side-by-side visual image comparison, thermal heatmap toggles, dynamic gating thresholds, and a clean systems telemetry inspection dialog.

---

## 1. System Architecture & High-Level Topology

```mermaid
flowchart TD
    User([User / Web Client]) -->|1. Upload Image + Threshold| API[FastAPI Gateway :8000]
    
    subgraph Tier1_Pipeline ["Tier-1 Edge Pipeline (Local CPU)"]
        API -->|2. Preprocess| Norm[Resize 256 -> CenterCrop 224 -> ImageNet Norm]
        Norm -->|3. Forward Pass| ONNX[ONNX Runtime Session : CPUExecutionProvider]
        ONNX -->|4. Softmax Probabilities| Gate{Confidence >= 75% Gate?}
    end
    
    Gate -->|YES : Path A| FastResult[Local Prediction + Top-3 + Botanical KB Lookup]
    Gate -->|NO or Force Fallback : Path B| Tier2_VLM
    
    subgraph Tier2_Pipeline ["Tier-2 Multimodal Arbitration (Cloud / VLM)"]
        Tier2_VLM[SpiceFallbackEngine : Gemini 2.5 Flash]
        Tier2_VLM --> Prompt[Structured Taxonomy System Prompt]
        Prompt --> VLMCheck{Is Authentic Spice?}
        VLMCheck -->|NO| OOD[OOD Non-Spice Rejection Card]
        VLMCheck -->|YES| VLMResult[Zero-Shot Botanical Taxonomic Resolution]
    end
    
    subgraph XAI_Pipeline ["Explainable AI Subsystem (Grad-CAM)"]
        API -->|POST /api/v1/explain| TorchCAM[PyTorch Native CAM Model]
        TorchCAM --> Hook[Hook features[-1] Conv Block : 576 Channels]
        Hook --> GradCalc[Pooled Gradients * Activations]
        GradCalc --> ColorMap[OpenCV COLORMAP_JET Synthesis]
        ColorMap --> Overlay[Alpha-Blended Heatmap + Pure JET Colormap]
    end
    
    FastResult --> UI[Results Dashboard & Comparison View]
    VLMResult --> UI
    OOD --> UI
    Overlay --> UI
```

---

## 2. Dataset Engineering & Taxonomy Disambiguation

### 2.1 The 14 Target Spice Categories
The dataset represents a carefully balanced collection of Indian spices, curated from 15 initial raw collection folders into 14 disambiguated classification targets:

| Index | Class Identifier | Common / English Name | Hindi / Regional Name | Scientific Binomial | Key Morphological Features |
| :---: | :--- | :--- | :--- | :--- | :--- |
| `0` | `black_cardamom` | Black Cardamom | Badi Elaichi / Moti Elaichi | *Amomum subulatum* | Large (20–30mm), dark brown to charcoal, thick ribbed outer skin. |
| `1` | `black_pepper` | Black Pepper | Kali Mirch | *Piper nigrum* | Small (3–5mm), spherical, wrinkled black-brown corns. |
| `2` | `chilli_byadgi` | Byadgi Chilli | Byadgi Mirch | *Capsicum annuum* | Long, deep crimson red, heavily crinkled/wrinkled surface, curved tip, low heat. |
| `3` | `chilli_guntur` | Guntur Sannam | Guntur Mirch | *Capsicum annuum* | Medium-thick, bright red, smooth surface, short pedicel, high heat. |
| `4` | `chilli_lavangi` | Lavangi Chilli | Lavangi Mirch | *Capsicum annuum* | Very small, conical, light red/orange, pungent, commonly used in Kolhapuri cuisine. |
| `5` | `chilli_sankeshwari` | Sankeshwari Chilli | Sankeshwari Mirch | *Capsicum annuum* | Extremely long, slender, wrinkled, orange-red hue, very pungent. |
| `6` | `cumin` | Cumin Seeds | Jeera | *Cuminum cyminum* | Small, elongated boat-shaped, longitudinal ribs, greyish-brown. |
| `7` | `dagad_phool` | Stone Flower | Kalpasi / Pathar Phool | *Parmotrema perlatum* | Dry, papery, foliose lichen with mottled greenish-black and grey patches. |
| `8` | `dalchini` | Cinnamon / Cassia | Dalchini | *Cinnamomum cassia / zeylanicum* | Thick, rough, curled reddish-brown bark quills. |
| `9` | `dhaniya` | Coriander Seeds | Sabut Dhaniya | *Coriandrum sativum* | Globular, hollow, vertical ribbed surface, light golden brown. |
| `10` | `dried_coconut` | Dry Copra / Coconut | Sukha Khopara | *Cocos nucifera* | Hard, curved white/creamy inner meat with fibrous dark brown outer husk. |
| `11` | `elaichi_dana` | Green Cardamom Seeds | Elaichi Dana | *Elettaria cardamomum* | Small, angular, sticky, dark brown/black loose seeds stripped of outer pod. |
| `12` | `elaichi_pod` | Green Cardamom Pod | Chhoti Elaichi | *Elettaria cardamomum* | Small (8–12mm), ovoid, pale green, papery tri-valved whole capsule pod. |
| `13` | `halkund` | Whole Dried Turmeric | Sabut Haldi / Halkund | *Curcuma longa* | Hard, cylindrical, knobby rhizome fingers with deep golden-yellow core. |

### 2.2 Taxonomy Challenges Solved
1. **Duplicate Merging**: The raw data contained two independent folders for cinnamon (`Dalchini 26` and `Dalchini(1) 61`). These were unified into the single class `dalchini` with normalized metadata.
2. **Anatomical Segregation**: Whole green cardamom pods (`elaichi_pod`) and loose cardamom seeds (`elaichi_dana`) were separated into separate classes to prevent multi-label ambiguity.
3. **Fine-Grained Chilli Differentiation**: Distinguishing four regional chilli cultivars (*Byadgi*, *Guntur*, *Lavangi*, *Sankeshwari*) requires deep convolutional representations capturing wrinkle frequency, aspect ratio, and color saturation.
4. **Class Imbalance Mitigation**: Samples per class varied from 13 to 86. Inverse-frequency class weights were computed and injected into the loss function during model training:
   $$w_c = \frac{N_{\text{total}}}{C \cdot N_c}, \quad \tilde{w}_c = \frac{w_c}{\frac{1}{C}\sum_k w_k}$$

---

## 3. Deep Learning Architecture & Training Pipeline

### 3.1 Neural Backbone (`src/model.py`)
- **Backbone**: `MobileNetV3-Small` (ImageNet pretrained initialization).
- **Core Building Blocks**: Inverted Bottleneck Residual Units with depthwise separable convolutions and Squeeze-and-Excitation (SE) channel attention modules.
- **Classification Head Adaptation**:
  ```python
  model = models.mobilenet_v3_small(weights="DEFAULT")
  in_features = model.classifier[3].in_features  # 1024
  model.classifier[3] = nn.Linear(in_features, num_classes=14)
  ```
  Hardswish non-linear activations, Batch Normalization, and Dropout ($p=0.2$) are preserved.

### 3.2 Two-Stage Training Protocol (`src/train.py`)
1. **Stage 1 (Linear Probing)**: The convolutional backbone is completely frozen (`freeze_backbone(model, True)`). Only the modified 14-class linear head is trained for 5 epochs using AdamW ($lr = 1\times 10^{-3}$) to initialize head weights without destroying pretrained feature maps.
2. **Stage 2 (Full Fine-Tuning)**: All layers are unfrozen (`freeze_backbone(model, False)`). The entire network is trained for 25 epochs using Cosine Annealing Learning Rate scheduling ($lr_{\min} = 1\times 10^{-6}$) with class-weighted Cross-Entropy loss.
3. **Data Augmentations**:
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `RandomHorizontalFlip(p=0.5)`
   - `RandomVerticalFlip(p=0.2)`
   - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
   - `RandomAffine(degrees=15, shear=5)`
   - Normalization with ImageNet constants ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).

### 3.3 ONNX Export & Optimization (`src/export_onnx.py`)
The best model checkpoint (`models/best_model.pth`) is exported to Open Neural Network Exchange (ONNX) format:
- **Opset Version**: 17
- **Dynamic Batch Axis**:
  ```python
  torch.onnx.export(
      model,
      dummy_input,  # shape: [1, 3, 224, 224]
      "models/model.onnx",
      input_names=["input"],
      output_names=["output"],
      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
      opset_version=17,
  )
  ```
- **Validation**: Numerical parity check between PyTorch and ONNX Runtime ensures $\max|\text{logits}_{\text{PT}} - \text{logits}_{\text{ONNX}}| < 1\times 10^{-4}$.

---

## 4. Explainable AI: Grad-CAM Implementation

### 4.1 Theory & Mathematical Formulation
Grad-CAM (Gradient-weighted Class Activation Mapping) calculates the visual importance of convolutional feature maps for a target prediction class $c$:

1. **Class Score Gradient**: The gradient of the unnormalized class score $Y^c$ is computed with respect to feature map activations $A^k$ of the final convolutional layer:
   $$\frac{\partial Y^c}{\partial A_{i,j}^k}$$
2. **Global Average Pooling of Gradients (Importance Weight)**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial Y^c}{\partial A_{i,j}^k}$$
   where $Z = U \times V$ is the spatial area of the feature map (typically $7 \times 7$).
3. **Weighted Forward Activation Map with Rectified Linear Unit (ReLU)**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k \right)$$
   The $\text{ReLU}$ ensures that only features positively contributing to class $c$ are preserved, discarding features associated with other classes.

### 4.2 Hook Placement (`src/explain.py`)
In `MobileNetV3-Small`, the chosen layer is `model.features[-1]`:
- This is the final $1 \times 1$ convolutional block preceding global average pooling.
- Output activation volume: $[1, 576, 7, 7]$ (576 channels at spatial dimension $7 \times 7$).
- Forward hook intercepts $A^k$; full backward hook intercepts $\frac{\partial Y^c}{\partial A^k}$.

### 4.3 Thermal Colormap Synthesis
1. Normalized $7 \times 7$ saliency tensor is scaled to $[0, 1]$ and resized to the original image dimensions using bilinear interpolation.
2. Quantized to an 8-bit unsigned integer array $[0, 255]$.
3. Transformed via OpenCV's JET thermal palette:
   ```python
   colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
   ```
4. Blended with the original BGR image using linear alpha-weighting:
   ```python
   blended = cv2.addWeighted(original_bgr, 0.6, colormap, 0.4, 0)
   ```
5. Both the **Blended Heatmap** and the **Pure JET Colormap** are encoded to Base64 strings and returned to the frontend.

### 4.4 Memory Stability
PyTorch autograd graphs retain computation trees unless cleaned. `src/explain.py` explicitly executes:
```python
del input_tensor, logits, class_score
activations = None
gradients = None
```
Automated testing confirms zero memory drift ($\Delta < 0.5\text{ MB}$) across 25 consecutive cycles.

---

## 5. Tier-2 Multimodal Fallback Engine (`src/fallback.py`)

### 5.1 Operating Principles & Decision Boundaries
- When the local ONNX model achieves $\ge 75.0\%$ confidence, the query is resolved entirely on CPU in $< 5\text{ ms}$.
- If confidence $< 75.0\%$ (or if the user toggles `force_fallback=True`), the system routes the request to Tier-2:
  ```python
  if top_1_prob >= threshold and not force_fallback:
      # Route to Tier-1 ONNX
  else:
      # Escalate to Tier-2 Gemini Fallback
  ```

### 5.2 Multimodal Prompt Architecture
The system sends the raw image to Google Gemini 2.5 Flash with a strict JSON contract prompt:
```json
{
  "is_spice": true,
  "identified_name": "Dalchini (Cinnamon/Cassia)",
  "scientific_name": "Cinnamomum cassia",
  "is_in_dataset_scope": true,
  "confidence_assessment": "Coarse, thick bark rolls exhibiting distinct cassia quill morphology.",
  "flavor_profile": "Sweet, aromatic, warm, pungent.",
  "common_uses": "Garam masala, bakery, chai blends.",
  "action_recommendation": "Authentic spice verified. Can proceed with recipe inclusion."
}
```

### 5.3 Out-of-Distribution (OOD) Protection
If a user uploads a non-spice image (e.g., an apple, a watch, or random noise):
1. The local model might output low-confidence spurious predictions with diffuse Grad-CAM heatmaps.
2. The gating threshold intercepts this ($< 75\%$).
3. Gemini evaluates `is_spice: false`, triggering the **OOD Rejection Interface** in the UI:
   - Sets title to "Non-Spice Subject".
   - Explains non-spice visual characteristics.
   - Instructs the user to provide an authentic culinary spice sample.

### 5.4 Production Resilience Features
- **Key Rotation**: Supports comma-separated keys (`GEMINI_API_KEYS=key1,key2`) with automatic round-robin failover on HTTP 429 rate limits.
- **Model Cascade**: Automatically tries `gemini-2.5-flash` first; on failure, falls back to `gemini-flash-latest`.
- **STUB Mode**: If no API key is supplied, gracefully degrades to a structured STUB mock response without throwing fatal 500 crashes.

---

## 6. Frontend Architecture & Design System

### 6.1 Technology Philosophy
- **Stack**: Pure Vanilla HTML5, Vanilla CSS3 (Custom Properties & Tokens), Vanilla ES6+ JavaScript.
- **Zero Overhead**: Zero external frameworks, zero Node.js server dependencies, zero client compilation steps.
- **Fluid Smooth Scrolling**: Integrated Lenis.js with strict modal scroll isolation (`data-lenis-prevent`).

### 6.2 Key UI Components
1. **Comparison Section (Side-by-Side Dual Viewport)**:
   - Displays the original spice image on the left (`#preview-img`) and the Grad-CAM thermal attribution heatmap on the right (`#heatmap-img`).
   - Includes a stateful view switcher:
     - **Grad-CAM Overlay**: Blended composite image ($\alpha = 0.6 / 0.4$).
     - **Grad-CAM Pure JET**: 100% thermal colormap without background image blending.
2. **Interactive Gating Slider**:
   - Custom range slider ($50\% - 95\%$) dynamically adjusting the decision threshold parameter submitted with inference queries.
3. **Botanical Profile Card**:
   - Clean, scannable table for scientific taxon, morphology features, flavor profile, and culinary uses.
4. **Balanced Layout Hierarchy**:
   - 2-column grid layout where the right card balances the height of the left image-comparison card.

### 6.3 Modern Technical Inspection Modal
*(Completely overhauled to eliminate AI-generated look and botanical clutter)*
- **Header**: Inference telemetry kicker with live status dot.
- **4-Tile Telemetry Stat Strip**:
  - `Identified Target`: Predicted spice name with tier status.
  - `Model Confidence`: Monospace percentage with pass/fail gate evaluation.
  - `Saliency Layer Hook`: Convolutional hook target (`features[-1]`, 576 channels).
  - `Execution Latency`: Millisecond timer for edge execution.
- **4-Stage Horizontal Stepper**:
  - `01 Tensor Normalization` ($224 \times 224 \times 3$ RGB, ImageNet $\mu,\sigma$).
  - `02 Feature Extraction` (11 inverted residual blocks, 576 maps).
  - `03 Decision Boundary` ($75.0\%$ threshold gate).
  - `04 Grad-CAM Attribution` (OpenCV JET thermal colormap).
- **Full-Width Systems Specification Table**:
  - Deep learning architecture details (Backbone, Tensor Shape, Preprocessing Norm, Layer Hook, Engine, Gating Threshold, Latency SLA, Thermal Colormap).
- **Mathematical Attribution Card**:
  - LaTeX-formatted Grad-CAM equations displaying importance weights $\alpha_k^c$ and saliency map $L_{\text{Grad-CAM}}^c$.
- **Zero Botanical Duplication**:
  - Botanical details remain on the primary dashboard card where they belong for culinary context; model inspection remains strictly technical.

---

## 7. API Contract & Endpoint Reference

### `GET /api/v1/health`
Checks system telemetry and readiness of all three internal engines.
```json
{
  "status": "healthy",
  "onnx_loaded": true,
  "cam_loaded": true,
  "fallback_ready": true
}
```

### `POST /api/v1/classify`
- **Request Form**:
  - `image`: UploadFile (binary image/jpeg, image/png)
  - `threshold`: float (default `0.75`)
  - `force_fallback`: bool (default `false`)
- **Tier-1 Response Example**:
  ```json
  {
    "tier": "TIER_1_ONNX_LOCAL",
    "prediction": {
      "class_id": "dalchini",
      "display_name": "Dalchini (Cinnamon/Cassia)",
      "confidence": 0.8872
    },
    "top_3": [
      { "class_id": "dalchini", "display_name": "Dalchini (Cinnamon/Cassia)", "probability": 0.8872 },
      { "class_id": "black_cardamom", "display_name": "Black Cardamom", "probability": 0.0412 },
      { "class_id": "halkund", "display_name": "Whole Dried Turmeric", "probability": 0.0215 }
    ],
    "metadata": {
      "scientific_name": "Cinnamomum cassia / zeylanicum",
      "flavor_profile": "Sweet, warm, and woody.",
      "common_uses": "Desserts, curries, and chai blends.",
      "key_visual_features": "Curled, reddish-brown bark pieces."
    },
    "latency_ms": 3.97,
    "model_parameters": {
      "backbone": "MobileNetV3-Small",
      "runtime_engine": "ONNX Runtime (CPUExecutionProvider)",
      "input_tensor_shape": "1 × 3 × 224 × 224",
      "normalization": "ImageNet (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])",
      "decision_boundary": "75%",
      "feature_extractor_layer": "model.features[-1] (576 channels)",
      "active_classes_count": 14
    }
  }
  ```

### `POST /api/v1/explain`
- **Request Form**:
  - `image`: UploadFile
  - `target_class_idx`: optional string or int (defaults to predicted top class)
- **Response Example**:
  ```json
  {
    "class_id": "dalchini",
    "display_name": "Dalchini (Cinnamon/Cassia)",
    "confidence": 0.8872,
    "heatmap_base64": "/9j/4AAQSkZJRgABAQAAAQ...",
    "thermal_base64": "/9j/4AAQSkZJRgABAQAAAQ...",
    "xai_parameters": {
      "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
      "target_layer": "model.features[-1] (Final InvertedResidual Block)",
      "feature_channels": 576,
      "input_resolution": "224x224x3 (ImageNet RGB Normalized)",
      "colormap": "OpenCV COLORMAP_JET (Thermal Pseudo-Color)",
      "backbone": "MobileNetV3-Small"
    }
  }
  ```

---

## 8. Verification, Testing & Benchmark Metrics

### 8.1 Empirical Benchmark Summary
| Metric | Benchmark Result | Engineering Significance |
| :--- | :---: | :--- |
| **Top-1 Test Accuracy** | **96.55%** | Evaluated on unseen test split across all 14 fine-grained classes. |
| **Macro F1-Score** | **0.9738** | Balanced performance across classes regardless of sample count skew. |
| **Gated Precision ($\ge 75\%$)** | **100.0%** | Zero false positives when the edge gating condition is satisfied. |
| **Edge Coverage Rate** | **91.4%** | Over 91% of typical user queries are resolved locally on CPU in $< 5\text{ ms}$. |
| **ONNX Inference Latency** | **3.97 ms** | Measured on standard multi-core CPU with zero discrete GPU required. |
| **Grad-CAM Latency** | **28.4 ms** | Full forward pass + backpropagation + colormap generation. |
| **Memory Leak Stability** | **$\Delta < 0.5\text{ MB}$** | Verified across 25 consecutive attribution runs with graph cleanup. |

### 8.2 Test Suite Architecture (21 / 21 Tests Passing)
The test suite is structured across five dedicated test files:
1. `tests/test_contracts.py`: Validates schema integrity, response status codes, payload structures for `/health`, `/classify`, and `/explain`.
2. `tests/test_pipeline.py`: Validates numerical parity between PyTorch and ONNX, softmax probability constraints ($\sum p_i = 1.0$), and top-3 ranking correctness.
3. `tests/test_performance.py`: Asserts sub-15ms edge inference latency and leak-free memory consumption across repeated Grad-CAM evaluations.
4. `tests/test_edge_cases.py`: Validates system behavior against corrupt images, non-image binaries, extreme aspect ratios, out-of-bounds threshold inputs ($< 0.0$ or $> 1.0$).
5. `src/test_fallback.py`: Tests live VLM fallback queries, STUB mode fallback behavior, and schema contract conformance.

---

## 9. Deployment & Operations Guide

### 9.1 Local Development
```bash
# 1. Activate isolated Python environment
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux / macOS

# 2. Configure environment variables in .env
GEMINI_API_KEY=your_actual_gemini_api_key_here

# 3. Launch FastAPI development server with hot-reload
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

### 9.2 Running Automated Tests
```bash
# Execute the full 21-test pytest suite
pytest -v
```

### 9.3 Production Docker Deployment
The application includes a production-ready `Dockerfile` optimized for memory-constrained environments (<1 GB RAM):
- Utilizes official lightweight `python:3.10-slim` base image.
- Installs CPU-only wheels for PyTorch and TorchVision (`--extra-index-url https://download.pytorch.org/whl/cpu`) reducing container size from ~4.5 GB to ~900 MB.
- Runs under an unprivileged `appuser` security account.

```bash
# Build the container
docker build -t spice-classifier:latest .

# Run container with environment configuration
docker run -d -p 8000:8000 \
  --name spice-classifier \
  -e GEMINI_API_KEY="your_gemini_api_key" \
  spice-classifier:latest
```

---

## 10. File & Directory Inventory

```
spice-classification/
├── .env                          # Local environment secrets (GEMINI_API_KEY)
├── .env.example                  # Template configuration
├── dataset_master.csv            # Master dataset catalog with split annotations
├── Dockerfile                    # Production lightweight Docker container specification
├── requirements.txt              # Production dependency specifications
├── README.md                     # High-level repository documentation
├── PROJECT_ARCHITECTURE_A_TO_Z.md# This comprehensive architecture manual
│
├── data/
│   ├── class_indices.json        # Class-to-index (0-13) bidirectional mappings
│   └── spice_knowledge.json      # Botanical metadata, taxon, morphology & flavor attributes
│
├── models/
│   ├── best_model.pth            # Fine-tuned PyTorch checkpoint (MobileNetV3-Small)
│   ├── model.onnx                # Exported optimized ONNX model graph
│   └── model.onnx.data           # External tensor weights file for ONNX
│
├── src/
│   ├── __init__.py               # Python package indicator
│   ├── api.py                    # FastAPI gateway, routes, and lifespan manager
│   ├── model.py                  # PyTorch model builder & backbone freeze logic
│   ├── train.py                  # Two-stage training pipeline & scheduler
│   ├── explain.py                # Grad-CAM PyTorch hooks & OpenCV JET colormap
│   ├── fallback.py               # Tier-2 Gemini 2.5 Flash multimodal fallback engine
│   ├── export_onnx.py            # ONNX dynamic-batch export & numerical validation
│   ├── dataset.py                # PyTorch Dataset loader & transform pipelines
│   ├── evaluate.py               # Accuracy, F1-score, and confusion matrix evaluation
│   ├── utils.py                  # Class-weighted loss calculation & device helper
│   ├── build_dataset.py          # Script to index images into dataset_master.csv
│   └── test_fallback.py          # Pytest suite for fallback engine & VLM contracts
│
├── static/
│   ├── index.html                # Semantic client dashboard with comparison & modal
│   ├── styles.css                # Enterprise design system, layout, and responsive tokens
│   └── app.js                    # UI state management, API requests, and modal controls
│
└── tests/
    ├── conftest.py               # Test fixtures, dummy image generators, test client
    ├── test_contracts.py         # API contract & schema verification tests
    ├── test_pipeline.py          # Forward pass, ONNX parity, and probability tests
    ├── test_performance.py       # Latency SLAs and memory stability tests
    └── test_edge_cases.py        # Malformed files, invalid formats, boundary stress tests
```

---

## 11. Summary of Recent Improvements

1. **Eliminated AI-Generated Look**: Replaced the verbose narrative banner in the inspection dialog with a 4-tile **Telemetry Stat Strip** modeled after enterprise observability tools (Weights & Biases, Triton Inference Server).
2. **Removed Botanical Metadata from the Inspection Modal**: The inspection dialog is now strictly reserved for deep learning engineering details (tensors, hooks, latency, gating, math). Botanical details (scientific binomials, flavor profiles, culinary uses) remain exclusively on the main results card where they serve practical culinary context without technical duplication.
3. **Integrated Dual-Viewport Visual Comparison**: Original spice image and Grad-CAM thermal attribution heatmap are displayed side-by-side in a single unified view with a toggle between blended heatmap and pure JET thermal spectrum.
4. **4-Stage Sequential Architecture**: Replaced generic steps with a streamlined 4-stage pipeline stepper: Tensor Normalization $\to$ Feature Extraction $\to$ Decision Boundary Gate $\to$ Grad-CAM Attribution.
5. **Rigorous Grad-CAM Mathematical Formulation**: Added formal attribution equations directly into the technical specifications modal.
6. **Isolated Scroll Mechanics**: Configured wheel and touch listeners with overscroll containment so scrolling within the modal never scrolls the background page.
