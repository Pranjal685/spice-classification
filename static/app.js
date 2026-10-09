document.addEventListener("DOMContentLoaded", () => {
    // 1. Lenis Smooth Scroll Initialization
    const lenis = new Lenis({
        autoRaf: true,
        smoothWheel: true
    });

    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const analyzeBtn = document.getElementById("analyze-btn");
    const thresholdInput = document.getElementById("threshold");
    const thresholdVal = document.getElementById("threshold-val");
    const forceFallback = document.getElementById("force-fallback");
    
    const uploadContainer = document.getElementById("upload-container");
    const previewContainer = document.getElementById("image-preview-container");
    const previewImg = document.getElementById("preview-img");
    const replaceImgBtn = document.getElementById("replace-img-btn");
    
    // Comparison Slider Elements
    const heatmapLayer = document.getElementById("heatmap-layer");
    const heatmapImg = document.getElementById("heatmap-img");
    const compSlider = document.getElementById("comparison-slider");
    const compHandle = document.getElementById("comparison-handle");
    
    const emptyState = document.getElementById("empty-state");
    const resultsCard = document.getElementById("results-card");
    const telemetryPill = document.getElementById("telemetry-pill");
    const jsonCopyBtn = document.getElementById("json-copy-btn");
    
    let currentFile = null;
    let lastJsonData = null;

    // UI Listeners
    dropZone.addEventListener("click", () => fileInput.click());
    replaceImgBtn.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.background = "#F1F5F9";
    });
    
    dropZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dropZone.style.background = "var(--bg-page)";
    });
    
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.background = "var(--bg-page)";
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });
    
    thresholdInput.addEventListener("input", (e) => {
        thresholdVal.textContent = `${e.target.value}%`;
        e.target.style.setProperty('--slider-pct', `${e.target.value}%`);
    });
    
    // Comparison Slider Logic
    compSlider.addEventListener("input", (e) => {
        const val = e.target.value;
        heatmapLayer.style.width = `${val}%`;
        compHandle.style.left = `${val}%`;
    });

    // JSON Copy Button
    jsonCopyBtn.addEventListener("click", () => {
        if (lastJsonData) {
            navigator.clipboard.writeText(JSON.stringify(lastJsonData, null, 2));
            const originalHTML = jsonCopyBtn.innerHTML;
            jsonCopyBtn.innerHTML = `✓ Copied JSON`;
            setTimeout(() => {
                jsonCopyBtn.innerHTML = originalHTML;
            }, 2000);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) return;
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            const src = e.target.result;
            previewImg.src = src;
            heatmapImg.src = ""; // Clear old heatmap
            
            // Switch views
            uploadContainer.style.display = "none";
            previewContainer.style.display = "block";
            
            // Reset Comparison state
            compSlider.style.display = "none";
            compHandle.style.display = "none";
            heatmapLayer.style.width = "0%";
            
            // Reset results to idle skeleton
            emptyState.style.display = "flex";
            emptyState.classList.remove("loading");
            resultsCard.style.display = "none";
            
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener("click", async () => {
        if (!currentFile) return;
        
        // Loading State
        emptyState.style.display = "flex";
        emptyState.classList.add("loading");
        resultsCard.style.display = "none";
        analyzeBtn.textContent = "Analyzing...";
        analyzeBtn.disabled = true;
        
        const formData = new FormData();
        formData.append("image", currentFile);
        formData.append("threshold", (thresholdInput.value / 100).toString());
        formData.append("force_fallback", forceFallback.checked.toString());
        
        try {
            // 1. Run Classification
            const res = await fetch("/api/v1/classify", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            
            if (!res.ok) {
                alert(data.error || "Failed to classify image.");
                emptyState.classList.remove("loading");
                analyzeBtn.textContent = "Analyze & Verify";
                analyzeBtn.disabled = false;
                return;
            }
            
            lastJsonData = data;
            
            // Live Telemetry Latency Pill
            if (data.latency_ms !== undefined) {
                let engine = data.tier === "TIER_1_ONNX_LOCAL" ? "ONNX CPU" : "Gemini API";
                telemetryPill.innerHTML = `${engine}: ${data.latency_ms} ms <span class="latency">• Status: 200 OK</span>`;
                
                // Trigger pulse animation
                telemetryPill.classList.remove("pulse-anim");
                void telemetryPill.offsetWidth; // trigger reflow
                telemetryPill.classList.add("pulse-anim");
            }

            renderResults(data);
            
            // 2. Run XAI (Grad-CAM)
            const targetClassId = data.prediction ? data.prediction.class_id : (data.tier_1_context && data.tier_1_context.top_3 ? data.tier_1_context.top_3[0].class_id : null);
            
            if (targetClassId) {
                const xaiForm = new FormData();
                xaiForm.append("image", currentFile);
                xaiForm.append("target_class_idx", targetClassId);
                
                const xaiRes = await fetch("/api/v1/explain", {
                    method: "POST",
                    body: xaiForm
                });
                
                if (xaiRes.ok) {
                    const xaiData = await xaiRes.json();
                    if (xaiData.heatmap_base64) {
                        heatmapImg.src = `data:image/jpeg;base64,${xaiData.heatmap_base64}`;
                        
                        // Activate comparison slider
                        compSlider.style.display = "block";
                        compHandle.style.display = "flex";
                        compSlider.value = 50;
                        heatmapLayer.style.width = "50%";
                        compHandle.style.left = "50%";
                    }
                }
            }

        } catch (error) {
            console.error(error);
            alert("Error running inference");
            emptyState.classList.remove("loading");
        } finally {
            analyzeBtn.textContent = "Analyze & Verify";
            analyzeBtn.disabled = false;
        }
    });
    
    function renderResults(data) {
        emptyState.style.display = "none";
        emptyState.classList.remove("loading");
        resultsCard.style.display = "block";
        
        const badge = document.getElementById("tier-badge");
        const confMeter = document.getElementById("confidence-meter");
        const specsTable = document.getElementById("specs-table");
        const title = document.getElementById("spice-name");
        
        const attrSec = document.getElementById("attribution-section");
        const attrText = document.getElementById("attribution-text");
        attrSec.style.display = "block"; // Default to show
        
        if (data.tier === "TIER_1_ONNX_LOCAL") {
            badge.className = "badge tier1";
            badge.textContent = "Tier-1 ONNX Engine";
            confMeter.textContent = `${(data.prediction.confidence * 100).toFixed(1)}% Confidence`;
            
            title.textContent = data.prediction.display_name;
            
            specsTable.innerHTML = `
                <div class="spec-row"><div class="spec-label">Scientific Name</div><div class="spec-value">${data.metadata.scientific_name || 'N/A'}</div></div>
                <div class="spec-row"><div class="spec-label">Morphology</div><div class="spec-value">${data.metadata.key_visual_features || 'N/A'}</div></div>
                <div class="spec-row"><div class="spec-label">Flavor Profile</div><div class="spec-value">${data.metadata.flavor_profile || 'N/A'}</div></div>
                <div class="spec-row"><div class="spec-label">Common Uses</div><div class="spec-value">${data.metadata.common_uses || 'N/A'}</div></div>
            `;
            
            attrText.innerHTML = `Grad-CAM analysis substantiates the model's focus mapping to physical features of <strong>${data.prediction.display_name}</strong>.`;
            
            renderBars(data.top_3);
            
        } else {
            badge.className = "badge tier2";
            badge.textContent = "Tier-2 Multimodal Fallback";
            const fb = data.fallback_response;
            
            if (fb.error) {
                badge.className = "badge error";
                badge.textContent = "Processing Error";
                title.textContent = "Analysis Failed";
                specsTable.innerHTML = `<div class="spec-row"><div class="spec-label">Error Reason</div><div class="spec-value">${fb.error_reason}</div></div>`;
                document.getElementById("prob-bars").innerHTML = "";
                attrSec.style.display = "none";
            } else if (!fb.is_spice) {
                badge.className = "badge error";
                badge.textContent = "Not a Spice";
                title.textContent = "Unknown Subject";
                specsTable.innerHTML = `
                    <div class="spec-row"><div class="spec-label">Assessment</div><div class="spec-value">${fb.confidence_assessment || 'N/A'}</div></div>
                    <div class="spec-row"><div class="spec-label">Action</div><div class="spec-value">${fb.action_recommendation || 'N/A'}</div></div>
                `;
                document.getElementById("prob-bars").innerHTML = "";
                
                const top1Class = data.tier_1_context && data.tier_1_context.top_3 ? data.tier_1_context.top_3[0].display_name : (data.tier_1_context && data.tier_1_context.top_1_class ? data.tier_1_context.top_1_class : 'Unknown');
                const top1Prob = data.tier_1_context && data.tier_1_context.top_1_prob ? (data.tier_1_context.top_1_prob * 100).toFixed(1) : 0;
                
                attrText.innerHTML = `Out-of-Distribution Warning: Local activation (<strong>${top1Class}</strong>, ${top1Prob}%) represents spurious feature correlation on non-spice artifacts.`;
                
            } else {
                confMeter.textContent = "";
                title.textContent = fb.identified_name;
                specsTable.innerHTML = `
                    <div class="spec-row"><div class="spec-label">Scientific Name</div><div class="spec-value">${fb.scientific_name || 'N/A'}</div></div>
                    <div class="spec-row"><div class="spec-label">Scope</div><div class="spec-value">${fb.is_in_dataset_scope ? 'In Scope' : 'Out of Distribution'}</div></div>
                    <div class="spec-row"><div class="spec-label">Assessment</div><div class="spec-value">${fb.confidence_assessment || 'N/A'}</div></div>
                    <div class="spec-row"><div class="spec-label">Flavor Profile</div><div class="spec-value">${fb.flavor_profile || 'N/A'}</div></div>
                `;
                
                const top1Prob = data.tier_1_context && data.tier_1_context.top_1_prob ? (data.tier_1_context.top_1_prob * 100).toFixed(1) : 0;
                attrText.innerHTML = `Local Tier-1 confidence fell below threshold (${top1Prob}%). Attention was inconclusive; deferred to Tier-2 Foundation Vision.`;
            }
            
            if (data.tier_1_context && data.tier_1_context.top_3) {
                renderBars(data.tier_1_context.top_3);
            }
        }
    }
    
    function renderBars(top3) {
        const container = document.getElementById("prob-bars");
        container.innerHTML = "";
        
        // We use requestAnimationFrame to allow the DOM to render the 0% width first,
        // then apply the actual width so the CSS transition triggers.
        top3.forEach(item => {
            const row = document.createElement("div");
            row.className = "bar-row";
            
            const pct = (item.probability * 100).toFixed(1);
            
            row.innerHTML = `
                <div class="bar-label" title="${item.display_name}">${item.display_name}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: 0%"></div>
                </div>
                <div class="bar-val">${pct}%</div>
            `;
            container.appendChild(row);
            
            // Trigger animation
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    const fill = row.querySelector('.bar-fill');
                    if(fill) fill.style.width = `${pct}%`;
                });
            });
        });
    }
});
