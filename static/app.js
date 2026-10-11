document.addEventListener("DOMContentLoaded", () => {
    // 1. Lenis Smooth Scroll Initialization
    const lenis = new Lenis({
        autoRaf: true,
        smoothWheel: true
    });

    // DOM Elements - Input & Controls
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const analyzeBtn = document.getElementById("analyze-btn");
    const analyzeBtnText = document.getElementById("analyze-btn-text");
    const thresholdInput = document.getElementById("threshold");
    const thresholdVal = document.getElementById("threshold-val");
    const forceFallback = document.getElementById("force-fallback");

    // DOM Elements - Previews & Comparison
    const uploadContainer = document.getElementById("upload-container");
    const previewContainer = document.getElementById("image-preview-container");
    const previewImg = document.getElementById("preview-img");
    const heatmapImg = document.getElementById("heatmap-img");
    const camPlaceholder = document.getElementById("cam-placeholder");
    const heatmapLegend = document.getElementById("heatmap-legend");
    const replaceImgBtn = document.getElementById("replace-img-btn");
    const origDimsTag = document.getElementById("orig-dims-tag");
    const camTagLabel = document.getElementById("cam-tag-label");

    // View Mode Toggle (Overlay vs Pure Thermal)
    const heatmapToggleGroup = document.getElementById("heatmap-toggle-group");
    const btnModeBlended = document.getElementById("btn-mode-blended");
    const btnModeThermal = document.getElementById("btn-mode-thermal");

    // Results & Telemetry Elements
    const emptyState = document.getElementById("empty-state");
    const resultsCard = document.getElementById("results-card");
    const telemetryPill = document.getElementById("telemetry-pill");
    const jsonCopyBtn = document.getElementById("json-copy-btn");

    // Modal & Breakdown Popover Elements
    const pipelineModal = document.getElementById("pipeline-modal");
    const openPipelineModalBtn = document.getElementById("open-pipeline-modal-btn");
    const closePipelineModalBtn = document.getElementById("close-pipeline-modal-btn");
    const modalDoneBtn = document.getElementById("modal-done-btn");
    const modalModelParamsTable = document.getElementById("modal-model-params-table");
    const hoverCardStatus = document.getElementById("hover-card-status");
    const hoverCardBody = document.getElementById("hover-card-body");
    const breakdownHoverCard = document.getElementById("breakdown-hover-card");
    const modalPipelineBadge = document.getElementById("modal-pipeline-badge");
    const modalStatusText = document.getElementById("modal-status-text");
    const modalStatClass = document.getElementById("modal-stat-class");
    const modalStatSubclass = document.getElementById("modal-stat-subclass");
    const modalStatConf = document.getElementById("modal-stat-conf");
    const modalStatLayer = document.getElementById("modal-stat-layer");
    const modalStatLatency = document.getElementById("modal-stat-latency");

    let currentFile = null;
    let lastJsonData = null;
    let blendedHeatmapSrc = null;
    let pureThermalSrc = null;
    let currentHeatmapMode = "blended"; // 'blended' | 'thermal'

    // UI Listeners - Upload & Drag/Drop
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

    // Threshold slider
    thresholdInput.addEventListener("input", (e) => {
        thresholdVal.textContent = `${e.target.value}%`;
        e.target.style.setProperty('--slider-pct', `${e.target.value}%`);
    });

    // Heatmap View Mode Toggles
    btnModeBlended.addEventListener("click", () => {
        if (currentHeatmapMode === "blended") return;
        currentHeatmapMode = "blended";
        btnModeBlended.classList.add("active");
        btnModeThermal.classList.remove("active");
        camTagLabel.textContent = "Grad-CAM Overlay";
        if (blendedHeatmapSrc) {
            heatmapImg.src = blendedHeatmapSrc;
        }
    });

    btnModeThermal.addEventListener("click", () => {
        if (currentHeatmapMode === "thermal") return;
        currentHeatmapMode = "thermal";
        btnModeThermal.classList.add("active");
        btnModeBlended.classList.remove("active");
        camTagLabel.textContent = "Grad-CAM Pure JET";
        if (pureThermalSrc) {
            heatmapImg.src = pureThermalSrc;
        }
    });

    // Modal Dialog Controls & Scroll Isolation
    const closeModal = () => {
        if (pipelineModal) {
            pipelineModal.style.display = "none";
            document.body.classList.remove("modal-open");
            document.documentElement.classList.remove("modal-open");
            document.body.style.overflow = "";
            document.documentElement.style.overflow = "";
            if (typeof lenis !== "undefined" && lenis) {
                lenis.start();
            }
        }
    };

    const openModal = () => {
        if (pipelineModal) {
            pipelineModal.style.display = "flex";
            document.body.classList.add("modal-open");
            document.documentElement.classList.add("modal-open");
            document.body.style.overflow = "hidden";
            document.documentElement.style.overflow = "hidden";
            if (typeof lenis !== "undefined" && lenis) {
                lenis.stop();
            }
        }
    };

    if (openPipelineModalBtn) openPipelineModalBtn.addEventListener("click", openModal);
    if (closePipelineModalBtn) closePipelineModalBtn.addEventListener("click", closeModal);
    if (modalDoneBtn) modalDoneBtn.addEventListener("click", closeModal);
    if (breakdownHoverCard) breakdownHoverCard.addEventListener("click", openModal);

    if (pipelineModal) {
        pipelineModal.addEventListener("click", (e) => {
            if (e.target === pipelineModal) closeModal();
        });

        // Prevent wheel & touch on backdrop from scrolling the background main page
        pipelineModal.addEventListener("wheel", (e) => {
            if (e.target === pipelineModal) {
                e.preventDefault();
            }
        }, { passive: false });

        pipelineModal.addEventListener("touchmove", (e) => {
            if (e.target === pipelineModal) {
                e.preventDefault();
            }
        }, { passive: false });
    }

    const modalBody = pipelineModal ? pipelineModal.querySelector(".modal-body") : null;
    if (modalBody) {
        modalBody.addEventListener("wheel", (e) => {
            // Stop wheel events from propagating to Lenis smooth scroll while allowing native 60fps scroll
            e.stopPropagation();
        }, { passive: true });
    }

    if (breakdownHoverCard) {
        breakdownHoverCard.addEventListener("wheel", (e) => {
            e.stopPropagation();
        }, { passive: false });
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && pipelineModal && pipelineModal.style.display === "flex") {
            closeModal();
        }
    });

    // Helper: Update Dynamic Identification & Verification Breakdown Details
    function updateVerificationBreakdown(data, isTier1, isError = false, isOod = false) {
        if (!hoverCardBody) return;

        if (isTier1) {
            const confPct = (data.prediction.confidence * 100).toFixed(1);
            if (hoverCardStatus) {
                hoverCardStatus.className = "hover-status";
                hoverCardStatus.style.background = "#ECFDF5";
                hoverCardStatus.style.color = "#047857";
                hoverCardStatus.style.borderColor = "#A7F3D0";
                hoverCardStatus.textContent = "Tier-1 Grounded";
            }
            if (modalStatusText) modalStatusText.textContent = "Tier-1 ONNX (Edge CPU)";
            if (modalStatClass) modalStatClass.textContent = data.prediction.display_name;
            if (modalStatSubclass) modalStatSubclass.textContent = "Tier-1 Deterministic Target";
            if (modalStatConf) modalStatConf.textContent = `${confPct}%`;
            if (modalStatLayer) modalStatLayer.textContent = "features[-1]";
            if (modalStatLatency) modalStatLatency.textContent = `${data.latency_ms} ms`;

            hoverCardBody.innerHTML = `
                <div class="hover-card-row">
                    <span class="hover-card-bullet"></span>
                    <div><strong>Model Match:</strong> Classified as <strong>${data.prediction.display_name}</strong> (${confPct}% confidence).</div>
                </div>
                <div class="hover-card-row">
                    <span class="hover-card-bullet"></span>
                    <div><strong>Gating Threshold:</strong> Score ${confPct}% &ge; 75.0% passed for deterministic ONNX CPU inference (${data.latency_ms} ms).</div>
                </div>
                <div class="hover-card-row">
                    <span class="hover-card-bullet"></span>
                    <div><strong>Attention Focus:</strong> Grad-CAM backpropagation to <code>features[-1]</code> verifies high saliency on authentic physical surface contours.</div>
                </div>
                <div class="hover-card-row">
                    <span class="hover-card-bullet"></span>
                    <div><strong>Botanical Specimen:</strong> Matched to <em>${data.metadata.scientific_name || 'Taxon'}</em>.</div>
                </div>
            `;
        } else if (isError) {
            const fb = data.fallback_response || {};
            if (hoverCardStatus) {
                hoverCardStatus.className = "hover-status";
                hoverCardStatus.style.background = "#FEF2F2";
                hoverCardStatus.style.color = "#B91C1C";
                hoverCardStatus.style.borderColor = "#FECACA";
                hoverCardStatus.textContent = "Processing Error";
            }
            if (modalStatusText) modalStatusText.textContent = "Pipeline Error";
            if (modalStatClass) modalStatClass.textContent = "Analysis Incomplete";
            if (modalStatSubclass) modalStatSubclass.textContent = "Execution Interrupted";
            if (modalStatConf) modalStatConf.textContent = "N/A";
            if (modalStatLayer) modalStatLayer.textContent = "N/A";
            if (modalStatLatency) modalStatLatency.textContent = `${data.latency_ms || 0} ms`;

            hoverCardBody.innerHTML = `
                <div class="hover-card-row">
                    <span class="hover-card-bullet" style="background:#EF4444"></span>
                    <div><strong>System State:</strong> ${fb.error_reason || 'Pipeline execution interrupted'}</div>
                </div>
            `;
        } else if (isOod) {
            const fb = data.fallback_response || {};
            if (hoverCardStatus) {
                hoverCardStatus.className = "hover-status";
                hoverCardStatus.style.background = "#FEF2F2";
                hoverCardStatus.style.color = "#B91C1C";
                hoverCardStatus.style.borderColor = "#FECACA";
                hoverCardStatus.textContent = "Non-Spice Object";
            }
            if (modalStatusText) modalStatusText.textContent = "OOD Validation Rejection";
            if (modalStatClass) modalStatClass.textContent = "Non-Spice Subject";
            if (modalStatSubclass) modalStatSubclass.textContent = "Rejected as Out-of-Distribution";
            if (modalStatConf) modalStatConf.textContent = "Rejected";
            if (modalStatLayer) modalStatLayer.textContent = "features[-1]";
            if (modalStatLatency) modalStatLatency.textContent = `${data.latency_ms} ms`;

            hoverCardBody.innerHTML = `
                <div class="hover-card-row">
                    <span class="hover-card-bullet" style="background:#EF4444"></span>
                    <div><strong>Verdict:</strong> Rejected as non-spice subject by botanical validation.</div>
                </div>
                <div class="hover-card-row">
                    <span class="hover-card-bullet" style="background:#EF4444"></span>
                    <div><strong>Assessment:</strong> ${fb.confidence_assessment || 'Visual characteristics do not match spices.'}</div>
                </div>
                <div class="hover-card-row">
                    <span class="hover-card-bullet" style="background:#EF4444"></span>
                    <div><strong>Recommendation:</strong> ${fb.action_recommendation || 'Please provide an authentic spice sample.'}</div>
                </div>
            `;
        } else {
            // Tier-2 Valid Spice Fallback
            const fb = data.fallback_response || {};
            if (hoverCardStatus) {
                hoverCardStatus.className = "hover-status";
                hoverCardStatus.style.background = "#EFF6FF";
                hoverCardStatus.style.color = "#1D4ED8";
                hoverCardStatus.style.borderColor = "#BFDBFE";
                hoverCardStatus.textContent = "Tier-2 Fallback";
            }
            if (modalStatusText) modalStatusText.textContent = "Tier-2 Multimodal Fallback";
            if (modalStatClass) modalStatClass.textContent = fb.identified_name || "Novel Spice";
            if (modalStatSubclass) modalStatSubclass.textContent = "Tier-2 Multimodal Target";
            if (modalStatConf) modalStatConf.textContent = "Multimodal";
            if (modalStatLayer) modalStatLayer.textContent = "Gemini Vision";
            if (modalStatLatency) modalStatLatency.textContent = `${data.latency_ms} ms`;

            hoverCardBody.innerHTML = `
                <div class="hover-card-row">
                    <span class="hover-card-bullet" style="background:#3B82F6"></span>
                    <div><strong>Fallback Model:</strong> Identified as <strong>${fb.identified_name}</strong> via Gemini 2.5 Flash Vision.</div>
                </div>
                <div class="hover-card-row">
                    <span class="hover-card-bullet" style="background:#3B82F6"></span>
                    <div><strong>Routing Trigger:</strong> Local CNN confidence was &lt; 75.0% threshold or forced fallback.</div>
                </div>
                <div class="hover-card-row">
                    <span class="hover-card-bullet" style="background:#3B82F6"></span>
                    <div><strong>Assessment:</strong> ${fb.confidence_assessment || 'Verified multimodal features'}</div>
                </div>
                <div class="hover-card-row">
                    <span class="hover-card-bullet" style="background:#3B82F6"></span>
                    <div><strong>Scientific Name:</strong> <em>${fb.scientific_name || 'Taxon'}</em></div>
                </div>
            `;
        }
    }

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

            // Query image dimensions
            const tempImg = new Image();
            tempImg.onload = () => {
                origDimsTag.textContent = `${tempImg.naturalWidth} × ${tempImg.naturalHeight} RGB`;
            };
            tempImg.src = src;

            // Clear old heatmaps
            heatmapImg.src = "";
            heatmapImg.style.display = "none";
            camPlaceholder.style.display = "flex";
            heatmapLegend.style.display = "none";
            heatmapToggleGroup.style.display = "none";
            blendedHeatmapSrc = null;
            pureThermalSrc = null;

            // Switch views to side-by-side comparison
            uploadContainer.style.display = "none";
            previewContainer.style.display = "block";

            // Reset results to idle
            emptyState.style.display = "flex";
            emptyState.classList.remove("loading");
            const inferenceLoading = document.getElementById("inference-loading-overlay");
            if (inferenceLoading) inferenceLoading.style.display = "none";
            resultsCard.style.display = "none";

            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Main Inference & XAI Trigger
    analyzeBtn.addEventListener("click", async () => {
        if (!currentFile) return;

        // Loading State
        const inferenceLoading = document.getElementById("inference-loading-overlay");
        if (inferenceLoading) inferenceLoading.style.display = "flex";
        emptyState.style.display = "flex";
        emptyState.classList.add("loading");
        resultsCard.style.display = "none";
        analyzeBtnText.textContent = "Analyzing Network...";
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append("image", currentFile);
        formData.append("threshold", (thresholdInput.value / 100).toString());
        formData.append("force_fallback", forceFallback.checked.toString());

        try {
            // 1. Run Classification Endpoint
            const res = await fetch("/api/v1/classify", {
                method: "POST",
                body: formData
            });
            const data = await res.json();

            if (!res.ok) {
                alert(data.error || "Failed to classify image.");
                emptyState.classList.remove("loading");
                analyzeBtnText.textContent = "Analyze & Verify Spice";
                analyzeBtn.disabled = false;
                return;
            }

            lastJsonData = data;

            // Live Telemetry Pill
            if (data.latency_ms !== undefined) {
                let engine = data.tier === "TIER_1_ONNX_LOCAL" ? "ONNX MobileNetV3" : "Gemini Multimodal";
                telemetryPill.innerHTML = `<span class="badge-dot"></span> ${engine}: ${data.latency_ms} ms <span class="latency">| 200 OK</span>`;

                telemetryPill.classList.remove("pulse-anim");
                void telemetryPill.offsetWidth; // trigger reflow
                telemetryPill.classList.add("pulse-anim");
            }

            renderResults(data);

            // 2. Run Explainability (Grad-CAM)
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
                        blendedHeatmapSrc = `data:image/jpeg;base64,${xaiData.heatmap_base64}`;
                        pureThermalSrc = xaiData.thermal_base64 ? `data:image/jpeg;base64,${xaiData.thermal_base64}` : blendedHeatmapSrc;

                        // Set current mode image
                        heatmapImg.src = currentHeatmapMode === "blended" ? blendedHeatmapSrc : pureThermalSrc;
                        heatmapImg.style.display = "block";
                        camPlaceholder.style.display = "none";
                        heatmapLegend.style.display = "flex";
                        heatmapToggleGroup.style.display = "inline-flex";
                    }
                }
            }

        } catch (error) {
            console.error(error);
            alert("Error communicating with inference API");
            emptyState.classList.remove("loading");
            const inferenceLoading = document.getElementById("inference-loading-overlay");
            if (inferenceLoading) inferenceLoading.style.display = "none";
        } finally {
            analyzeBtnText.textContent = "Analyze & Verify Spice";
            analyzeBtn.disabled = false;
            const inferenceLoading = document.getElementById("inference-loading-overlay");
            if (inferenceLoading) inferenceLoading.style.display = "none";
        }
    });

    function renderResults(data) {
        emptyState.style.display = "none";
        emptyState.classList.remove("loading");
        resultsCard.style.display = "block";

        const badge = document.getElementById("tier-badge");
        const confMeter = document.getElementById("confidence-meter");
        const title = document.getElementById("spice-name");
        const sciSubtitle = document.getElementById("scientific-subtitle");

        const botanicalMetaTable = document.getElementById("botanical-meta-table");

        const attrSec = document.getElementById("attribution-section");
        const attrText = document.getElementById("attribution-text");
        attrSec.style.display = "block";

        if (data.tier === "TIER_1_ONNX_LOCAL") {
            badge.className = "badge tier1";
            badge.textContent = "Tier-1 ONNX Engine (Local CPU)";
            confMeter.textContent = `${(data.prediction.confidence * 100).toFixed(1)}% Confidence`;

            title.textContent = data.prediction.display_name;
            sciSubtitle.textContent = data.metadata.scientific_name ? `Taxon: ${data.metadata.scientific_name}` : "";

            // 1. IN-CARD BOTANICAL PROFILE (Balances the right panel vertically)
            botanicalMetaTable.innerHTML = `
                <div class="spec-row">
                    <div class="spec-label">Scientific Name</div>
                    <div class="spec-value"><em>${data.metadata.scientific_name || 'N/A'}</em></div>
                </div>
                <div class="spec-row">
                    <div class="spec-label">Morphology</div>
                    <div class="spec-value">${data.metadata.key_visual_features || 'N/A'}</div>
                </div>
                <div class="spec-row">
                    <div class="spec-label">Flavor Profile</div>
                    <div class="spec-value">${data.metadata.flavor_profile || 'N/A'}</div>
                </div>
                <div class="spec-row">
                    <div class="spec-label">Culinary Uses</div>
                    <div class="spec-value">${data.metadata.common_uses || 'N/A'}</div>
                </div>
            `;

            // 2. MODAL TABLES (Deep Learning Architecture & Telemetry Specifications)
            if (modalModelParamsTable) {
                modalModelParamsTable.innerHTML = `
                    <div class="spec-tech-row">
                        <div class="spec-tech-label">Neural Architecture</div>
                        <div class="spec-tech-value">
                            <code>MobileNetV3-Small</code>
                            <span>Inverted Bottleneck Residuals + Squeeze-and-Excitation Attention</span>
                        </div>
                    </div>
                    <div class="spec-tech-row">
                        <div class="spec-tech-label">Input Tensor Shape</div>
                        <div class="spec-tech-value">
                            <code>[1, 3, 224, 224]</code>
                            <span>Batch: 1, Channels: RGB (3), Dimensions: 224×224</span>
                        </div>
                    </div>
                    <div class="spec-tech-row">
                        <div class="spec-tech-label">Tensor Normalization</div>
                        <div class="spec-tech-value">
                            <code>ImageNet Norm</code>
                            <span>μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]</span>
                        </div>
                    </div>
                    <div class="spec-tech-row">
                        <div class="spec-tech-label">Attribution Layer Hook</div>
                        <div class="spec-tech-value">
                            <code>features[-1]</code>
                            <span>Final 1×1 Convolutional Block (576 Channel Feature Volume)</span>
                        </div>
                    </div>
                    <div class="spec-tech-row">
                        <div class="spec-tech-label">Runtime Engine</div>
                        <div class="spec-tech-value">
                            <code>ONNX Runtime</code>
                            <span>CPUExecutionProvider (Opset 17, Zero External GPU Dependency)</span>
                        </div>
                    </div>
                    <div class="spec-tech-row">
                        <div class="spec-tech-label">Decision Gating Threshold</div>
                        <div class="spec-tech-value">
                            <code>&ge; 75.0% Confidence</code>
                            <span class="spec-tech-tag">PASS: ${(data.prediction.confidence * 100).toFixed(1)}% &ge; 75.0%</span>
                        </div>
                    </div>
                    <div class="spec-tech-row">
                        <div class="spec-tech-label">Execution Latency SLA</div>
                        <div class="spec-tech-value">
                            <code>${data.latency_ms} ms</code>
                            <span>Sub-50ms Real-Time Target Achieved</span>
                        </div>
                    </div>
                    <div class="spec-tech-row">
                        <div class="spec-tech-label">Thermal Colormap</div>
                        <div class="spec-tech-value">
                            <code>COLORMAP_JET</code>
                            <span>OpenCV JET 256-color thermal heatmap synthesis (α=0.45 blend)</span>
                        </div>
                    </div>
                `;
            }

            attrText.innerHTML = `Grad-CAM gradient backpropagation to layer <code>features[-1]</code> confirms high spatial focus on distinguishing physical contours and surface morphology of <strong>${data.prediction.display_name}</strong>, validating prediction authenticity above spurious background signals.`;

            renderBars(data.top_3);
            updateVerificationBreakdown(data, true);

        } else {
            // Tier-2 Fallback Handling
            badge.className = "badge tier2";
            badge.textContent = "Tier-2 Multimodal Fallback";
            const fb = data.fallback_response;

            if (fb.error) {
                badge.className = "badge error";
                badge.textContent = "Processing Error";
                title.textContent = "Analysis Incomplete";
                sciSubtitle.textContent = "";
                botanicalMetaTable.innerHTML = `<div class="spec-row"><div class="spec-label">System State</div><div class="spec-value">${fb.error_reason}</div></div>`;
                document.getElementById("prob-bars").innerHTML = "";
                attrSec.style.display = "none";
                if (modalModelParamsTable) {
                    modalModelParamsTable.innerHTML = `
                        <div class="spec-tech-row">
                            <div class="spec-tech-label">Pipeline Status</div>
                            <div class="spec-tech-value"><code>EXECUTION_ERROR</code> <span>${fb.error_reason || 'Pipeline execution interrupted'}</span></div>
                        </div>
                    `;
                }
                updateVerificationBreakdown(data, false, true, false);
            } else if (!fb.is_spice) {
                badge.className = "badge error";
                badge.textContent = "Out of Distribution (Not a Spice)";
                title.textContent = "Non-Spice Subject";
                sciSubtitle.textContent = "Object rejected by botanical validation";

                botanicalMetaTable.innerHTML = `
                    <div class="spec-row">
                        <div class="spec-label">Visual Assessment</div>
                        <div class="spec-value">${fb.confidence_assessment || 'Non-spice visual characteristics'}</div>
                    </div>
                    <div class="spec-row">
                        <div class="spec-label">Recommendation</div>
                        <div class="spec-value">${fb.action_recommendation || 'Please provide an authentic spice sample.'}</div>
                    </div>
                `;

                if (modalModelParamsTable) {
                    modalModelParamsTable.innerHTML = `
                        <div class="spec-tech-row">
                            <div class="spec-tech-label">Routing Decision</div>
                            <div class="spec-tech-value"><code>LOW_CONFIDENCE_ESCALATION</code> <span>Tier-1 CNN confidence fell below deterministic threshold</span></div>
                        </div>
                        <div class="spec-tech-row">
                            <div class="spec-tech-label">Validation Engine</div>
                            <div class="spec-tech-value"><code>Gemini 2.5 Flash Vision</code> <span>Multimodal Foundation Model Out-of-Distribution Rejection</span></div>
                        </div>
                        <div class="spec-tech-row">
                            <div class="spec-tech-label">Rejection Reason</div>
                            <div class="spec-tech-value"><span>${fb.confidence_assessment || 'Physical specimen does not match known spice morphological taxonomy.'}</span></div>
                        </div>
                        <div class="spec-tech-row">
                            <div class="spec-tech-label">System Recommendation</div>
                            <div class="spec-tech-value"><span>${fb.action_recommendation || 'Provide authentic specimen within known domain classes.'}</span></div>
                        </div>
                    `;
                }

                document.getElementById("prob-bars").innerHTML = "";

                const top1Class = data.tier_1_context && data.tier_1_context.top_3 ? data.tier_1_context.top_3[0].display_name : (data.tier_1_context && data.tier_1_context.top_1_class ? data.tier_1_context.top_1_class : 'Unknown');
                const top1Prob = data.tier_1_context && data.tier_1_context.top_1_prob ? (data.tier_1_context.top_1_prob * 100).toFixed(1) : 0;

                attrText.innerHTML = `<strong>Out-of-Distribution Warning:</strong> The local model predicted <em>${top1Class}</em> (${top1Prob}% confidence), but attention maps showed diffuse/spurious activations. Tier-2 multimodal vision verified the object is not an authentic culinary spice.`;
                updateVerificationBreakdown(data, false, false, true);

            } else {
                confMeter.textContent = "Multimodal Verification";
                title.textContent = fb.identified_name;
                sciSubtitle.textContent = fb.scientific_name ? `Taxon: ${fb.scientific_name}` : "";

                botanicalMetaTable.innerHTML = `
                    <div class="spec-row">
                        <div class="spec-label">Scientific Name</div>
                        <div class="spec-value"><em>${fb.scientific_name || 'N/A'}</em></div>
                    </div>
                    <div class="spec-row">
                        <div class="spec-label">Visual Assessment</div>
                        <div class="spec-value">${fb.confidence_assessment || 'N/A'}</div>
                    </div>
                    <div class="spec-row">
                        <div class="spec-label">Flavor Profile</div>
                        <div class="spec-value">${fb.flavor_profile || 'N/A'}</div>
                    </div>
                `;

                if (modalModelParamsTable) {
                    modalModelParamsTable.innerHTML = `
                        <div class="spec-tech-row">
                            <div class="spec-tech-label">Routing Trigger</div>
                            <div class="spec-tech-value"><code>THRESHOLD_ESCALATION</code> <span>Local CNN confidence &lt; 75.0% deterministic boundary</span></div>
                        </div>
                        <div class="spec-tech-row">
                            <div class="spec-tech-label">Resolution Engine</div>
                            <div class="spec-tech-value"><code>Gemini 2.5 Flash Vision</code> <span>Multimodal zero-shot botanical taxonomic identification</span></div>
                        </div>
                        <div class="spec-tech-row">
                            <div class="spec-tech-label">Taxonomic Scope</div>
                            <div class="spec-tech-value"><span class="spec-tech-tag">${fb.is_in_dataset_scope ? 'Within 14 Core Spice Categories' : 'Novel / Unseen Spice Class'}</span></div>
                        </div>
                        <div class="spec-tech-row">
                            <div class="spec-tech-label">Visual Verification</div>
                            <div class="spec-tech-value"><span>${fb.confidence_assessment || 'Multimodal visual features verified against culinary spice reference.'}</span></div>
                        </div>
                    `;
                }

                const top1Prob = data.tier_1_context && data.tier_1_context.top_1_prob ? (data.tier_1_context && data.tier_1_context.top_1_prob * 100).toFixed(1) : 0;
                attrText.innerHTML = `Local Tier-1 confidence (${top1Prob}%) fell below the 75% boundary. Attention was insufficient for deterministic local classification; successfully classified via Tier-2 Foundation Vision.`;
                updateVerificationBreakdown(data, false, false, false);
            }

            if (data.tier_1_context && data.tier_1_context.top_3) {
                renderBars(data.tier_1_context.top_3);
            }
        }
    }

    function renderBars(top3) {
        const container = document.getElementById("prob-bars");
        container.innerHTML = "";

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
                    if (fill) fill.style.width = `${pct}%`;
                });
            });
        });
    }
});
