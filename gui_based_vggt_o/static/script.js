// script.js

document.addEventListener('DOMContentLoaded', () => {
  // --- Startup health check (CUDA / checkpoint) ---
  (async () => {
    try {
      const res = await fetch('/api/health');
      const h = await res.json();
      const msgs = [];
      if (!h.cuda) {
        let m = 'No CUDA GPU detected — the VGGT inference step (2) will be very slow or fail.';
        if (h.torch_error) m += ' (torch: ' + h.torch_error + ')';
        msgs.push(m);
      }
      if (!h.checkpoint_present) {
        msgs.push('VGGT checkpoint not found at ' + (h.checkpoint_path || 'checkpoints/') + ' — step 2 will fail.');
      }
      if (msgs.length) {
        const banner = document.createElement('div');
        banner.style.cssText = 'background:#7f1d1d;color:#fff;padding:0.6rem 1rem;font-size:0.85rem;text-align:center;';
        banner.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> ' + msgs.join('<br>') +
          '<br><span style="opacity:0.8;font-size:0.78rem;">Server interpreter: ' + (h.python || '?') +
          '. Launch the server from the <code>vggt_o</code> conda env so it has CUDA torch + the checkpoint.</span>';
        document.body.prepend(banner);
      }
    } catch (e) { /* health endpoint optional */ }
  })();

  // --- Stepper Logic ---
  const steps = document.querySelectorAll('.step');
  const views = document.querySelectorAll('.view-section');

  function goToStep(stepIndex) {
    steps.forEach((step, idx) => {
      step.classList.remove('active');
      if (idx < stepIndex) step.classList.add('completed');
      else step.classList.remove('completed');
    });
    steps[stepIndex].classList.add('active');

    views.forEach((view) => view.classList.remove('active'));
    document.getElementById(`view-${stepIndex + 1}`).classList.add('active');
    
    // Maximize space for Step 3 (Annotation Canvas)
    const mainContent = document.querySelector('.main-content');
    const sidebar = document.querySelector('.sidebar');
    if (stepIndex === 2) {
      mainContent.style.padding = '0';
      mainContent.style.overflow = 'hidden';
      sidebar.style.width = '220px';
      sidebar.style.padding = '1.5rem 1rem';
      sidebar.querySelector('h1').style.fontSize = '1.2rem';
    } else {
      mainContent.style.padding = '2rem';
      mainContent.style.overflow = 'auto';
      sidebar.style.width = '300px';
      sidebar.style.padding = '2rem';
      sidebar.querySelector('h1').style.fontSize = '1.5rem';
    }
    
    // Trigger resize to fix canvas sizing
    setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
  }

  steps.forEach((step, idx) => {
    step.addEventListener('click', () => {
      if (step.classList.contains('completed') || step.classList.contains('active')) {
        goToStep(idx);
      }
    });
  });

  // --- Step 1: File Upload ---
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileList = document.getElementById('file-list');
  const btnNext1 = document.getElementById('btn-next-1');
  let selectedFiles = [];

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

  function handleFiles(files) {
    selectedFiles = Array.from(files);
    fileList.innerHTML = '';
    selectedFiles.forEach(file => {
      const item = document.createElement('div');
      item.className = 'file-item';
      item.innerHTML = `<i class="fa-solid fa-video"></i> ${file.name} <span style="margin-left:auto; color:var(--text-secondary)">${(file.size / (1024*1024)).toFixed(2)} MB</span>`;
      fileList.appendChild(item);
    });
    btnNext1.disabled = selectedFiles.length === 0;
  }

  btnNext1.addEventListener('click', async () => {
    btnNext1.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Uploading...';
    btnNext1.disabled = true;

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));

    try {
      const res = await fetch('/api/upload_videos', { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        const info = document.getElementById('upload-info');
        if (info) {
          let html = '';
          if (data.cameras && data.cameras.length) {
            html += `<div style="color: var(--success); font-size: 0.9rem;"><i class="fa-solid fa-circle-check"></i> Detected ${data.cameras.length} camera(s): <strong>${data.cameras.join(', ')}</strong>` +
                    (data.frame_count ? ` &middot; ${data.frame_count} synchronized frames` : '') + `</div>`;
          }
          if (data.warning) {
            html += `<div style="color: var(--warning); font-size: 0.85rem; margin-top: 0.5rem;"><i class="fa-solid fa-triangle-exclamation"></i> ${data.warning}</div>`;
          }
          info.innerHTML = html;
        }
        goToStep(1);
      } else {
        alert('Upload failed: ' + (data.message || res.statusText));
      }
    } catch (err) {
      console.error(err);
      alert('Upload error');
    } finally {
      btnNext1.innerHTML = 'Next Step <i class="fa-solid fa-arrow-right"></i>';
      btnNext1.disabled = false;
    }
  });

  // --- Step 2: Run Pipeline ---
  const btnRunPipeline = document.getElementById('btn-run-pipeline');
  const terminal = document.getElementById('terminal-output');
  const btnNext2 = document.getElementById('btn-next-2');

  function appendTerminal(text) {
    terminal.innerHTML += `<div>> ${text}</div>`;
    terminal.scrollTop = terminal.scrollHeight;
  }

  let logLineCount = 0;
  let pollTimer = null;

  function renderStats(stats) {
    if (!stats || !stats.num_frames) return;
    document.getElementById('stat-processed').innerText = stats.num_frames;
    document.getElementById('stat-included').innerText = stats.num_inlier_frames;

    let maxError = 0;
    if (stats.cameras) {
      for (const cam of Object.values(stats.cameras)) {
        if (cam.rotation_sigma_deg > maxError) maxError = cam.rotation_sigma_deg;
      }
    }
    const isConsistent = maxError <= 5.0;
    document.getElementById('stat-error').innerText = maxError.toFixed(2) + "°";
    const consistencyEl = document.getElementById('stat-consistency');
    if (isConsistent) {
      consistencyEl.innerText = "PASS";
      consistencyEl.style.color = "var(--success)";
    } else {
      consistencyEl.innerText = "WARN (High Variance)";
      consistencyEl.style.color = "var(--warning)";
    }
    document.getElementById('vggt-stats').style.display = 'block';
  }

  async function pollStatus() {
    try {
      const res = await fetch('/api/pipeline_status');
      const data = await res.json();

      // Append only new log lines
      if (data.log && data.log.length > logLineCount) {
        for (let i = logLineCount; i < data.log.length; i++) {
          appendTerminal(data.log[i]);
        }
        logLineCount = data.log.length;
      }

      if (data.state === 'done') {
        clearInterval(pollTimer);
        pollTimer = null;
        renderStats(data.stats);
        btnRunPipeline.disabled = false;
        btnRunPipeline.innerHTML = '<i class="fa-solid fa-play"></i> Run Pipeline';
        btnNext2.disabled = false;
        btnNext2.classList.add('btn-primary');
      } else if (data.state === 'error') {
        clearInterval(pollTimer);
        pollTimer = null;
        appendTerminal("Pipeline failed: " + (data.message || 'unknown error'));
        btnRunPipeline.disabled = false;
        btnRunPipeline.innerHTML = '<i class="fa-solid fa-play"></i> Retry Pipeline';
      }
    } catch (err) {
      // transient network error during long run; keep polling
      console.warn('poll error', err);
    }
  }

  btnRunPipeline.addEventListener('click', async () => {
    btnRunPipeline.disabled = true;
    btnRunPipeline.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running...';
    btnNext2.disabled = true;
    terminal.innerHTML = '';
    logLineCount = 0;
    appendTerminal("Starting pipeline execution (the VGGT step runs on the GPU and may take several minutes)...");
    document.getElementById('vggt-stats').style.display = 'none';

    try {
      const maxFrames = parseInt(document.getElementById('max-frames-input').value) || 100;
      const keepBest = parseFloat(document.getElementById('keep-best-input').value) || 100.0;

      const res = await fetch(`/api/run_pipeline?max_frames=${maxFrames}&keep_best_percent=${keepBest}`, { method: 'POST' });
      const data = await res.json();

      if (data.status === "started") {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(pollStatus, 1500);
        pollStatus();
      } else {
        appendTerminal("Could not start pipeline: " + (data.message || 'unknown error'));
        btnRunPipeline.disabled = false;
        btnRunPipeline.innerHTML = '<i class="fa-solid fa-play"></i> Run Pipeline';
      }
    } catch (err) {
      appendTerminal(`Error: ${err.message}`);
      btnRunPipeline.disabled = false;
      btnRunPipeline.innerHTML = '<i class="fa-solid fa-play"></i> Run Pipeline';
    }
  });

  btnNext2.addEventListener('click', () => {
    goToStep(2);
    initAnnotationCanvas();
  });


  // --- Step 3: Annotation ---
  // UI Elements
  const frameSlider = document.getElementById('frame-slider');
  const frameLabel = document.getElementById('frame-label');
  const btnPrevFrame = document.getElementById('btn-prev-frame');
  const btnNextFrame = document.getElementById('btn-next-frame');
  const brightnessSlider = document.getElementById('brightness-slider');
  const btnZoomIn = document.getElementById('btn-zoom-in');
  const btnZoomOut = document.getElementById('btn-zoom-out');
  const btnZoomReset = document.getElementById('btn-zoom-reset');
  const zoomLabel = document.getElementById('zoom-label');
  
  const toolAnnotate = document.getElementById('tool-annotate');
  const toolPan = document.getElementById('tool-pan');
  const pointSizeSlider = document.getElementById('point-size-slider');
  
  const mainWorkspace = document.getElementById('main-workspace');
  const canvasWrapper = document.getElementById('canvas-wrapper');
  const mainImage = document.getElementById('main-image');
  const mainCanvas = document.getElementById('main-canvas');
  const thumbnailsStrip = document.getElementById('thumbnails-strip');
  
  const btnAddPoint = document.getElementById('btn-add-point');
  const pointLabelInput = document.getElementById('point-label');
  const pointsListDiv = document.getElementById('points-list');
  const distancesListDiv = document.getElementById('distances-list');
  const btnNext3 = document.getElementById('btn-next-3');

  // State
  let allFrames = [];
  let currentFrameIdx = 0;
  let activeCamera = null;
  let camerasInFrame = [];
  
  let pointsData = {};
  let distancesData = [];
  let activePointId = null;
  let pointCounter = 1;
  let distanceCounter = 1;
  
  // Transform State
  let zoom = 1;
  let panX = 0;
  let panY = 0;
  let isPanning = false;
  let startX = 0, startY = 0;
  
  // Tool State
  let currentTool = 'annotate'; // 'annotate' or 'pan'

  async function initAnnotationCanvas() {
    const res = await fetch('/api/frames_data');
    const data = await res.json();
    if (!data.frames || data.frames.length === 0) return;
    
    allFrames = data.frames;
    
    frameSlider.max = allFrames.length - 1;
    setFrame(0);
  }

  function setFrame(idx) {
    if (idx < 0 || idx >= allFrames.length) return;
    currentFrameIdx = idx;
    frameSlider.value = idx;
    
    const frame = allFrames[currentFrameIdx];
    frameLabel.innerText = `Frame ${frame.index}`;
    camerasInFrame = Object.keys(frame.images).sort();
    
    if (!camerasInFrame.includes(activeCamera)) {
      activeCamera = camerasInFrame[0];
    }
    
    renderThumbnails();
    loadActiveView();
  }

  frameSlider.addEventListener('input', (e) => setFrame(parseInt(e.target.value)));
  btnPrevFrame.addEventListener('click', () => setFrame(currentFrameIdx - 1));
  btnNextFrame.addEventListener('click', () => setFrame(currentFrameIdx + 1));

  function renderThumbnails() {
    thumbnailsStrip.innerHTML = '';
    const frame = allFrames[currentFrameIdx];
    
    camerasInFrame.forEach(cam => {
      const thumb = document.createElement('div');
      thumb.style.height = '100%';
      thumb.style.aspectRatio = '16/9';
      thumb.style.cursor = 'pointer';
      thumb.style.border = cam === activeCamera ? '2px solid var(--accent-color)' : '2px solid transparent';
      thumb.style.borderRadius = '4px';
      thumb.style.overflow = 'hidden';
      thumb.style.position = 'relative';
      
      const thumbInner = document.createElement('div');
      thumbInner.style.width = '100%';
      thumbInner.style.height = '100%';
      thumbInner.style.transformOrigin = '0 0';
      thumbInner.style.position = 'relative';
      thumbInner.style.pointerEvents = 'none'; // let mouse events pass to thumb
      
      const img = new Image();
      img.src = frame.images[cam];
      img.style.width = '100%';
      img.style.height = '100%';
      img.style.objectFit = 'contain';
      img.style.background = '#000';
      thumbInner.appendChild(img);
      
      const thumbCanvas = document.createElement('canvas');
      thumbCanvas.style.position = 'absolute';
      thumbCanvas.style.top = '0';
      thumbCanvas.style.left = '0';
      thumbCanvas.style.width = '100%';
      thumbCanvas.style.height = '100%';
      thumbCanvas.style.pointerEvents = 'none';
      thumbCanvas.dataset.thumbCam = cam;
      
      thumb.appendChild(thumbInner);
      thumb.appendChild(thumbCanvas);
      
      const lbl = document.createElement('div');
      lbl.innerText = `Cam ${cam}`;
      lbl.style.position = 'absolute';
      lbl.style.bottom = '0';
      lbl.style.left = '0';
      lbl.style.background = 'rgba(0,0,0,0.7)';
      lbl.style.padding = '4px 8px';
      lbl.style.fontSize = '0.9rem';
      lbl.style.fontWeight = 'bold';
      thumb.appendChild(lbl);
      
      thumb.onclick = () => {
        activeCamera = cam;
        renderThumbnails();
        loadActiveView();
      };
      
      let tZoom = 1;
      let tPanX = 0;
      let tPanY = 0;
      thumb.dataset.tZoom = 1;
      
      thumb.addEventListener('wheel', (e) => {
        e.preventDefault();
        const zoomFactor = e.deltaY < 0 ? 1.2 : 0.8;
        const rect = thumb.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        tPanX = mouseX - (mouseX - tPanX) * zoomFactor;
        tPanY = mouseY - (mouseY - tPanY) * zoomFactor;
        tZoom *= zoomFactor;
        
        if (tZoom <= 1) {
          tZoom = 1; tPanX = 0; tPanY = 0;
        }
        thumb.dataset.tZoom = tZoom;
        thumb.dataset.tPanX = tPanX;
        thumb.dataset.tPanY = tPanY;
        thumbInner.style.transform = `translate(${tPanX}px, ${tPanY}px) scale(${tZoom})`;
        drawThumbnails();
      });
      
      thumbnailsStrip.appendChild(thumb);
      
      img.onload = () => {
        thumbCanvas.width = thumbCanvas.clientWidth;
        thumbCanvas.height = thumbCanvas.clientHeight;
        drawThumbnails();
      };
    });
  }

  function drawThumbnails() {
    camerasInFrame.forEach(cam => {
      const thumbCanvas = document.querySelector(`canvas[data-thumb-cam="${cam}"]`);
      if (!thumbCanvas) return;
      const ctx = thumbCanvas.getContext('2d');
      ctx.clearRect(0, 0, thumbCanvas.width, thumbCanvas.height);
      
      // We assume naturalWidth exists because img.onload triggered this.
      const frame = allFrames[currentFrameIdx];
      // Need natural bounds of the image
      const thumbImg = thumbCanvas.parentElement.querySelector('img');
      if(!thumbImg || !thumbImg.naturalWidth) return;
      
      const imgRatio = thumbImg.naturalWidth / thumbImg.naturalHeight;
      const canvasRatio = thumbCanvas.width / thumbCanvas.height;
      let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
      
      if (imgRatio > canvasRatio) {
        drawWidth = thumbCanvas.width;
        drawHeight = thumbCanvas.width / imgRatio;
        offsetY = (thumbCanvas.height - drawHeight) / 2;
      } else {
        drawHeight = thumbCanvas.height;
        drawWidth = thumbCanvas.height * imgRatio;
        offsetX = (thumbCanvas.width - drawWidth) / 2;
      }
      
      const thumb = thumbCanvas.parentElement;
      const tZoom = parseFloat(thumb.dataset.tZoom || 1);
      const tPanX = parseFloat(thumb.dataset.tPanX || 0);
      const tPanY = parseFloat(thumb.dataset.tPanY || 0);

      for (const [id, pt] of Object.entries(pointsData)) {
        if (pt.observations && pt.observations[cam]) {
          const obs = pt.observations[cam];
          const cx = offsetX + (obs.source_x / thumbImg.naturalWidth) * drawWidth;
          const cy = offsetY + (obs.source_y / thumbImg.naturalHeight) * drawHeight;
          
          const screenX = (cx * tZoom) + tPanX;
          const screenY = (cy * tZoom) + tPanY;

          // Draw full circle
          ctx.beginPath();
          ctx.arc(screenX, screenY, 5, 0, 2 * Math.PI);
          ctx.fillStyle = id === activePointId ? '#3b82f6' : '#10b981';
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = '#fff';
          ctx.stroke();
          
          // Draw large label
          ctx.fillStyle = id === activePointId ? '#3b82f6' : '#10b981';
          ctx.font = `bold 14px Inter`;
          ctx.fillText(pt.label, screenX + 7, screenY - 7);
        }
      }
    });
  }

  function loadActiveView() {
    const frame = allFrames[currentFrameIdx];
    mainImage.src = frame.images[activeCamera];
    mainImage.onload = () => {
      mainCanvas.width = mainCanvas.clientWidth;
      mainCanvas.height = mainCanvas.clientHeight;
      drawCanvas();
    };
  }

  window.addEventListener('resize', () => {
    if (mainImage.src) {
      mainCanvas.width = mainCanvas.clientWidth;
      mainCanvas.height = mainCanvas.clientHeight;
      drawCanvas();
    }
  });

  // Toolbar & Transforms
  function updateTransform() {
    canvasWrapper.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    zoomLabel.innerText = `${Math.round(zoom * 100)}%`;
  }

  // Tool Mode Toggles
  toolAnnotate.addEventListener('click', () => {
    currentTool = 'annotate';
    toolAnnotate.style.background = 'var(--accent-color)';
    toolAnnotate.style.color = 'white';
    toolPan.style.background = 'transparent';
    toolPan.style.color = 'var(--text-secondary)';
    mainCanvas.style.cursor = 'crosshair';
  });

  toolPan.addEventListener('click', () => {
    currentTool = 'pan';
    toolPan.style.background = 'var(--accent-color)';
    toolPan.style.color = 'white';
    toolAnnotate.style.background = 'transparent';
    toolAnnotate.style.color = 'var(--text-secondary)';
    mainCanvas.style.cursor = 'grab';
  });

  brightnessSlider.addEventListener('input', () => { drawCanvas(); });
  pointSizeSlider.addEventListener('input', () => { drawCanvas(); });

  btnZoomIn.addEventListener('click', () => { zoom *= 1.2; updateTransform(); });
  btnZoomOut.addEventListener('click', () => { zoom /= 1.2; updateTransform(); });
  btnZoomReset.addEventListener('click', () => { zoom = 1; panX = 0; panY = 0; updateTransform(); });

  mainWorkspace.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    
    // Zoom around mouse point
    const rect = mainWorkspace.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    panX = mouseX - (mouseX - panX) * zoomFactor;
    panY = mouseY - (mouseY - panY) * zoomFactor;
    zoom *= zoomFactor;
    updateTransform();
  });

  mainWorkspace.addEventListener('mousedown', (e) => {
    if (e.button === 1 || e.button === 2 || e.shiftKey || currentTool === 'pan') {
      // Middle click, right click, shift+click or Pan Tool selected
      isPanning = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
      mainWorkspace.style.cursor = 'grabbing';
      if(currentTool === 'pan') mainCanvas.style.cursor = 'grabbing';
      e.preventDefault();
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (isPanning) {
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      updateTransform();
    }
  });

  window.addEventListener('mouseup', () => {
    isPanning = false;
    mainWorkspace.style.cursor = 'grab';
    if(currentTool === 'pan') mainCanvas.style.cursor = 'grab';
  });

  // Avoid context menu on right click to allow panning
  mainWorkspace.addEventListener('contextmenu', e => e.preventDefault());

  // Click to place points
  mainCanvas.addEventListener('mousedown', (e) => {
    if (isPanning || e.button !== 0 || e.shiftKey || currentTool === 'pan') return; // Only left click in annotate mode places points
    
    if (!activePointId) {
      alert('Please select or create a point first by typing a label and clicking +');
      return;
    }
    if (!mainImage.naturalWidth) return;

    // We calculate exactly where the click landed in the logical [0, mainCanvas.width] coordinate space.
    // By using getBoundingClientRect() and its width/height, we automatically account for any CSS transform applied to the wrapper!
    const rect = mainCanvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) / (rect.width / mainCanvas.width);
    const clickY = (e.clientY - rect.top) / (rect.height / mainCanvas.height);
    
    const imgRatio = mainImage.naturalWidth / mainImage.naturalHeight;
    const canvasRatio = mainCanvas.width / mainCanvas.height;
    let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
    
    if (imgRatio > canvasRatio) {
      drawWidth = mainCanvas.width;
      drawHeight = mainCanvas.width / imgRatio;
      offsetY = (mainCanvas.height - drawHeight) / 2;
    } else {
      drawHeight = mainCanvas.height;
      drawWidth = mainCanvas.height * imgRatio;
      offsetX = (mainCanvas.width - drawWidth) / 2;
    }
    
    if (clickX >= offsetX && clickX <= offsetX + drawWidth &&
        clickY >= offsetY && clickY <= offsetY + drawHeight) {
        
      const imgX = ((clickX - offsetX) / drawWidth) * mainImage.naturalWidth;
      const imgY = ((clickY - offsetY) / drawHeight) * mainImage.naturalHeight;
      
      if(!pointsData[activePointId].observations) pointsData[activePointId].observations = {};
      pointsData[activePointId].observations[activeCamera] = {
        x: imgX, y: imgY, source_x: imgX, source_y: imgY
      };
      
      drawCanvas();
      renderSidebar();
    }
  });

  function drawCanvas() {
    const ctx = mainCanvas.getContext('2d');
    ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);

    if (!mainImage.naturalWidth) return;

    const imgRatio = mainImage.naturalWidth / mainImage.naturalHeight;
    const canvasRatio = mainCanvas.width / mainCanvas.height;
    let drawWidth, drawHeight, offsetX = 0, offsetY = 0;

    if (imgRatio > canvasRatio) {
      drawWidth = mainCanvas.width;
      drawHeight = mainCanvas.width / imgRatio;
      offsetY = (mainCanvas.height - drawHeight) / 2;
    } else {
      drawHeight = mainCanvas.height;
      drawWidth = mainCanvas.height * imgRatio;
      offsetX = (mainCanvas.width - drawWidth) / 2;
    }

    // Draw the image exactly onto the canvas space
    ctx.filter = `brightness(${brightnessSlider.value})`;
    ctx.drawImage(mainImage, offsetX, offsetY, drawWidth, drawHeight);
    ctx.filter = 'none';

    // Counteract zoom to keep points uniformly sized regardless of scale
    const invZoom = 1 / zoom;
    const ptRadius = parseInt(pointSizeSlider.value) * invZoom;

    for (const [id, pt] of Object.entries(pointsData)) {
      if (pt.observations && pt.observations[activeCamera]) {
        const obs = pt.observations[activeCamera];
        const cx = offsetX + (obs.source_x / mainImage.naturalWidth) * drawWidth;
        const cy = offsetY + (obs.source_y / mainImage.naturalHeight) * drawHeight;

        // Draw Full Circle
        ctx.beginPath();
        ctx.arc(cx, cy, ptRadius, 0, 2 * Math.PI);
        ctx.fillStyle = id === activePointId ? '#3b82f6' : '#10b981';
        ctx.fill();
        
        ctx.lineWidth = 1.5 * invZoom;
        ctx.strokeStyle = '#fff';
        ctx.stroke();
        
        // Label
        ctx.fillStyle = id === activePointId ? '#3b82f6' : '#10b981';
        ctx.font = `bold ${14 * invZoom}px Inter`;
        ctx.fillText(pt.label, cx + ptRadius + (4 * invZoom), cy - ptRadius - (2 * invZoom));
      }
    }
    
    drawThumbnails();
  }

  // --- Keyboard Shortcuts ---
  document.addEventListener('keydown', (e) => {
    // Only trigger if we are actively on the annotation view
    if (document.getElementById('view-3').classList.contains('active')) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      
      if (e.key === '1') {
        toolAnnotate.click();
      } else if (e.key === '2') {
        toolPan.click();
      } else if (e.key === '3') {
        e.preventDefault();
        btnAddPoint.click();
      }
    }
  });

  // --- Sidebar Logic ---
  btnAddPoint.addEventListener('click', () => {
    const label = pointLabelInput.value.trim() || `Point ${pointCounter}`;
    const id = `P${pointCounter.toString().padStart(3, '0')}`;
    pointsData[id] = { id, label, sigma_px: 6.0, observations: {} };
    activePointId = id;
    pointCounter++;
    pointLabelInput.value = '';
    renderSidebar();
    drawCanvas();
  });

  function renderSidebar() {
    let ptsHtml = `<h3>Points</h3>`;
    for (const [id, pt] of Object.entries(pointsData)) {
      const isActive = id === activePointId;
      const obsCount = Object.keys(pt.observations || {}).length;
      ptsHtml += `
        <div style="padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 4px; cursor: pointer; background: ${isActive ? 'var(--accent-glow)' : 'rgba(255,255,255,0.05)'}; border: 1px solid ${isActive ? 'var(--accent-color)' : 'transparent'};" onclick="setActivePoint('${id}')">
          <div style="display:flex; justify-content:space-between;">
            <strong>${pt.label}</strong> <span style="font-size:0.8rem; color:${obsCount >= 2 ? 'var(--success)' : 'var(--warning)'};">${obsCount} views</span>
          </div>
        </div>
      `;
    }
    pointsListDiv.innerHTML = ptsHtml;

    let distHtml = `<h3>Known Distances (mm)</h3>`;
    distancesData.forEach((d, idx) => {
      distHtml += `
        <div style="background: rgba(255,255,255,0.05); padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 4px; font-size: 0.9rem;">
          ${pointsData[d.point_a]?.label} ↔ ${pointsData[d.point_b]?.label}: <strong>${d.distance_mm}mm</strong>
          <div style="margin-top:0.25rem;">
            <label style="font-size: 0.8rem;"><input type="checkbox" ${d.holdout ? 'checked' : ''} onchange="toggleHoldout(${idx}, this.checked)"> Holdout for QA</label>
          </div>
        </div>
      `;
    });
    distHtml += `<button class="btn" onclick="addDistancePrompt()" style="width: 100%; font-size: 0.85rem; margin-top: 0.5rem;"><i class="fa-solid fa-plus"></i> Add Distance</button>`;
    distancesListDiv.innerHTML = distHtml;
  }

  window.setActivePoint = (id) => {
    activePointId = id;
    renderSidebar();
    drawCanvas();
  };

  window.toggleHoldout = (idx, val) => {
    distancesData[idx].holdout = val;
    distancesData[idx].use_for_refinement = !val;
  };

  window.addDistancePrompt = () => {
    const ptIds = Object.keys(pointsData);
    if(ptIds.length < 2) { alert('Need at least 2 points to add a distance.'); return; }
    
    const pAStr = prompt("Enter first point ID (e.g. P001):\n" + ptIds.map(id => id + ": " + pointsData[id].label).join("\n"));
    if (!pAStr) return;
    const pA = ptIds.find(id => id.toLowerCase() === pAStr.trim().toLowerCase() || pointsData[id].label.toLowerCase() === pAStr.trim().toLowerCase());
    
    const pBStr = prompt("Enter second point ID:");
    if (!pBStr) return;
    const pB = ptIds.find(id => id.toLowerCase() === pBStr.trim().toLowerCase() || pointsData[id].label.toLowerCase() === pBStr.trim().toLowerCase());

    const dist = prompt("Enter true distance in mm:");
    if(pA && pB && dist && pointsData[pA] && pointsData[pB]) {
      distancesData.push({
        id: `D${distanceCounter.toString().padStart(3, '0')}`,
        point_a: pA, point_b: pB,
        distance_mm: parseFloat(dist), sigma_mm: 0.5,
        holdout: false, use_for_refinement: true
      });
      distanceCounter++;
      renderSidebar();
    } else {
      alert("Invalid points or distance provided.");
    }
  };

  // --- Step 4: Run Refinement & Quality Gate ---
  btnNext3.addEventListener('click', async () => {
    // --- Client-side validation -------------------------------------
    const pts = Object.values(pointsData);
    const triangulable = pts.filter(p => Object.keys(p.observations || {}).length >= 2);
    if (triangulable.length < 2) {
      alert('Need at least 2 points, each clicked in 2 or more views, before refining.');
      return;
    }
    if (distancesData.length === 0) {
      alert('Add at least 1 known distance (mm) so the metric scale can be set.');
      return;
    }
    if (!distancesData.some(d => d.holdout)) {
      const proceed = confirm(
        'No held-out distance is reserved.\n\n' +
        'The held-out distance is the real trust test for the calibration. ' +
        'Without it you cannot verify accuracy.\n\nProceed anyway?'
      );
      if (!proceed) return;
    }

    btnNext3.disabled = true;
    btnNext3.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Refining...';

    const payload = {
      schema_version: 3,
      distance_units: "mm",
      cameras: camerasInFrame,
      points: Object.values(pointsData),
      distances: distancesData
    };

    try {
      const res = await fetch('/api/refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.status === 'success') {
        renderQualityGate(data.summary, data.prior_stats, data.has_holdout);
        goToStep(3);
      } else {
        alert("Refinement error: " + data.message);
      }
    } catch (e) {
      alert("Network error: " + e.message);
    } finally {
      btnNext3.innerHTML = 'Run Refinement <i class="fa-solid fa-check-double"></i>';
      btnNext3.disabled = false;
    }
  });

  function renderQualityGate(summary, prior_stats, hasHoldout) {
    const qg = document.querySelector('.quality-gate');

    // Tier 1 — gross-error filter (reprojection RMSE). Low is necessary, not sufficient.
    let rmseRaw = summary?.reprojection_px?.refined?.rmse;
    let rmsePx = (rmseRaw === undefined || rmseRaw === null) ? "N/A" : parseFloat(rmseRaw);
    let tier1Status = "status-warn", tier1Text = "N/A";
    if (rmsePx !== "N/A") {
      tier1Text = rmsePx < 5.0 ? "PASS" : "FAIL";
      tier1Status = rmsePx < 5.0 ? "status-pass" : "status-fail";
      rmsePx = rmsePx.toFixed(2);
    }

    // Tier 2 — rig quality (VGGT prior rotation spread).
    let rotSpread = "N/A";
    let tier2Status = "status-warn", tier2Text = "N/A";
    if (prior_stats?.cameras) {
      const sigmas = Object.values(prior_stats.cameras).map(c => c.rotation_sigma_deg || 0);
      const mean = sigmas.reduce((a, b) => a + b, 0) / sigmas.length;
      rotSpread = mean.toFixed(2);
      tier2Text = mean <= 5.0 ? "PASS" : "WARN";
      tier2Status = mean <= 5.0 ? "status-pass" : "status-warn";
    }

    // Tier 3 — acceptance test (held-out distance error). This is the real number.
    let holdout = summary?.distance_residual_mm?.refined_holdout?.rmse;
    let holdoutHtml;
    if (holdout !== undefined && holdout !== null) {
      const val = parseFloat(holdout);
      const trusted = val < 2.0;
      holdoutHtml = `
        <div style="color: var(--text-secondary);">Tier 3: Held-out Distance Error</div>
        <div class="metric-value" style="color: var(--accent-color);">${val.toFixed(2)}<span style="font-size: 1rem;">mm</span></div>
        <div class="metric-status ${trusted ? 'status-pass' : 'status-warn'}">${trusted ? 'TRUSTED' : 'CHECK'}</div>
        <p style="font-size: 0.8rem; margin-top: 1rem;">Out-of-sample accuracy on the reserved distance.</p>
      `;
    } else {
      holdoutHtml = `
        <div style="color: var(--text-secondary);">Tier 3: Held-out Distance Error</div>
        <div class="metric-value" style="color: var(--text-secondary);">NO DATA</div>
        <div class="metric-status status-warn">NO HOLDOUT</div>
        <p style="font-size: 0.8rem; margin-top: 1rem;">Reserve a known distance as hold-out — it is the only out-of-sample trust test. Reprojection alone does not predict accuracy.</p>
      `;
    }

    qg.innerHTML = `
      <div class="metric-card">
        <div style="color: var(--text-secondary);">Tier 1: Reprojection RMSE</div>
        <div class="metric-value">${rmsePx}<span style="font-size: 1rem;">px</span></div>
        <div class="metric-status ${tier1Status}">${tier1Text}</div>
        <p style="font-size: 0.8rem; margin-top: 1rem;">Gross-error filter (should be &lt; ~5px). Low value is necessary, not a pass on its own.</p>
      </div>
      <div class="metric-card">
        <div style="color: var(--text-secondary);">Tier 2: Prior Rotation Spread</div>
        <div class="metric-value">${rotSpread}<span style="font-size: 1rem;">°</span></div>
        <div class="metric-status ${tier2Status}">${tier2Text}</div>
        <p style="font-size: 0.8rem; margin-top: 1rem;">High spread ⇒ hard rig, prior less reliable.</p>
      </div>
      <div class="metric-card" style="border-color: var(--accent-color); background: rgba(59, 130, 246, 0.05);">
        ${holdoutHtml}
      </div>
    `;
  }

  // --- Export ------------------------------------------------------------
  const btnExportAnipose = document.getElementById('btn-export-anipose');
  const btnExportJson = document.getElementById('btn-export-json');
  if (btnExportAnipose) {
    btnExportAnipose.addEventListener('click', () => { window.location.href = '/api/export?fmt=anipose'; });
  }
  if (btnExportJson) {
    btnExportJson.addEventListener('click', () => { window.location.href = '/api/export?fmt=json'; });
  }
});
