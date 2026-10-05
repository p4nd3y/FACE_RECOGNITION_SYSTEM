/**
 * AURA Biometric Intelligence Platform - Enterprise Client Application
 * Handles real-time video stream, KNN inference telemetry, enrollment,
 * registry management, attendance logs, and Web Audio synthesizers.
 */

class BiometricApp {
  constructor() {
    this.stream = null;
    this.isStreaming = false;
    this.isProcessingFrame = false;
    this.audioEnabled = true;
    this.audioCtx = null;
    this.activeTab = 'tab-kiosk';

    // Enrollment state
    this.enrollStream = null;
    this.capturedCrops = [];
    this.burstInProgress = false;

    // Metrics tracking
    this.frameCount = 0;
    this.lastFpsTime = performance.now();
    this.currentFps = 0.0;

    this.initElements();
    this.initEventListeners();
    this.initClock();
    this.handleRoute();
    this.loadSystemData();
  }

  initElements() {
    // Nav
    this.navTabs = document.querySelectorAll('.nav-tab');
    this.tabPanes = document.querySelectorAll('.tab-pane');

    // Header Telemetry
    this.headerModelVal = document.getElementById('headerModelVal');
    this.headerSubjectsVal = document.getElementById('headerSubjectsVal');
    this.systemClockDisplay = document.getElementById('systemClockDisplay');
    this.streamStatusText = document.getElementById('streamStatusText');

    // Kiosk elements
    this.video = document.getElementById('webcamVideo');
    this.canvas = document.getElementById('overlayCanvas');
    this.staticPreviewImg = document.getElementById('staticPreviewImg');
    this.startCameraBtn = document.getElementById('startCameraBtn');
    this.stopCameraBtn = document.getElementById('stopCameraBtn');
    this.captureSnapshotBtn = document.getElementById('captureSnapshotBtn');
    this.switchSourceBtn = document.getElementById('switchSourceBtn');
    this.staticImageInput = document.getElementById('staticImageInput');
    this.uploadPromptBtn = document.getElementById('uploadImagePromptBtn');
    this.idleOverlay = document.getElementById('viewportIdleOverlay');
    this.cameraStatusBadge = document.getElementById('cameraStatusBadge');
    this.cameraStatusText = document.getElementById('cameraStatusText');
    this.fpsCounter = document.getElementById('fpsCounter');
    this.latencyDisplay = document.getElementById('latencyDisplay');
    this.facesCountDisplay = document.getElementById('facesCountDisplay');
    this.metricTypeDisplay = document.getElementById('metricTypeDisplay');
    this.toggleAudioBtn = document.getElementById('toggleAudioBtn');
    this.scanningLaser = document.getElementById('scanningLaser');

    // Match Card elements
    this.matchStatusBadge = document.getElementById('matchStatusBadge');
    this.matchAvatarImg = document.getElementById('matchAvatarImg');
    this.matchCircleProgress = document.getElementById('matchCircleProgress');
    this.matchCircleText = document.getElementById('matchCircleText');
    this.matchNameDisplay = document.getElementById('matchNameDisplay');
    this.matchEmpIdDisplay = document.getElementById('matchEmpIdDisplay');
    this.matchDeptDisplay = document.getElementById('matchDeptDisplay');
    this.matchDistDisplay = document.getElementById('matchDistDisplay');
    this.matchTimestampDisplay = document.getElementById('matchTimestampDisplay');
    this.matchNeighborList = document.getElementById('matchNeighborList');
    this.liveActivityList = document.getElementById('liveActivityList');
    this.refreshLiveStreamBtn = document.getElementById('refreshLiveStreamBtn');

    // Enrollment elements
    this.enrollFullName = document.getElementById('enrollFullName');
    this.enrollEmpId = document.getElementById('enrollEmpId');
    this.enrollDepartment = document.getElementById('enrollDepartment');
    this.enrollRole = document.getElementById('enrollRole');
    this.enrollEmail = document.getElementById('enrollEmail');
    this.modeWebcamBurstBtn = document.getElementById('modeWebcamBurstBtn');
    this.modeFileUploadBtn = document.getElementById('modeFileUploadBtn');
    this.burstPanel = document.getElementById('burstPanel');
    this.dropzonePanel = document.getElementById('dropzonePanel');
    this.burstCountDisplay = document.getElementById('burstCountDisplay');
    this.burstStatusMsg = document.getElementById('burstStatusMsg');
    this.burstProgressFill = document.getElementById('burstProgressFill');
    this.startBurstCaptureBtn = document.getElementById('startBurstCaptureBtn');
    this.dropzoneBox = document.getElementById('dropzoneBox');
    this.batchFilesInput = document.getElementById('batchFilesInput');
    this.browseFilesBtn = document.getElementById('browseFilesBtn');
    this.submitEnrollmentBtn = document.getElementById('submitEnrollmentBtn');
    this.clearSamplesBtn = document.getElementById('clearSamplesBtn');
    this.enrollVideo = document.getElementById('enrollVideo');
    this.enrollCanvas = document.getElementById('enrollCanvas');
    this.sampleCropsGrid = document.getElementById('sampleCropsGrid');

    // Registry elements
    this.registrySearchInput = document.getElementById('registrySearchInput');
    this.registryDeptFilter = document.getElementById('registryDeptFilter');
    this.reloadRegistryBtn = document.getElementById('reloadRegistryBtn');
    this.subjectCardsGrid = document.getElementById('subjectCardsGrid');

    // Attendance elements
    this.kpiTotalCheckins = document.getElementById('kpiTotalCheckins');
    this.kpiVerifiedCount = document.getElementById('kpiVerifiedCount');
    this.kpiUnknownCount = document.getElementById('kpiUnknownCount');
    this.kpiAvgConfidence = document.getElementById('kpiAvgConfidence');
    this.attendanceStatusFilter = document.getElementById('attendanceStatusFilter');
    this.exportCsvBtn = document.getElementById('exportCsvBtn');
    this.clearAttendanceBtn = document.getElementById('clearAttendanceBtn');
    this.attendanceTableBody = document.getElementById('attendanceTableBody');

    // Diagnostics elements
    this.knnKSlider = document.getElementById('knnKSlider');
    this.knnKValDisplay = document.getElementById('knnKValDisplay');
    this.knnMetricSelect = document.getElementById('knnMetricSelect');
    this.knnWeightsSelect = document.getElementById('knnWeightsSelect');
    this.knnThresholdSlider = document.getElementById('knnThresholdSlider');
    this.knnThresholdValDisplay = document.getElementById('knnThresholdValDisplay');
    this.applyKnnConfigBtn = document.getElementById('applyKnnConfigBtn');
    this.refreshDiagBtn = document.getElementById('refreshDiagBtn');
    this.distanceMatrixWrapper = document.getElementById('distanceMatrixWrapper');

    // Toast
    this.toastContainer = document.getElementById('toastContainer');
  }

  initEventListeners() {
    // Tab Switching
    this.navTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const target = tab.getAttribute('data-tab');
        this.switchTab(target);
      });
    });

    window.addEventListener('hashchange', () => this.handleRoute());

    // Stream Controls
    this.startCameraBtn.addEventListener('click', () => this.startWebcam());
    this.stopCameraBtn.addEventListener('click', () => this.stopWebcam());
    this.captureSnapshotBtn.addEventListener('click', () => this.processSingleFrame(true));

    this.switchSourceBtn.addEventListener('click', () => this.staticImageInput.click());
    this.uploadPromptBtn.addEventListener('click', () => this.staticImageInput.click());
    this.staticImageInput.addEventListener('change', (e) => this.handleStaticImageUpload(e));

    this.toggleAudioBtn.addEventListener('click', () => {
      this.audioEnabled = !this.audioEnabled;
      this.toggleAudioBtn.innerHTML = this.audioEnabled ? '<i class="fa-solid fa-volume-high"></i>' : '<i class="fa-solid fa-volume-xmark"></i>';
      this.showToast(`Biometric audio chime ${this.audioEnabled ? 'enabled' : 'muted'}`, 'info');
    });

    this.refreshLiveStreamBtn.addEventListener('click', () => this.loadRecentAttendance());

    // Enrollment Mode Toggles
    this.modeWebcamBurstBtn.addEventListener('click', () => {
      this.modeWebcamBurstBtn.classList.add('active');
      this.modeFileUploadBtn.classList.remove('active');
      this.burstPanel.style.display = 'block';
      this.dropzonePanel.style.display = 'none';
      this.startEnrollCamera();
    });

    this.modeFileUploadBtn.addEventListener('click', () => {
      this.modeFileUploadBtn.classList.add('active');
      this.modeWebcamBurstBtn.classList.remove('active');
      this.burstPanel.style.display = 'none';
      this.dropzonePanel.style.display = 'block';
      this.stopEnrollCamera();
    });

    this.startBurstCaptureBtn.addEventListener('click', () => this.executeBurstCapture());
    this.browseFilesBtn.addEventListener('click', () => this.batchFilesInput.click());
    this.dropzoneBox.addEventListener('click', () => this.batchFilesInput.click());
    this.batchFilesInput.addEventListener('change', (e) => this.handleBatchFilesUpload(e));

    // Drag and Drop
    this.dropzoneBox.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.dropzoneBox.style.borderColor = 'var(--cyan-neon)';
    });
    this.dropzoneBox.addEventListener('dragleave', () => {
      this.dropzoneBox.style.borderColor = '';
    });
    this.dropzoneBox.addEventListener('drop', (e) => {
      e.preventDefault();
      this.dropzoneBox.style.borderColor = '';
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        this.processBatchFiles(e.dataTransfer.files);
      }
    });

    this.submitEnrollmentBtn.addEventListener('click', () => this.submitEnrollment());
    this.clearSamplesBtn.addEventListener('click', () => this.clearEnrollmentSamples());

    // Registry Filter & Reload
    this.registrySearchInput.addEventListener('input', () => this.filterRegistry());
    this.registryDeptFilter.addEventListener('change', () => this.filterRegistry());
    this.reloadRegistryBtn.addEventListener('click', () => this.loadRegistry(true));

    // Attendance Filters & Export
    this.attendanceStatusFilter.addEventListener('change', () => this.loadAttendanceTable());
    this.exportCsvBtn.addEventListener('click', () => window.open('/api/attendance/export', '_blank'));
    this.clearAttendanceBtn.addEventListener('click', () => this.clearAttendanceLogs());

    // KNN Config Sliders
    this.knnKSlider.addEventListener('input', (e) => this.knnKValDisplay.innerText = e.target.value);
    this.knnThresholdSlider.addEventListener('input', (e) => this.knnThresholdValDisplay.innerText = e.target.value);
    this.applyKnnConfigBtn.addEventListener('click', () => this.applyKNNConfig());
    this.refreshDiagBtn.addEventListener('click', () => this.loadKNNDiagnostics());
  }

  initClock() {
    const update = () => {
      const now = new Date();
      this.systemClockDisplay.innerText = now.toUTCString().slice(17, 25) + ' UTC';
    };
    update();
    setInterval(update, 1000);
  }

  handleRoute() {
    const hash = window.location.hash.replace('#', '');
    if (hash && document.getElementById(hash)) {
      this.switchTab(hash);
    } else {
      this.switchTab('tab-kiosk');
    }
  }

  switchTab(tabId) {
    this.activeTab = tabId;
    window.location.hash = tabId;

    this.navTabs.forEach(tab => {
      const isTarget = tab.getAttribute('data-tab') === tabId;
      tab.classList.toggle('active', isTarget);
      tab.setAttribute('aria-selected', isTarget);
    });

    this.tabPanes.forEach(pane => {
      pane.classList.toggle('active', pane.id === tabId);
    });

    if (tabId === 'tab-enroll') {
      if (this.modeWebcamBurstBtn.classList.contains('active')) {
        this.startEnrollCamera();
      }
    } else {
      this.stopEnrollCamera();
    }

    if (tabId === 'tab-registry') this.loadRegistry();
    if (tabId === 'tab-attendance') this.loadAttendanceTable();
    if (tabId === 'tab-diagnostics') this.loadKNNDiagnostics();
  }

  // -------------------------------------------------------------------------
  // Web Audio Synthesizer (Zero External Dependencies)
  // -------------------------------------------------------------------------
  playChime(type = 'success') {
    if (!this.audioEnabled) return;
    try {
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }

      const ctx = this.audioCtx;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.connect(gain);
      gain.connect(ctx.destination);

      const now = ctx.currentTime;

      if (type === 'success') {
        // High-tech ascending harmonic ping
        osc.type = 'sine';
        osc.frequency.setValueAtTime(660, now);
        osc.frequency.exponentialRampToValueAtTime(1320, now + 0.14);
        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.28);
        osc.start(now);
        osc.stop(now + 0.28);
      } else if (type === 'alert') {
        // Warning alert buzz
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(320, now);
        osc.frequency.setValueAtTime(240, now + 0.1);
        gain.gain.setValueAtTime(0.15, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
        osc.start(now);
        osc.stop(now + 0.35);
      } else {
        // Gentle click
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(880, now);
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
        osc.start(now);
        osc.stop(now + 0.08);
      }
    } catch (e) {
      console.warn('Audio chime playback failed:', e);
    }
  }

  // -------------------------------------------------------------------------
  // Live Kiosk Webcam Stream & HUD Overlay
  // -------------------------------------------------------------------------
  async startWebcam() {
    try {
      this.streamStatusText.innerText = 'INITIALIZING WEBCAM HARDWARE...';
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });

      this.stream = stream;
      this.video.srcObject = stream;
      this.video.style.display = 'block';
      this.staticPreviewImg.style.display = 'none';
      this.idleOverlay.style.display = 'none';

      this.startCameraBtn.style.display = 'none';
      this.stopCameraBtn.style.display = 'inline-flex';
      this.captureSnapshotBtn.style.display = 'inline-flex';
      this.scanningLaser.style.display = 'block';

      this.cameraStatusBadge.className = 'hud-badge';
      this.cameraStatusText.innerText = 'LIVE OPTICAL FEED';
      this.streamStatusText.innerText = 'STREAM ACTIVE • REAL-TIME K-NN INFERENCE ONLINE';

      this.isStreaming = true;
      this.video.play();
      this.showToast('Webcam stream initialized successfully', 'success');

      this.video.onloadedmetadata = () => {
        this.canvas.width = this.video.videoWidth || 640;
        this.canvas.height = this.video.videoHeight || 480;
        this.runInferenceLoop();
      };
    } catch (err) {
      console.error('Camera access error:', err);
      this.showToast('Could not access camera. Please check browser permissions.', 'alert');
      this.streamStatusText.innerText = 'CAMERA ACCESS DENIED • USE PHOTO UPLOAD';
    }
  }

  stopWebcam() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.isStreaming = false;
    this.video.style.display = 'none';
    this.idleOverlay.style.display = 'flex';
    this.startCameraBtn.style.display = 'inline-flex';
    this.stopCameraBtn.style.display = 'none';
    this.captureSnapshotBtn.style.display = 'none';
    this.scanningLaser.style.display = 'none';

    this.cameraStatusText.innerText = 'CAMERA IDLE';
    this.streamStatusText.innerText = 'SENSOR STANDBY • HARDWARE READY';
    this.clearCanvas();
  }

  async runInferenceLoop() {
    if (!this.isStreaming) return;

    if (!this.isProcessingFrame) {
      await this.processSingleFrame();
    }

    // Update FPS telemetry
    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsTime >= 1000) {
      this.currentFps = (this.frameCount * 1000) / (now - this.lastFpsTime);
      this.fpsCounter.innerText = this.currentFps.toFixed(1);
      this.frameCount = 0;
      this.lastFpsTime = now;
    }

    requestAnimationFrame(() => this.runInferenceLoop());
  }

  async processSingleFrame(forceLog = false) {
    if (!this.video.videoWidth || this.isProcessingFrame) return;

    this.isProcessingFrame = true;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 640;
    tempCanvas.height = 480;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(this.video, 0, 0, 640, 480);
    const base64Data = tempCanvas.toDataURL('image/jpeg', 0.75);

    try {
      const response = await fetch('/api/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: base64Data,
          camera_id: 'Kiosk_Cam_01',
          draw_hud: false,
          auto_log: true,
        }),
      });

      const data = await response.json();
      if (data.success) {
        this.renderViewportDetections(data.results, 640, 480);
        this.latencyDisplay.innerText = `${data.inference_time_ms} ms`;
        this.facesCountDisplay.innerText = data.faces_detected;

        if (data.results.length > 0) {
          const primary = data.results[0];
          this.updateMatchCard(primary);
          if (primary.logged) {
            this.playChime(primary.is_unknown ? 'alert' : 'success');
            this.loadRecentAttendance();
            this.loadAttendanceMetricsOnly();
          }
        }
      }
    } catch (err) {
      console.warn('Inference error:', err);
    } finally {
      this.isProcessingFrame = false;
    }
  }

  renderViewportDetections(results, srcW, srcH) {
    const ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (!results || results.length === 0) return;

    const scaleX = this.canvas.width / srcW;
    const scaleY = this.canvas.height / srcH;

    results.forEach(res => {
      const bbox = res.bbox;
      const x = bbox.x * scaleX;
      const y = bbox.y * scaleY;
      const w = bbox.width * scaleX;
      const h = bbox.height * scaleY;

      const isUnknown = res.is_unknown;
      const color = isUnknown ? '#ef4444' : '#00f2fe';
      const bracketLen = Math.min(w, h) * 0.22;

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;

      // Top-Left
      ctx.beginPath();
      ctx.moveTo(x, y + bracketLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + bracketLen, y);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(x + w - bracketLen, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + bracketLen);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(x, y + h - bracketLen);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + bracketLen, y + h);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(x + w - bracketLen, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - bracketLen);
      ctx.stroke();

      // Label Pill
      const label = isUnknown ? 'UNKNOWN SUBJECT' : `${res.label.toUpperCase()} [${res.confidence}%]`;
      ctx.font = '600 13px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = 'rgba(6, 11, 20, 0.9)';
      ctx.fillRect(x, Math.max(0, y - 24), textWidth + 16, 22);
      ctx.strokeRect(x, Math.max(0, y - 24), textWidth + 16, 22);

      ctx.fillStyle = color;
      ctx.fillText(label, x + 8, Math.max(14, y - 8));
    });
  }

  clearCanvas() {
    const ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  // -------------------------------------------------------------------------
  // Static Photo Upload Recognition
  // -------------------------------------------------------------------------
  handleStaticImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    this.stopWebcam();
    const reader = new FileReader();
    reader.onload = async (event) => {
      const b64 = event.target.result;
      this.staticPreviewImg.src = b64;
      this.staticPreviewImg.style.display = 'block';
      this.idleOverlay.style.display = 'none';

      this.showToast('Evaluating biometric image...', 'info');

      try {
        const resp = await fetch('/api/recognize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_base64: b64,
            camera_id: 'Static_Upload',
            draw_hud: false,
            auto_log: true,
          }),
        });
        const data = await resp.json();
        if (data.success && data.results.length > 0) {
          const primary = data.results[0];
          this.updateMatchCard(primary);
          this.playChime(primary.is_unknown ? 'alert' : 'success');
          this.showToast(`Identification Complete: ${primary.label}`, primary.is_unknown ? 'alert' : 'success');
          this.loadRecentAttendance();
        } else {
          this.showToast('No frontal face detected in uploaded image.', 'alert');
        }
      } catch (err) {
        console.error(err);
        this.showToast('Failed to analyze image.', 'alert');
      }
    };
    reader.readAsDataURL(file);
  }

  // -------------------------------------------------------------------------
  // Match Showcase Card & Neighbors Update
  // -------------------------------------------------------------------------
  updateMatchCard(result) {
    const isUnknown = result.is_unknown;
    const name = result.label;
    const conf = result.confidence;
    const dist = result.distance;

    this.matchStatusBadge.className = `status-badge ${isUnknown ? 'unknown' : 'verified'}`;
    this.matchStatusBadge.innerText = isUnknown ? 'IMPOSTOR ALERT' : 'VERIFIED MATCH';

    this.matchNameDisplay.innerText = name;
    this.matchEmpIdDisplay.innerHTML = `<i class="fa-solid fa-hashtag"></i> ${result.employee_id || 'N/A'}`;
    this.matchDeptDisplay.innerHTML = `<i class="fa-solid fa-building"></i> ${result.department || 'General'}`;
    this.matchDistDisplay.innerHTML = `<i class="fa-solid fa-ruler-combined"></i> Dist (L2): ${dist}`;
    this.matchTimestampDisplay.innerHTML = `<i class="fa-solid fa-clock"></i> ${new Date().toLocaleTimeString()}`;

    // Update circular progress SVG
    this.matchCircleProgress.setAttribute('stroke-dasharray', `${conf}, 100`);
    this.matchCircleProgress.style.stroke = isUnknown ? 'var(--crimson-alert)' : 'var(--cyan-neon)';

    // Render Neighbors
    this.renderNeighborsList(result.neighbors);
  }

  renderNeighborsList(neighbors) {
    if (!neighbors || neighbors.length === 0) {
      this.matchNeighborList.innerHTML = '<div class="neighbor-empty">No neighbor data.</div>';
      return;
    }

    this.matchNeighborList.innerHTML = neighbors.map(n => `
      <div class="neighbor-row">
        <span class="neighbor-rank">#${n.rank}</span>
        <span class="neighbor-name">${n.label}</span>
        <span class="neighbor-dist">L2: ${n.distance} (w: ${n.weight})</span>
      </div>
    `).join('');
  }

  // -------------------------------------------------------------------------
  // Biometric Enrollment Studio
  // -------------------------------------------------------------------------
  async startEnrollCamera() {
    if (this.enrollStream) return;
    try {
      this.enrollStream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } });
      this.enrollVideo.srcObject = this.enrollStream;
      this.enrollVideo.play();
    } catch (err) {
      console.warn('Enrollment camera error:', err);
    }
  }

  stopEnrollCamera() {
    if (this.enrollStream) {
      this.enrollStream.getTracks().forEach(track => track.stop());
      this.enrollStream = null;
    }
  }

  async executeBurstCapture() {
    if (!this.enrollStream) {
      await this.startEnrollCamera();
    }
    if (this.burstInProgress) return;

    this.burstInProgress = true;
    this.startBurstCaptureBtn.disabled = true;
    this.burstStatusMsg.innerText = 'CAPTURING SAMPLES...';

    const targetCount = 30;
    this.capturedCrops = [];
    this.sampleCropsGrid.innerHTML = '';

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 100;
    tempCanvas.height = 100;
    const ctx = tempCanvas.getContext('2d');

    for (let i = 0; i < targetCount; i++) {
      ctx.drawImage(this.enrollVideo, 0, 0, 100, 100);
      const cropB64 = tempCanvas.toDataURL('image/jpeg', 0.85);
      this.capturedCrops.push(cropB64);

      // Add thumbnail
      const thumb = document.createElement('div');
      thumb.className = 'crop-thumbnail-item';
      thumb.innerHTML = `<img src="${cropB64}" alt="Face Crop">`;
      this.sampleCropsGrid.appendChild(thumb);

      const pct = Math.round(((i + 1) / targetCount) * 100);
      this.burstProgressFill.style.width = `${pct}%`;
      this.burstCountDisplay.innerText = `${i + 1} / ${targetCount} Samples`;

      this.playChime('click');
      await new Promise(r => setTimeout(r, 65));
    }

    this.burstStatusMsg.innerText = 'BURST ACQUISITION COMPLETE';
    this.burstInProgress = false;
    this.startBurstCaptureBtn.disabled = false;
    this.submitEnrollmentBtn.disabled = false;
    this.showToast('Acquired 30 high-definition face vectors.', 'success');
  }

  handleBatchFilesUpload(e) {
    if (e.target.files && e.target.files.length > 0) {
      this.processBatchFiles(e.target.files);
    }
  }

  processBatchFiles(files) {
    this.capturedCrops = [];
    this.sampleCropsGrid.innerHTML = '';
    const fileList = Array.from(files).slice(0, 50);

    let loaded = 0;
    fileList.forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const b64 = event.target.result;
        this.capturedCrops.push(b64);

        const thumb = document.createElement('div');
        thumb.className = 'crop-thumbnail-item';
        thumb.innerHTML = `<img src="${b64}" alt="Uploaded Crop">`;
        this.sampleCropsGrid.appendChild(thumb);

        loaded++;
        if (loaded === fileList.length) {
          this.submitEnrollmentBtn.disabled = false;
          this.showToast(`Imported ${loaded} portrait files ready for enrollment.`, 'success');
        }
      };
      reader.readAsDataURL(file);
    });
  }

  clearEnrollmentSamples() {
    this.capturedCrops = [];
    this.sampleCropsGrid.innerHTML = '<div class="empty-crops-placeholder"><i class="fa-solid fa-image"></i><p>No facial crops acquired yet.</p></div>';
    this.burstProgressFill.style.width = '0%';
    this.burstCountDisplay.innerText = '0 / 30 Samples';
    this.burstStatusMsg.innerText = 'READY';
    this.submitEnrollmentBtn.disabled = true;
  }

  async submitEnrollment() {
    const fullName = this.enrollFullName.value.trim();
    if (!fullName) {
      this.showToast('Please enter full subject name.', 'alert');
      this.enrollFullName.focus();
      return;
    }

    if (this.capturedCrops.length === 0) {
      this.showToast('Please capture or upload face image samples first.', 'alert');
      return;
    }

    this.submitEnrollmentBtn.disabled = true;
    this.submitEnrollmentBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Vectorizing & Fitting KNN...';

    const payload = {
      full_name: fullName,
      employee_id: this.enrollEmpId.value.trim() || undefined,
      department: this.enrollDepartment.value,
      role: this.enrollRole.value.trim() || 'Staff Member',
      email: this.enrollEmail.value.trim() || undefined,
      images_base64: this.capturedCrops,
    };

    try {
      const res = await fetch('/api/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.success) {
        this.showToast(`Successfully enrolled ${fullName}! K-NN Index retrained.`, 'success');
        this.playChime('success');
        this.clearEnrollmentSamples();
        this.enrollFullName.value = '';
        this.enrollEmpId.value = '';
        this.enrollRole.value = '';
        this.enrollEmail.value = '';
        this.loadSystemData();
      } else {
        this.showToast(data.detail || 'Enrollment failed.', 'alert');
      }
    } catch (err) {
      console.error(err);
      this.showToast('Server error during enrollment.', 'alert');
    } finally {
      this.submitEnrollmentBtn.disabled = false;
      this.submitEnrollmentBtn.innerHTML = '<i class="fa-solid fa-cloud-check"></i> Enroll Subject & Retrain K-NN Model';
    }
  }

  // -------------------------------------------------------------------------
  // Subject Registry Management
  // -------------------------------------------------------------------------
  async loadRegistry(syncToast = false) {
    try {
      const res = await fetch('/api/registry');
      const data = await res.json();
      this.allSubjects = data.subjects || [];
      this.renderRegistryCards(this.allSubjects);
      this.headerSubjectsVal.innerText = this.allSubjects.length;

      if (syncToast) {
        this.showToast(`Registry synchronized: ${this.allSubjects.length} enrolled subjects.`, 'success');
      }
    } catch (err) {
      console.error(err);
    }
  }

  renderRegistryCards(subjects) {
    if (!subjects || subjects.length === 0) {
      this.subjectCardsGrid.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">
          <i class="fa-solid fa-users-slash" style="font-size: 32px; margin-bottom: 10px;"></i>
          <p>No biometric profiles found in registry. Enroll personnel in the Enrollment Studio.</p>
        </div>`;
      return;
    }

    this.subjectCardsGrid.innerHTML = subjects.map(s => {
      const avatar = s.avatar_base64 || "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' fill='%2364748b' viewBox='0 0 16 16'%3E%3Cpath d='M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0z'/%3E%3Cpath fill-rule='evenodd' d='M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm8-7a7 7 0 0 0-5.468 11.37C3.242 11.226 4.805 10 8 10s4.757 1.225 5.468 2.37A7 7 0 0 0 8 1z'/%3E%3C/svg%3E";
      return `
        <div class="subject-card">
          <div class="subject-card-top">
            <div class="subject-card-avatar">
              <img src="${avatar}" alt="${s.full_name}">
            </div>
            <div>
              <div class="subject-card-name">${s.full_name}</div>
              <div class="subject-card-role">${s.role} • ${s.department}</div>
              <span class="subject-card-badge">${s.employee_id || 'ID-N/A'}</span>
            </div>
          </div>
          <div class="subject-card-metrics">
            <div>
              <div class="metric-box-label">TRAINING VECTORS</div>
              <div class="metric-box-val">${s.sample_count} Samples</div>
            </div>
            <div>
              <div class="metric-box-label">VECTOR DIMENSION</div>
              <div class="metric-box-val">${s.vector_dim}D (RGB)</div>
            </div>
          </div>
          <div class="subject-card-actions">
            <button class="btn-danger btn-sm" onclick="app.deleteSubject('${s.person_key}', '${s.full_name}')">
              <i class="fa-solid fa-trash"></i> Delete
            </button>
          </div>
        </div>
      `;
    }).join('');
  }

  filterRegistry() {
    const q = this.registrySearchInput.value.toLowerCase();
    const dept = this.registryDeptFilter.value;

    const filtered = (this.allSubjects || []).filter(s => {
      const matchText = s.full_name.toLowerCase().includes(q) || (s.employee_id && s.employee_id.toLowerCase().includes(q)) || s.role.toLowerCase().includes(q);
      const matchDept = dept === 'ALL' || s.department === dept;
      return matchText && matchDept;
    });

    this.renderRegistryCards(filtered);
  }

  async deleteSubject(personKey, name) {
    if (!confirm(`Are you sure you want to delete '${name}' from biometric registry?`)) return;

    try {
      const res = await fetch(`/api/registry/${personKey}`, { method: 'DELETE' });
      const data = await res.json();
      if (data.success) {
        this.showToast(`Deleted subject: ${name}`, 'info');
        this.loadRegistry();
        this.loadSystemData();
      }
    } catch (err) {
      console.error(err);
      this.showToast('Failed to delete subject.', 'alert');
    }
  }

  // -------------------------------------------------------------------------
  // Attendance Logs & KPI Telemetry
  // -------------------------------------------------------------------------
  async loadAttendanceTable() {
    const status = this.attendanceStatusFilter.value;
    try {
      const res = await fetch(`/api/attendance?status=${status}&limit=100`);
      const data = await res.json();

      // KPIs
      this.updateKPIs(data.metrics);

      // Rows
      if (!data.logs || data.logs.length === 0) {
        this.attendanceTableBody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 30px;">No attendance records found.</td></tr>`;
        return;
      }

      this.attendanceTableBody.innerHTML = data.logs.map(log => {
        const isVerified = log.status === 'VERIFIED';
        return `
          <tr>
            <td>#${log.id}</td>
            <td>${log.timestamp}</td>
            <td><strong>${log.person_name}</strong></td>
            <td>${log.employee_id}</td>
            <td>${log.department}</td>
            <td><strong style="color: ${isVerified ? 'var(--cyan-neon)' : 'var(--crimson-alert)'}">${log.confidence_pct}%</strong></td>
            <td>${log.distance_score}</td>
            <td><span class="table-status-badge ${isVerified ? 'verified' : 'alert'}">${log.status}</span></td>
            <td>${log.camera_id}</td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      console.error(err);
    }
  }

  async loadAttendanceMetricsOnly() {
    try {
      const res = await fetch('/api/attendance?limit=1');
      const data = await res.json();
      this.updateKPIs(data.metrics);
    } catch (err) {
      console.warn(err);
    }
  }

  updateKPIs(m) {
    if (!m) return;
    this.kpiTotalCheckins.innerText = m.total_today;
    this.kpiVerifiedCount.innerText = m.verified_today;
    this.kpiUnknownCount.innerText = m.unknown_today;
    this.kpiAvgConfidence.innerText = `${m.avg_confidence}%`;
  }

  async loadRecentAttendance() {
    try {
      const res = await fetch('/api/attendance?limit=6');
      const data = await res.json();
      if (!data.logs || data.logs.length === 0) {
        this.liveActivityList.innerHTML = '<div class="neighbor-empty">No check-in events recorded today.</div>';
        return;
      }

      this.liveActivityList.innerHTML = data.logs.map(l => {
        const isVerified = l.status === 'VERIFIED';
        const avatar = l.snapshot_b64 || "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' fill='%2364748b' viewBox='0 0 16 16'%3E%3Cpath d='M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0z'/%3E%3Cpath fill-rule='evenodd' d='M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm8-7a7 7 0 0 0-5.468 11.37C3.242 11.226 4.805 10 8 10s4.757 1.225 5.468 2.37A7 7 0 0 0 8 1z'/%3E%3C/svg%3E";
        return `
          <div class="activity-item">
            <div class="activity-user">
              <img src="${avatar}" class="activity-user-avatar" alt="Avatar">
              <div>
                <div class="activity-user-name">${l.person_name}</div>
                <div class="activity-time">${l.department} • Conf: ${l.confidence_pct}%</div>
              </div>
            </div>
            <span class="table-status-badge ${isVerified ? 'verified' : 'alert'}">${l.status}</span>
          </div>
        `;
      }).join('');
    } catch (err) {
      console.warn(err);
    }
  }

  async clearAttendanceLogs() {
    if (!confirm('Are you sure you want to clear all historical attendance records?')) return;
    try {
      await fetch('/api/attendance/clear', { method: 'POST' });
      this.showToast('Attendance records cleared.', 'info');
      this.loadAttendanceTable();
      this.loadRecentAttendance();
    } catch (err) {
      console.error(err);
    }
  }

  // -------------------------------------------------------------------------
  // KNN Diagnostic & Calibration Lab
  // -------------------------------------------------------------------------
  async loadKNNDiagnostics() {
    try {
      const res = await fetch('/api/knn/diagnostics');
      const data = await res.json();

      this.knnKSlider.value = data.k;
      this.knnKValDisplay.innerText = data.k;
      this.knnMetricSelect.value = data.metric;
      this.knnWeightsSelect.value = data.weights;
      this.knnThresholdSlider.value = data.unknown_threshold;
      this.knnThresholdValDisplay.innerText = data.unknown_threshold;

      this.headerModelVal.innerText = `KNN (k=${data.k}, ${data.metric.toUpperCase()})`;

      // Render Matrix
      this.renderDistanceMatrix(data.inter_class_distance_matrix);
    } catch (err) {
      console.error(err);
    }
  }

  renderDistanceMatrix(matrix) {
    if (!matrix || Object.keys(matrix).length === 0) {
      this.distanceMatrixWrapper.innerHTML = '<div class="neighbor-empty">Distance matrix unavailable (insufficient subjects).</div>';
      return;
    }

    const keys = Object.keys(matrix);
    let html = '<table class="matrix-table"><thead><tr><th>SUBJECT / CLUSTER</th>';
    keys.forEach(k => html += `<th>${k}</th>`);
    html += '</tr></thead><tbody>';

    keys.forEach(rowKey => {
      html += `<tr><th>${rowKey}</th>`;
      keys.forEach(colKey => {
        const val = matrix[rowKey][colKey];
        const isSelf = rowKey === colKey;
        html += `<td class="matrix-cell ${isSelf ? 'self' : ''}">${val}</td>`;
      });
      html += '</tr>';
    });

    html += '</tbody></table>';
    this.distanceMatrixWrapper.innerHTML = html;
  }

  async applyKNNConfig() {
    const payload = {
      k: parseInt(this.knnKSlider.value, 10),
      metric: this.knnMetricSelect.value,
      weights: this.knnWeightsSelect.value,
      unknown_threshold: parseFloat(this.knnThresholdSlider.value),
    };

    try {
      const res = await fetch('/api/knn/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.success) {
        this.showToast('KNN hyperparameters updated successfully!', 'success');
        this.metricTypeDisplay.innerText = `${payload.metric.toUpperCase()} (${payload.weights})`;
        this.loadKNNDiagnostics();
      }
    } catch (err) {
      console.error(err);
      this.showToast('Failed to apply configuration.', 'alert');
    }
  }

  async loadSystemData() {
    await this.loadRegistry();
    await this.loadRecentAttendance();
    await this.loadKNNDiagnostics();
  }

  showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? 'fa-circle-check' : (type === 'alert' ? 'fa-triangle-exclamation' : 'fa-circle-info');
    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${message}</span>`;
    this.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }
}

// Global bootstrap
document.addEventListener('DOMContentLoaded', () => {
  window.app = new BiometricApp();
});
