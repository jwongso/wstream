class WStreamASR {
    constructor() {
        // Web audio context
        this.context = null;

        // Audio data
        this.audio = null;
        this.audio0 = null;

        // The stream handle
        this.handle = null;

        // Module
        this.Module = null;

        // Recording state
        this.mediaRecorder = null;
        this.currentStream = null;
        this.doRecording = false;
        this.startTime = 0;
        this.isProcessing = false;
        this.lastStatus = '';

        // Constants
        this.kSampleRate = 16000;
        this.kRestartRecording_s = 120;
        this.kIntervalAudio_ms = 3000;
        this.kUpdateInterval_ms = 250;

        // Performance tracking
        this.audioChunksProcessed = 0;
        this.lastAudioPush = Date.now();

        // Filter settings
        this.showMetrics = true;
        this.confidenceFilterValue = 0; // Default: show all
        this.filteredCount = 0;
        this.transcriptionData = [];

        // UI elements
        this.setupUI();
        this.setupEventListeners();
    }

    setupUI() {
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.transcriptionBox = document.getElementById('transcription');
        this.initBtn = document.getElementById('initBtn');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.modelSelect = document.getElementById('modelSelect');
        this.showMetricsCheckbox = document.getElementById('showMetrics');
        this.confidenceFilterInput = document.getElementById('confidenceFilter');
        this.confidenceFilterValueSpan = document.getElementById('confidenceFilterValue');
    }

    setupEventListeners() {
        this.initBtn.addEventListener('click', () => this.initialize());
        this.startBtn.addEventListener('click', () => this.onStart());
        this.stopBtn.addEventListener('click', () => this.onStop());
        this.clearBtn.addEventListener('click', () => this.clearTranscription());

        // Metrics toggle
        if (this.showMetricsCheckbox) {
            this.showMetricsCheckbox.addEventListener('change', (e) => {
                this.showMetrics = e.target.checked;
                this.toggleMetricsDisplay();
            });

            // Load saved preference
            const savedPref = localStorage.getItem('showMetrics');
            if (savedPref !== null) {
                this.showMetrics = savedPref === 'true';
                this.showMetricsCheckbox.checked = this.showMetrics;
            }
        }

        // Confidence filter
        if (this.confidenceFilterInput) {
            this.confidenceFilterInput.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.confidenceFilterValue = isNaN(value) ? 0 : value;

                if (this.confidenceFilterValueSpan) {
                    this.confidenceFilterValueSpan.textContent = this.confidenceFilterValue.toFixed(0);
                }

                // Save preference
                localStorage.setItem('confidenceFilter', this.confidenceFilterValue.toString());

                // Apply filter to existing entries
                this.applyConfidenceFilter();
            });

            // Load saved filter value
            const savedFilter = localStorage.getItem('confidenceFilter');
            if (savedFilter !== null) {
                const value = parseFloat(savedFilter);
                this.confidenceFilterValue = isNaN(value) ? 0 : value;
                this.confidenceFilterInput.value = this.confidenceFilterValue;
                if (this.confidenceFilterValueSpan) {
                    this.confidenceFilterValueSpan.textContent = this.confidenceFilterValue.toFixed(0);
                }
            }
        }
    }

    toggleMetricsDisplay() {
        localStorage.setItem('showMetrics', this.showMetrics);

        const allMetrics = this.transcriptionBox.querySelectorAll('.transcription-metrics');
        allMetrics.forEach(metrics => {
            if (this.showMetrics) {
                metrics.classList.remove('hidden');
            } else {
                metrics.classList.add('hidden');
            }
        });
    }

    applyConfidenceFilter() {
        let visibleCount = 0;
        let filteredCount = 0;

        // Apply filter to all existing entries
        this.transcriptionData.forEach(data => {
            if (data.confidence < this.confidenceFilterValue) {
                data.element.classList.add('filtered-out');
                filteredCount++;
            } else {
                data.element.classList.remove('filtered-out');
                visibleCount++;
            }
        });

        this.filteredCount = filteredCount;
        this.updateFilterStatus();
    }

    updateFilterStatus() {
        let statusElement = document.getElementById('filterStatus');

        // Ensure confidenceFilterValue is defined
        const filterValue = this.confidenceFilterValue || 0;

        if (!statusElement && this.filteredCount > 0) {
            statusElement = document.createElement('div');
            statusElement.id = 'filterStatus';
            statusElement.className = 'filter-status';
            const transcriptionContainer = this.transcriptionBox.parentNode;
            if (transcriptionContainer) {
                transcriptionContainer.insertBefore(statusElement, this.transcriptionBox);
            }
        }

        if (statusElement) {
            if (this.filteredCount > 0) {
                statusElement.textContent = `ℹ️ ${this.filteredCount} transcription(s) hidden (confidence < ${filterValue}%)`;
                statusElement.classList.add('active');
            } else {
                statusElement.classList.remove('active');
            }
        }
    }

    async initialize() {
        try {
            this.setStatus('loading', 'Loading WASM module...');
            this.initBtn.disabled = true;

            // Load WASM module
            this.Module = await WStreamModule();

            // Get selected model
            const modelPath = this.modelSelect.value;

            // Load model file
            const response = await fetch(modelPath);
            if (!response.ok) {
                throw new Error(`Failed to load model: ${response.statusText}`);
            }

            const modelData = await response.arrayBuffer();

            this.setStatus('loading', 'Initializing Whisper...');

            // Write model to filesystem
            const modelFileName = modelPath.split('/').pop();
            const tempPath = `/tmp/${modelFileName}`;

            try {
                this.Module.FS.mkdir('/tmp');
            } catch (e) {
                // Directory might already exist
            }

            const uint8Array = new Uint8Array(modelData);
            this.Module.FS.writeFile(tempPath, uint8Array);

            // Initialize whisper
            this.handle = this.Module.init(tempPath);

            if (!this.handle) {
                throw new Error('Failed to initialize Whisper');
            }

            console.log('[JS] Whisper initialized with handle:', this.handle);

            this.setStatus('ready', 'Ready');
            this.startBtn.disabled = false;

            // Start update interval
            this.startUpdateInterval();

        } catch (error) {
            console.error('[JS] Initialization error:', error);
            this.setStatus('error', `Error: ${error.message}`);
            this.initBtn.disabled = false;
        }
    }

    onStart() {
        if (!this.handle) {
            console.error('[JS] Whisper not initialized');
            return;
        }

        this.startRecording();
    }

    onStop() {
        this.stopRecording();
    }

    startRecording() {
        // Create or reuse audio context
        if (!this.context || this.context.state === 'closed') {
            this.context = new AudioContext({
                sampleRate: this.kSampleRate,
                channelCount: 1,
                echoCancellation: false,
                autoGainControl: true,
                noiseSuppression: true,
                latency: 'interactive'
            });
        }

        // Update status using handle
        if (this.handle) {
            this.Module.set_status(this.handle, "");
        }

        this.startBtn.disabled = true;
        this.stopBtn.disabled = false;
        this.initBtn.disabled = true;
        this.modelSelect.disabled = true;

        this.doRecording = true;
        this.startTime = Date.now();

        // Reset audio buffers for new recording session
        this.audio = null;
        this.audio0 = null;

        // Get user media
        navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: this.kSampleRate,
                sampleSize: 16,
                echoCancellation: false,
                autoGainControl: true,
                noiseSuppression: true,
                latency: 0
            }
        })
        .then((stream) => {
            this.currentStream = stream;
            this.setupMediaRecorder(stream);
        })
        .catch((err) => {
            console.error('[JS] Error getting audio stream:', err);
            this.setStatus('error', `Error: ${err.message}`);
            this.stopRecording();
        });
    }

    setupMediaRecorder(stream) {
        // Use incremental chunks instead of accumulating
        this.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        let audioChunks = [];

        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };

        // Process chunks when recording stops
        this.mediaRecorder.onstop = async () => {
            if (audioChunks.length > 0 && this.context && this.context.state !== 'closed') {
                const blob = new Blob(audioChunks, { type: 'audio/webm' });
                audioChunks = []; // Clear chunks immediately

                try {
                    await this.processAudioBlob(blob);
                } catch (error) {
                    console.error('[JS] Error processing audio blob:', error);
                }
            }

            // Restart recording if still active
            if (this.doRecording) {
                // Check if we need to restart due to time limit
                const elapsed = (Date.now() - this.startTime) / 1000;
                if (elapsed >= this.kRestartRecording_s) {
                    console.log('[JS] Restarting recording due to time limit');
                    // Save current audio before restart
                    this.audio0 = this.audio;
                    this.startTime = Date.now();
                }

                // Continue recording
                setTimeout(() => {
                    if (this.doRecording) {
                        this.setupMediaRecorder(this.currentStream);
                    }
                }, 10);
            }
        };

        // Start recording
        this.mediaRecorder.start();

        // Stop and restart at intervals
        this.recordingInterval = setTimeout(() => {
            if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop();
            }
        }, this.kIntervalAudio_ms);

        this.setStatus('recording', 'Recording...');
    }

    async processAudioBlob(blob) {
        // Check if context is valid
        if (!this.context || this.context.state === 'closed') {
            console.error('[JS] Audio context is not available');
            return;
        }

        try {
            const arrayBuffer = await blob.arrayBuffer();
            const audioBuffer = await this.context.decodeAudioData(arrayBuffer);

            // Get mono channel
            const channelData = audioBuffer.getChannelData(0);

            // Resample if needed
            const resampled = this.resampleAudio(channelData, audioBuffer.sampleRate, this.kSampleRate);

            // Manage audio buffer
            if (this.audio0 === null) {
                this.audio = resampled;
            } else {
                // Combine with previous audio
                const combined = new Float32Array(this.audio0.length + resampled.length);
                combined.set(this.audio0, 0);
                combined.set(resampled, this.audio0.length);
                this.audio = combined;

                // Limit total buffer size
                const maxSamples = this.kSampleRate * this.kRestartRecording_s;
                if (this.audio.length > maxSamples) {
                    // Keep only recent audio
                    this.audio = this.audio.slice(-maxSamples);
                }
            }

            // Send to processing
            await this.processAudioAsync(this.audio);

            // Update counter
            this.audioChunksProcessed++;

            // Log every 10 chunks
            if (this.audioChunksProcessed % 10 === 0) {
                console.log(`[JS] Processed ${this.audioChunksProcessed} chunks, ` +
                          `buffer size: ${this.audio.length} samples (${(this.audio.length/16000).toFixed(1)}s)`);
            }

        } catch (error) {
            console.error('[JS] Error in processAudioBlob:', error);
        }
    }

    resampleAudio(input, inputSampleRate, outputSampleRate) {
        if (inputSampleRate === outputSampleRate) {
            return new Float32Array(input);
        }

        const ratio = inputSampleRate / outputSampleRate;
        const outputLength = Math.floor(input.length / ratio);
        const output = new Float32Array(outputLength);

        for (let i = 0; i < outputLength; i++) {
            const inputIndex = i * ratio;
            const inputIndexFloor = Math.floor(inputIndex);
            const inputIndexCeil = Math.ceil(inputIndex);
            const fraction = inputIndex - inputIndexFloor;

            if (inputIndexCeil >= input.length) {
                output[i] = input[inputIndexFloor];
            } else {
                // Linear interpolation
                output[i] = input[inputIndexFloor] * (1 - fraction) +
                           input[inputIndexCeil] * fraction;
            }
        }

        return output;
    }

    async processAudioAsync(audioBuffer) {
        if (!this.handle || !audioBuffer || audioBuffer.length === 0) return;

        // Prevent overlapping processing
        if (this.isProcessing) {
            return;
        }

        this.isProcessing = true;

        try {
            // Push audio to worker thread
            const result = this.Module.set_audio(this.handle, audioBuffer);
            if (result !== 0) {
                console.error('[JS] Error pushing audio, result:', result);
            }

            this.lastAudioPush = Date.now();

        } catch (error) {
            console.error('[JS] Error pushing audio:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    startUpdateInterval() {
        this.updateInterval = setInterval(() => {
            if (!this.handle) return;

            // Poll for transcriptions
            const transcribed = this.Module.get_transcribed(this.handle);
            if (transcribed && transcribed.length > 0) {
                this.onTranscription(transcribed);
            }

            // Update status from C++ (less frequently)
            const status = this.Module.get_status(this.handle);
            if (status && status !== this.lastStatus) {
                this.lastStatus = status;
                // Update UI to show processing state
                if (status.includes('processing')) {
                    this.statusIndicator.classList.add('processing');
                } else {
                    this.statusIndicator.classList.remove('processing');
                }
            }
        }, this.kUpdateInterval_ms);
    }

    stopRecording() {
        if (this.handle) {
            this.Module.set_status(this.handle, "paused");
        }

        this.doRecording = false;

        // Clean up recording interval
        if (this.recordingInterval) {
            clearTimeout(this.recordingInterval);
            this.recordingInterval = null;
        }

        // Stop media recorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                this.mediaRecorder.stop();
            } catch (e) {
                console.error('[JS] Error stopping media recorder:', e);
            }
        }

        // Stop all tracks
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (e) {
                    console.error('[JS] Error stopping track:', e);
                }
            });
            this.currentStream = null;
        }

        // Reset audio buffers
        this.audio0 = null;
        this.audio = null;

        // Don't close context here - reuse it
        // Only close if explicitly cleaning up

        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.initBtn.disabled = false;
        this.modelSelect.disabled = false;

        this.setStatus('ready', 'Ready');
    }

    onTranscription(text) {
        const entry = document.createElement('div');
        entry.className = 'transcription-entry';

        // Create text element
        const textElement = document.createElement('div');
        textElement.className = 'transcription-text';
        textElement.textContent = text;
        entry.appendChild(textElement);

        // Default confidence value
        let confidence = 100;

        // Try to get confidence metrics
        if (this.handle && this.Module.get_confidence_metrics) {
            try {
                const metricsJson = this.Module.get_confidence_metrics(this.handle);
                const metrics = JSON.parse(metricsJson);

                // Store confidence value
                confidence = metrics.confidence || 0;

                // Handle special case where confidence is 0 (no speech or error)
                if (confidence === 0 || metrics.n_tokens === 0) {
                    // Skip empty transcriptions
                    return;
                }

                // Create metrics element
                const metricsElement = document.createElement('div');
                metricsElement.className = 'transcription-metrics';

                // Hide if preference is off
                if (!this.showMetrics) {
                    metricsElement.classList.add('hidden');
                }

                // Confidence with color coding
                const confClass = this.getConfidenceClass(confidence);

                // Format metrics display
                const logprobDisplay = metrics.avg_logprob !== undefined ? metrics.avg_logprob.toFixed(2) : 'N/A';
                const entropyDisplay = metrics.entropy !== undefined ? metrics.entropy.toFixed(2) : '0.00';

                metricsElement.innerHTML = `
                    <span class="metric ${confClass}"
                        title="Transcription confidence">
                        ${confidence.toFixed(0)}%
                    </span>
                    <span class="metric logprob" title="Average log probability">
                        logP: ${logprobDisplay}
                    </span>
                    <span class="metric entropy" title="Entropy (uncertainty)">
                        H: ${entropyDisplay}
                    </span>
                `;

                entry.appendChild(metricsElement);

                // Add visual indicator for confidence level
                entry.dataset.confidence = confidence;
                if (confidence < 50) {
                    entry.classList.add('low-confidence');
                } else if (confidence < 80) {
                    entry.classList.add('medium-confidence');
                } else {
                    entry.classList.add('high-confidence');
                }

            } catch (e) {
                console.error('[JS] Error parsing metrics:', e);
            }
        }

        // Apply filter
        const filterValue = this.confidenceFilterValue || 0;
        if (confidence < filterValue) {
            entry.classList.add('filtered-out');
        }

        // Add to DOM
        this.transcriptionBox.appendChild(entry);
        this.transcriptionData.push({
            element: entry,
            text: text,
            confidence: confidence,
            timestamp: Date.now()
        });

        // Scroll to bottom if not filtered
        if (confidence >= filterValue) {
            this.transcriptionBox.scrollTop = this.transcriptionBox.scrollHeight;
        }

        // Keep only last 10 entries
        while (this.transcriptionData.length > 10) {
            const removed = this.transcriptionData.shift();
            if (removed.element.parentNode) {
                removed.element.remove();
            }
        }

        // Recalculate filtered count
        this.filteredCount = this.transcriptionData.filter(data =>
            data.confidence < (this.confidenceFilterValue || 0)
        ).length;

        // Update filter status
        this.updateFilterStatus();
    }

    getConfidenceClass(confidence) {
        if (confidence >= 90) return 'confidence-excellent';
        if (confidence >= 75) return 'confidence-good';
        if (confidence >= 50) return 'confidence-fair';
        return 'confidence-poor';
    }

    clearTranscription() {
        this.transcriptionBox.innerHTML = '';
        this.transcriptionData = [];
        this.filteredCount = 0;

        // Remove filter status
        const statusElement = document.getElementById('filterStatus');
        if (statusElement) {
            statusElement.remove();
        }
    }

    setStatus(type, text) {
        this.statusText.textContent = text;
        this.statusIndicator.className = 'status-indicator';
        this.statusIndicator.classList.add(`status-${type}`);
    }

    addSimpleTranscription(text) {
        const entry = document.createElement('div');
        entry.className = 'transcription-entry';
        entry.textContent = text;
        this.transcriptionBox.appendChild(entry);
    }

    clearTranscription() {
        this.transcriptionBox.innerHTML = '';
        this.transcriptionData = [];
        this.filteredCount = 0;

        // Remove filter status
        const statusElement = document.getElementById('filterStatus');
        if (statusElement) {
            statusElement.remove();
        }
    }

    setStatus(type, text) {
        this.statusText.textContent = text;
        this.statusIndicator.className = 'status-indicator';
        this.statusIndicator.classList.add(`status-${type}`);
    }

    cleanup() {
        // Stop any ongoing recording
        if (this.doRecording) {
            this.stopRecording();
        }

        // Now close audio context
        if (this.context && this.context.state !== 'closed') {
            this.context.close();
            this.context = null;
        }

        // Clear update interval
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }

        // Free WASM resources
        if (this.handle) {
            this.Module.free_instance(this.handle);
            this.handle = null;
        }
    }
}

// Initialize when page loads
let asr;
window.addEventListener('load', () => {
    asr = new WStreamASR();
    asr.setStatus('ready', 'Ready to initialize');
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (asr) {
        asr.cleanup();
    }
});
