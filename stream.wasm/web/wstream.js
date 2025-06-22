class WStreamASR {
    constructor() {
        // Web audio context
        this.context = null;

        // Audio data
        this.audio = null;
        this.audio0 = null;

        // The stream instance
        this.instance = null;

        // Module
        this.Module = null;

        // Recording state
        this.mediaRecorder = null;
        this.doRecording = false;
        this.startTime = 0;
        this.isProcessing = false;

        // Constants (matching original)
        this.kSampleRate = 16000;
        this.kRestartRecording_s = 120;
        this.kIntervalAudio_ms = 5000; // 5 seconds

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
    }

    setupEventListeners() {
        this.initBtn.addEventListener('click', () => this.initialize());
        this.startBtn.addEventListener('click', () => this.onStart());
        this.stopBtn.addEventListener('click', () => this.onStop());
        this.clearBtn.addEventListener('click', () => this.clearTranscription());
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

            // Write to FS
            const modelFileName = modelPath.split('/').pop();
            const tempPath = `/tmp/${modelFileName}`;

            const uint8Array = new Uint8Array(modelData);
            this.Module.FS.writeFile(tempPath, uint8Array);

            // Initialize whisper
            this.instance = this.Module.init(tempPath);

            if (!this.instance) {
                throw new Error('Failed to initialize Whisper');
            }

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
        if (!this.instance) {
            console.error('[JS] Whisper not initialized');
            return;
        }

        this.startRecording();
    }

    onStop() {
        this.stopRecording();
    }

    startRecording() {
        if (!this.context) {
            this.context = new AudioContext({
                sampleRate: this.kSampleRate,
                channelCount: 1,
                echoCancellation: false,
                autoGainControl: true,
                noiseSuppression: true,
            });
        }

        this.Module.set_status("");

        this.startBtn.disabled = true;
        this.stopBtn.disabled = false;
        this.initBtn.disabled = true;
        this.modelSelect.disabled = true;

        this.doRecording = true;
        this.startTime = Date.now();

        const chunks = [];
        let stream = null;

        navigator.mediaDevices.getUserMedia({audio: true, video: false})
            .then((s) => {
                stream = s;
                this.mediaRecorder = new MediaRecorder(stream);

                this.mediaRecorder.ondataavailable = (e) => {
                    chunks.push(e.data);

                    const blob = new Blob(chunks, { 'type': 'audio/ogg; codecs=opus' });
                    const reader = new FileReader();

                    reader.onload = (event) => {
                        const buf = new Uint8Array(reader.result);

                        if (!this.context) {
                            return;
                        }

                        this.context.decodeAudioData(buf.buffer, (audioBuffer) => {
                            const offlineContext = new OfflineAudioContext(
                                audioBuffer.numberOfChannels,
                                audioBuffer.length,
                                audioBuffer.sampleRate
                            );
                            const source = offlineContext.createBufferSource();
                            source.buffer = audioBuffer;
                            source.connect(offlineContext.destination);
                            source.start(0);

                            offlineContext.startRendering().then((renderedBuffer) => {
                                this.audio = renderedBuffer.getChannelData(0);

                                // Create combined audio buffer
                                const audioAll = new Float32Array(
                                    this.audio0 == null ? this.audio.length : this.audio0.length + this.audio.length
                                );

                                if (this.audio0 != null) {
                                    audioAll.set(this.audio0, 0);
                                }
                                audioAll.set(this.audio, this.audio0 == null ? 0 : this.audio0.length);

                                // Process audio asynchronously
                                if (this.instance && !this.isProcessing) {
                                    this.processAudioAsync(audioAll);
                                }
                            });
                        }, (e) => {
                            this.audio = null;
                        });
                    }

                    reader.readAsArrayBuffer(blob);
                };

                this.mediaRecorder.onstop = (e) => {
                    if (this.doRecording) {
                        setTimeout(() => {
                            this.startRecording();
                        });
                    }
                };

                this.mediaRecorder.start(this.kIntervalAudio_ms);

                this.setStatus('recording', 'Recording...');
            })
            .catch((err) => {
                console.error('[JS] Error getting audio stream:', err);
                this.setStatus('error', `Error: ${err.message}`);
            });

        // Restart recording interval
        const interval = setInterval(() => {
            if (!this.doRecording) {
                clearInterval(interval);
                if (this.mediaRecorder) {
                    this.mediaRecorder.stop();
                }
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }

                this.startBtn.disabled = false;
                this.stopBtn.disabled = true;
                this.initBtn.disabled = false;
                this.modelSelect.disabled = false;

                this.mediaRecorder = null;
            }

            // Restart recording if audio is too long
            if (this.audio != null && this.audio.length > this.kSampleRate * this.kRestartRecording_s) {
                if (this.doRecording) {
                    clearInterval(interval);
                    this.audio0 = this.audio;
                    this.audio = null;

                    if (this.mediaRecorder) {
                        this.mediaRecorder.stop();
                    }
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                    }
                }
            }
        }, 100);
    }

    async processAudioAsync(audioBuffer) {
        // Set flag to prevent multiple processing
        this.isProcessing = true;

        try {
            // Yield to the browser to keep UI responsive
            await new Promise(resolve => setTimeout(resolve, 0));

            // Process audio
            this.Module.set_audio(this.instance, audioBuffer);

            // Yield again to let UI update
            await new Promise(resolve => setTimeout(resolve, 0));

        } catch (error) {
            console.error('[JS] Error processing audio:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    startUpdateInterval() {
        setInterval(() => {
            if (!this.instance) return;

            const transcribed = this.Module.get_transcribed();
            if (transcribed && transcribed.length > 0) {
                this.onTranscription(transcribed);
            }
        }, 100);
    }

    stopRecording() {
        this.Module.set_status("paused");
        this.doRecording = false;
        this.audio0 = null;
        this.audio = null;
        this.context = null;

        this.setStatus('ready', 'Ready');
    }

    onTranscription(text) {
        // Add transcription without timestamp
        const entry = document.createElement('div');
        entry.className = 'transcription-entry';
        entry.textContent = text;

        this.transcriptionBox.appendChild(entry);
        this.transcriptionBox.scrollTop = this.transcriptionBox.scrollHeight;

        // Keep only last 10 entries
        const entries = this.transcriptionBox.getElementsByClassName('transcription-entry');
        while (entries.length > 10) {
            entries[0].remove();
        }
    }

    clearTranscription() {
        this.transcriptionBox.innerHTML = '';
    }

    setStatus(type, text) {
        this.statusText.textContent = text;
        this.statusIndicator.className = 'status-indicator';
        this.statusIndicator.classList.add(`status-${type}`);
    }
}

// Initialize when page loads
let asr;
window.addEventListener('load', () => {
    asr = new WStreamASR();
    asr.setStatus('ready', 'Ready to initialize');
});
