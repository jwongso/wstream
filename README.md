# wstream - Real-time Speech Transcription with Whisper

wstream is a high-performance, real-time speech transcription system built on top of Whisper.cpp. It provides WebSocket-based audio streaming and transcription capabilities, allowing both local microphone input and remote audio streaming from clients.

## Features
- Real-time Speech Recognition: Powered by OpenAI's Whisper model through whisper.cpp
- Multiple Audio Sources:
  - Local microphone capture via SDL2
  - Remote audio streaming through WebSocket clients
- Dynamic source switching at runtime
- WebSocket Server/Client Architecture:
- Server accepts multiple concurrent client connections
- Clients can stream audio and receive transcriptions
- JSON-based protocol with Base64 audio encoding
- Cross-Platform Support: Works on Linux, macOS, and Windows
- Low Latency: Optimized audio buffering and processing pipeline
- Flexible Deployment: Can run as a local transcription service or network-accessible server

## Architecture
```
┌─────────────────┐                    ┌──────────────────┐
│  wstream_client │◄──── WebSocket ───►│   wstream server │
│   (PortAudio)   │      (Port 8080)   │                  │
└─────────────────┘                    │  ┌────────────┐  │
                                       │  │  Whisper   │  │
┌─────────────────┐                    │  │   Engine   │  │
│  Other Clients  │◄──── WebSocket ───►│  └────────────┘  │
└─────────────────┘     (Port 8080)    │         ▲        │
                                       │         │        │
                                       │  ┌────────────┐  │
                                       │  │   Audio    │  │
                                       │  │  Processor │  │
                                       │  └────────────┘  │
                                       │         ▲        │
                                       │    ┌────┴────┐   │
                                       │    │   SDL   │   │
                                       │    │   Mic   │   │
                                       │    └─────────┘   │
                                       └──────────────────┘
```

## Building

### Prerequisites
- CMake 3.16 or higher
- C++20 compatible compiler
- Boost.Beast (WebSocket support)
- nlohmann/json
- SDL2 (for local microphone)
- PortAudio (for client microphone)

### Installing Dependencies
- macOS (Homebrew):
```bash
brew install cmake boost nlohmann-json sdl2 portaudio
```
- Linux (Ubuntu/Debian):
```bash
sudo apt-get install cmake libboost-all-dev nlohmann-json3-dev libsdl2-dev portaudio19-dev
```
- Linux (Fedora):
```bash
sudo dnf install cmake boost-devel json-devel SDL2-devel portaudio-devel
```

### Build Instructions
```bash
# Clone the repository
git clone https://github.com/yourusername/wstream.git
cd wstream
# Create build directory
mkdir build && cd build
# Configure
cmake ..
# Build
cmake --build . -j$(nproc)
# Binaries will be in build/bin/
```

## Usage

### Running the Server
```bash
# Start with default settings (SDL microphone, port 8080)
./build/bin/wstream
# Specify a Whisper model
./build/bin/wstream models/ggml-base.en.bin
# Start with WebSocket audio source
./build/bin/wstream --audio-source websocket
# Custom port
./build/bin/wstream --port 9090
# Help
./build/bin/wstream --help
```

### Running the Client
```bash
# Connect to local server
./build/bin/wstream_client
# Connect to remote server
./build/bin/wstream_client ws://192.168.1.100:8080
# Use specific audio device
./build/bin/wstream_client ws://localhost:8080 "MacBook Air Microphone"
# List available audio devices
./build/bin/wstream_client --list-devices
# Enable verbose mode
./build/bin/wstream_client --verbose ws://localhost:8080
# Use raw PCM instead of Base64 encoding
./build/bin/wstream_client --raw ws://localhost:8080
```

### Interactive Commands (Client)
Once the client is running, you can use these commands:
- start             - Start audio recording
- stop              - Stop audio recording
- status            - Get server status
- source <type>     - Set audio source (websocket/microphone)
- devices           - List available audio devices
- device <name>     - Select audio device by name
- verbose <on|off>  - Toggle verbose mode
- test audio        - Record 5 seconds of test audio to file
- help              - Show available commands
- quit              - Exit application

## Server Runtime Commands
When the server switches to WebSocket audio source, clients can control it:

### Start client and recording:
```bash
./build/bin/wstream_client ws://localhost:8080
> start
```
The server will automatically switch to WebSocket audio source when a client connects
Speak into your microphone - transcriptions appear on both client and server

## WebSocket Protocol
### Message Format
All WebSocket messages use JSON format.

### Audio Data (Client → Server)
```json
{
  "type": "audio",
  "audio": "base64_encoded_pcm_data",
  "encoding": "base64",
  "sample_rate": 16000,
  "channels": 1,
  "session_id": "client-session",
  "language": "en",
  "timestamp": 1234567890
}
```

### Transcription Result (Server → Client)
```json
{
  "type": "transcribe",
  "content": "Hello, this is the transcribed text",
  "session_id": "client-session",
  "confidence": 0.95,
  "timestamp": 1234567890
}
```

### Command Messages (Client → Server)
```json
{
  "type": "command",
  "action": "set_audio_source",
  "source": "websocket"
}
```

### Command Response (Server → Client)
```json
{
  "type": "response",
  "action": "set_audio_source",
  "status": "success",
  "message": "Audio source switched to websocket"
}
```

## Audio Specifications
Sample Rate: 16 kHz
Channels: Mono (1 channel)
Format: 16-bit signed PCM
Encoding: Base64 (default) or raw
Chunk Size: 1024 samples per packet (64ms)
Minimum for Transcription: 16000 samples (1 second)

## Troubleshooting
### No Transcription Output
Check audio levels: Enable verbose mode to see audio level information
Verify microphone permissions:
macOS: System Preferences → Security & Privacy → Microphone
Linux: Check PulseAudio/ALSA settings
Test audio recording:
`> test audio`
Then play the recorded file:
`ffplay -f s16le -ar 16000 -ac 1 test_audio.raw`

### Client Can't Connect
Check server is running: Look for "WebSocket server is running on port 8080"
Verify network connectivity: ping <server-ip>
Check firewall: Ensure port 8080 (or custom port) is open
Try local connection first: ws://localhost:8080

### Audio Device Issues
List available devices:
`./build/bin/wstream_client --list-devices`
Select specific device:
`> device "Your Device Name"`
Check device permissions: Ensure the application has microphone access
Server Crashes on Audio Source Switch

### This has been fixed, but if you encounter issues:
Ensure only one client is setting the audio source
Stop recording before switching sources
Check server logs for error messages
Building Issues
PortAudio not found on macOS:
```bash
brew install portaudio
export PKG_CONFIG_PATH="/opt/homebrew/opt/portaudio/lib/pkgconfig:$PKG_CONFIG_PATH"
```
Boost not found:
```bash
# Ubuntu/Debian
sudo apt-get install libboost-all-dev
# macOS
brew install boost
```

## Performance Tuning

### Server Configuration
- Model Selection: Smaller models (base, small) provide lower latency
- Audio Buffer Size: Adjust MIN_SAMPLES in websocket_audio_source.cpp for latency vs. accuracy trade-off
- Network: Use wired connection for best reliability

### Client Configuration
- Audio Device: Use default system device for best compatibility
- Buffer Size: Smaller buffers (512-1024 samples) for lower latency
- Encoding: Use --raw flag to skip Base64 encoding for slightly better performance

## Development

### Project Structure
```
wstream/
├── src/                    # Server source files
│   ├── whisper.cpp        # Whisper engine integration
│   └── ...
├── stream/                 # Streaming components
│   ├── wstream_app.cpp   # Main application logic
│   ├── audio_processor.cpp
│   ├── websocket_server.cpp
│   └── ...
├── client/                 # Client application
│   ├── main.cpp
│   ├── audio_recorder.cpp
│   ├── websocket_client.cpp
│   └── base64.cpp
├── common/                 # Shared utilities
└── models/                 # Whisper model files (not included)
```

### Adding New Audio Sources
Inherit from audio_source base class
Implement required virtual methods:
- initialize()
- start()
- stop()
- get_audio_samples()
- is_active()
Register in audio_source_factory

## License
MIT License

## Acknowledgments
Whisper.cpp by Georgi Gerganov
OpenAI Whisper for the original model
Boost.Beast for WebSocket implementation
PortAudio for cross-platform audio capture

## Contributing
Fork and PR
