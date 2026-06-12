# 🎙️ Real-Time AI Voice Command Node

An end-to-end speech recognition and browser automation platform. The system streams raw uncompressed audio data straight from a client-side webpage dashboard into a backend Flask server, processing audio vectors through a custom PyTorch Convolutional Neural Network (CNN) for real-time phrase inference.

---

## 📌 Project Overview & "Why"

While traditional voice assistants rely on heavy, black-box cloud APIs that compress multimedia streams over slow network boundaries, this platform demonstrates how to construct a **lightweight, localized edge-inference pipeline**. 

By bypassing heavy C++ decoding wrappers (`torchcodec`, `pydub`, `soundfile`) that break across different cross-platform operating systems and Python runtimes, this project leverages the browser's native **Web Audio API**. It samples microphone amplitudes directly into uncompressed Float32 memory streams at exactly 16,000Hz, transmitting pure numerical vectors over an asynchronous network to a PyTorch CNN backend. 

This architecture allows for real-time automation control with minimal resource footprints.

---

## ⚙️ Architectural Workflow Blueprint

Data streams sequentially across the application stack without encountering un-optimized format encapsulation bottlenecks:
[Web Browser Mic Node] ──► Captured via JavaScript AudioContext at native 16,000Hz.
│
▼
[Raw Float32 Buffer]   ──► Transmitted over POST requests as application/octet-stream Blobs.
│
▼
[Flask Backend API]    ──► Parses numerical bytes directly into memory via NumPy arrays.
│
▼
[PyTorch Core Tensor]  ──► Re-shapes and pads array limits to a static 1-second boundary: (1, 1, 16000)
│
▼
[SpeechCNN Inference]  ──► Evaluates learned feature spaces to output structural logit arrays.
│
▼
[Browser OS Execution] ──► Returns command strings to trigger real-time actions (e.g., Opening Google tabs).


---

## 🧠 Deep Learning Model Architecture

The deep learning classifier uses a modular **SpeechCNN** block configured to read 1D waveform time-series sequences. 

### Processing Pipeline Inside the Model:
1. **Mathematical Sequence Normalization:** The input signal is scaled within static floating-point bounds ($-1.0$ to $+1.0$) to maintain weight stability.
2. **Feature Extraction Layout:** The model feeds 1-second audio segments into alternating layers of 1D Convolutions (`nn.Conv1d`), Batch Normalization (`nn.BatchNorm1d`), Non-Linear Activation Functions (`nn.ReLU`), and Max Pooling (`nn.MaxPool1d`) to automatically extract temporal sound features.
3. **Logit Mapping:** Dense Linear layers map hidden feature states directly to categorical command probabilities using cross-entropy criteria.

The pipeline natively interprets up to 35 discrete short commands trained via the Google Speech Commands dataset.

---

## 🛠️ Repository File Architecture

```text
├── speech_model.pth       # Saved PyTorch model checkpoint weights and class lookup labels
├── app.py                 # Clean Flask server orchestration (Direct raw buffer ingestion)
├── model.py               # Modular SpeechCNN PyTorch network architecture specs
├── README.md              # Professional repository portfolio layout and technical guide
└── templates/
    └── index.html         # Frontend voice interface using client-side Web Audio API context
🚀 Execution & Quickstart Guide
1. Requirements & Dependencies
Ensure your local Python virtual environment has the foundational packages installed. This setup is fully optimized to run on modern runtimes without requiring legacy audio encoding libraries:

Bash
pip install torch torchaudio numpy flask
2. Booting the Application Server
Run the Flask deployment script from your active terminal directory:

Bash
python app.py
3. Executing Automation Node
Open your web browser and navigate to http://127.0.0.8:5000/.

Click the Record Command button to open the audio stream pipeline.

Speak a core dataset command keyword clearly into your microphone (e.g., "Go").

The system will run real-time inference on the vector stream and trigger browser automation scripts, opening a new tab instantaneously!