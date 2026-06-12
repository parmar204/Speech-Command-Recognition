import os
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from model import SpeechCNN 

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize and Load Saved Model State ---
checkpoint_path = "speech_model.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Missing weight artifact file target at local path: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)
labels_list = checkpoint['labels_list']

model = SpeechCNN(num_classes=len(labels_list))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio data payload located.'}), 400
        
    file_storage_object = request.files['audio_data']
    audio_bytes_payload = file_storage_object.read()
    
    try:
        # 1. Instantly parse the browser bytes back into a raw Float32 numpy array
        raw_numpy_array = np.frombuffer(audio_bytes_payload, dtype=np.float32)
        
        # 2. Convert directly to a PyTorch tensor
        waveform = torch.from_numpy(raw_numpy_array).clone()
        waveform = waveform.unsqueeze(0)  # Shape becomes (1, Samples)
            
        # 3. Enforce 1-second structural sequence length constraints (16,000 samples)
        target_audio_samples = 16000
        if waveform.size(1) < target_audio_samples:
            padding_difference = target_audio_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding_difference))
        else:
            waveform = waveform[:, :target_audio_samples]
            
        # Add batch tracking index dimension -> shape becomes (1, 1, 16000)
        inference_ready_tensor = waveform.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction_logits = model(inference_ready_tensor)
            winning_class_index = prediction_logits.argmax(dim=1).item()
            
        final_word_command = labels_list[winning_class_index]
        return jsonify({'command': final_word_command})
        
    except Exception as runtime_processing_exception:
        return jsonify({'error': str(runtime_processing_exception)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)