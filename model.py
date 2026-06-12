import torch
import torch.nn as nn
import torchaudio

class SpeechCNN(nn.Module):
    def __init__(self, num_classes=35):
        super().__init__()

        # Audio feature extraction pipeline configuration
        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 40x101 -> 20x50
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 20x50 -> 10x25
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 10 * 25, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.mfcc_extractor(x)  # Raw Waveform to MFCC Image matrix
        x = self.features(x)        # Deep CNN Convolution tracking
        x = x.view(x.size(0), -1)    # Structural Flatten step
        x = self.classifier(x)      # Prediction vector scaling
        return x
    
