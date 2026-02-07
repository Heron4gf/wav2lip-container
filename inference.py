"""
Inference ottimizzata per Wav2Lip
Basato su: https://github.com/Rudrabha/Wav2Lip
Modifiche: caching bbox, batching efficiente
"""

import cv2
import numpy as np
import torch
from scipy import signal
import librosa

# Cache globale per bbox
_cached_bbox = None

def load_model(checkpoint_path, device):
    from models import Wav2Lip
    
    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    
    model = model.to(device)
    model.eval()
    return model

def face_detect_cached(frame):
    """Face detection con cache semplice."""
    global _cached_bbox
    
    if _cached_bbox is not None:
        return _cached_bbox
    
    # Import locale per evitare circular
    import face_detection
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, 
        flip_input=False, 
        device='cuda'
    )
    
    image = cv2.resize(frame, (256, 256))
    bbox = detector.get_detections_for_batch(np.array([image]))[0]
    
    if bbox is None:
        # Fallback: usa centro immagine
        h, w = frame.shape[:2]
        size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        bbox = (cx - size, cy - size, cx + size, cy + size)
    
    _cached_bbox = bbox
    return bbox

def preprocess_data(wav, num_frames, mel_step_size=16):
    """Genera mel spectrogram chunks."""
    # Parametri mel
    hop_length = 200  # 12.5ms @ 16kHz
    n_fft = 800
    
    # Calcola mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=80,
        fmin=55,
        fmax=7600
    )
    
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalizza
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
    
    # Genera chunks
    mel_chunks = []
    for i in range(num_frames):
        start_idx = int(i * hop_length / 16000 * 16000 / hop_length)
        end_idx = start_idx + mel_step_size
        
        if end_idx > mel_spec.shape[1]:
            # Padding se necessario
            pad_width = end_idx - mel_spec.shape[1]
            chunk = np.pad(mel_spec[:, start_idx:], ((0, 0), (0, pad_width)), mode='edge')
        else:
            chunk = mel_spec[:, start_idx:end_idx]
        
        mel_chunks.append(chunk)
    
    return mel_chunks

def get_smoothened_boxes(boxes, T):
    """Smoothing temporale per i bbox (utile per video, per immagine statica non serve)."""
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes
