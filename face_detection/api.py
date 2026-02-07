import torch
import numpy as np
import cv2
from enum import Enum

class LandmarksType(Enum):
    _2D = 1
    _3D = 2

class FaceAlignment:
    def __init__(self, landmarks_type=LandmarksType._2D, flip_input=False, device='cuda'):
        self.device = device
        self.flip_input = flip_input
        # Usa Haar Cascade come fallback - funziona subito senza installazioni
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def get_detections_for_batch(self, images):
        """Restituisce bbox per batch di immagini."""
        batch_boxes = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            
            if len(faces) > 0:
                # Prendi il box più grande
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                batch_boxes.append((x, y, x + w, y + h))
            else:
                # Fallback: centro immagine
                h, w = img.shape[:2]
                size = min(h, w) // 3
                cx, cy = w // 2, h // 2
                batch_boxes.append((cx - size, cy - size, cx + size, cy + size))
        
        return batch_boxes
