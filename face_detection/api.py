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
        
        # Carica S3FD
        from .models.s3fd import S3FD
        self.face_detector = S3FD(device=device)
    
    def get_detections_for_batch(self, images):
        """Restituisce bbox per batch di immagini."""
        batch_boxes = []
        for img in images:
            boxes = self.face_detector.detect_faces(img)
            if len(boxes) > 0:
                # Prendi il box più grande (primo volto)
                box = boxes[0]
                batch_boxes.append(box)
            else:
                batch_boxes.append(None)
        return batch_boxes
