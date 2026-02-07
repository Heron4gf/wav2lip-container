import os
import io
import base64
import tempfile
import cv2
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time
import subprocess

# Import inference modificato
from inference import load_model as load_wav2lip_model, face_detect_cached, preprocess_data

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/wav2lip_gan.pth"
FPS = 20
MEL_STEP_SIZE = 16

app = FastAPI(title="Wav2Lip Optimized API")

# Globali per modello pre-caricato
model = None

class LipSyncResponse(BaseModel):
    processing_time_ms: float
    frames_generated: int
    fps: int

@app.on_event("startup")
async def startup_event():
    global model
    print(f"[STARTUP] Caricamento modello su {DEVICE}...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modello non trovato: {MODEL_PATH}")
    
    model = load_wav2lip_model(MODEL_PATH, DEVICE)
    
    # Ottimizzazione: FP16 se su GPU
    if DEVICE == "cuda":
        model.half()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print(f"[STARTUP] Modello caricato. Precisione: {'FP16' if DEVICE == 'cuda' else 'FP32'}")

@app.post("/lipsync", response_class=StreamingResponse)
async def lipsync(
    image: UploadFile = File(..., description="Immagine statica (JPG/PNG)"),
    audio: UploadFile = File(..., description="File audio (WAV consigliato)"),
    fps: int = 20,
    pad: float = 0.0  # Padding per bbox faccia
):
    """
    Endpoint per lipsync su immagine statica.
    - image: Foto del viso
    - audio: Audio di input (qualunque durata, consigliato 10s)
    - fps: Frame rate output (default 20)
    """
    start_time = time.time()
    
    try:
        # 1. Leggi immagine
        img_bytes = await image.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(400, "Immagine non valida")
        
        # Resize a 256x256 (target risoluzione)
        frame = cv2.resize(frame, (256, 256))
        
        # 2. Leggi audio
        audio_bytes = await audio.read()
        
        # Salva temporaneamente per librosa (non gestisce bytes direttamente bene)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Carica audio
        wav, sr = sf.read(tmp_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # Mono
        
        # Resample a 16kHz se necessario
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # 3. Calcola numero frame necessari
        duration = len(wav) / sr
        num_frames = int(duration * fps)
        
        # 4. Face detection (una sola volta - cached)
        face_bbox = face_detect_cached(frame)
        if face_bbox is None:
            raise HTTPException(400, "Nessun volto rilevato nell'immagine")
        
        # 5. Prepara dati per batching
        mel_chunks = preprocess_data(wav, num_frames, MEL_STEP_SIZE)
        
        # 6. Genera batch di inferenza
        generated_frames = []
        batch_size = 128  # Ottimizzato per GPU memory vs speed
        
        with torch.no_grad():
            for i in range(0, len(mel_chunks), batch_size):
                batch_mels = mel_chunks[i:i+batch_size]
                batch_frames = [frame.copy() for _ in range(len(batch_mels))]
                
                # Prepara tensore
                img_batch = torch.FloatTensor(
                    np.array([preprocess_frame(f, face_bbox, pad) for f in batch_frames])
                ).to(DEVICE)
                mel_batch = torch.FloatTensor(np.array(batch_mels)).to(DEVICE)
                
                # FP16 su GPU
                if DEVICE == "cuda":
                    img_batch = img_batches.half()
                    mel_batch = mel_batch.half()
                
                # Inferenza
                pred = model(mel_batch, img_batch)
                
                # Post-process
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                pred = pred.astype(np.uint8)
                
                generated_frames.extend(pred)
        
        # 7. Ricomponi video
        # Crea video temporaneo
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid_tmp:
            vid_path = vid_tmp.name
        
        h, w = 256, 256
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))
        
        for gen_frame in generated_frames:
            # Incolla il viso generato sul frame originale
            final_frame = overlay_face(frame.copy(), gen_frame, face_bbox)
            out.write(final_frame)
        
        out.release()
        
        # 8. Mux audio con ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as final_tmp:
            final_path = final_tmp.name
        
        # ffmpeg rapido: copy video codec, aac audio
        cmd = [
            'ffmpeg', '-y', '-i', vid_path, '-i', tmp_path,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-shortest', final_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Pulizia temp
        os.unlink(tmp_path)
        os.unlink(vid_path)
        
        # 9. Risposta
        processing_time = (time.time() - start_time) * 1000
        
        def iterfile():
            with open(final_path, "rb") as f:
                yield from f
            # Cleanup dopo invio
            try:
                os.unlink(final_path)
            except:
                pass
        
        headers = {
            "X-Processing-Time-Ms": str(int(processing_time)),
            "X-Frames-Generated": str(len(generated_frames)),
            "X-FPS": str(fps)
        }
        
        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers=headers
        )
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(500, str(e))

def preprocess_frame(frame, bbox, pad):
    """Estrae e normalizza la regione del viso."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Aggiungi padding
    pad_h = int((y2-y1) * pad)
    pad_w = int((x2-x1) * pad)
    
    y1 = max(0, y1 - pad_h)
    y2 = min(h, y2 + pad_h)
    x1 = max(0, x1 - pad_w)
    x2 = min(w, x2 + pad_w)
    
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (96, 96))  # Wav2Lip input size
    
    # Normalizza
    face = face.astype(np.float32) / 255.
    face = np.transpose(face, (2, 0, 1))
    return face

def overlay_face(background, generated_face, bbox):
    """Incolla il viso generato sul background."""
    h, w = background.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Resize generated face a bbox size
    gen_resized = cv2.resize(generated_face, (x2-x1, y2-y1))
    
    # Smoothing (feathering) opzionale per bordi
    mask = np.ones((y2-y1, x2-x1), dtype=np.float32)
    # Gaussian blur sul bordo della maschera per blending morbido
    ksize = 15
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask = np.stack([mask] * 3, axis=2)
    
    roi = background[y1:y2, x1:x2].astype(np.float32)
    blended = (gen_resized.astype(np.float32) * mask + roi * (1 - mask)).astype(np.uint8)
    
    background[y1:y2, x1:x2] = blended
    return background

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
