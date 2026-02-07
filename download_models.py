#!/usr/bin/env python3
import os
import urllib.request

MODELS_DIR = "models"
FACE_MODELS_DIR = "face_detection/models"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FACE_MODELS_DIR, exist_ok=True)

MODELS = {
    # Wav2Lip GAN
    f"{MODELS_DIR}/wav2lip_gan.pth": "https://iiitdgpel-my.sharepoint.com/personal/rudrabha_mukhopadhyay_iiit_ac_in/_layouts/15/download.aspx?share=EdjIaHqoI3dBg1d3C1pQoggBn_E6d3IV3x1ap1e2BkhGGw",
    
    # Face detection S3FD
    f"{FACE_MODELS_DIR}/s3fd.pth": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
}

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"[SKIP] {dest} esiste già")
        return
    
    print(f"[DOWNLOAD] {dest}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] {dest}")
    except Exception as e:
        print(f"[ERROR] {e}")
        print(f"Scarica manualmente da: {url} -> {dest}")

if __name__ == "__main__":
    for dest, url in MODELS.items():
        download_file(url, dest)
    
    print("\nNota: Se il download automatico fallisce, scarica manualmente:")
    print("1. wav2lip_gan.pth -> https://github.com/Rudrabha/Wav2Lip/releases")
    print("2. s3fd.pth -> https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth")
