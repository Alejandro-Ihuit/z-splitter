import runpod
import os, subprocess, shutil, zipfile
import torch, torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

def split_audio(file_path):
    model = get_model('htdemucs')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav, sr = torchaudio.load(file_path)

    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(model, wav[None].to(device), device=device)[0]
    stem_names = model.sources

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join("/tmp", base_name)
    os.makedirs(out_dir, exist_ok=True)

    stems_to_keep = {"vocals", "bass", "other"}
    saved_files = []

    for stem, name in zip(sources, stem_names):
        if name in stems_to_keep:
            out_path = os.path.join(out_dir, f"{name}.wav")
            torchaudio.save(out_path, stem.cpu(), sr)
            saved_files.append(out_path)

    zip_path = f"{out_dir}.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in saved_files:
            zipf.write(f, os.path.basename(f))

    return zip_path

def handler(event):
    # Espera un JSON con {"audio_url": "..."}
    audio_url = event["input"]["audio_url"]
    local_path = "/tmp/input.wav"

    subprocess.run(["curl", "-L", audio_url, "-o", local_path], check=True)

    result_zip = split_audio(local_path)

    return {"zip_file": result_zip}

runpod.serverless.start({"handler": handler})
