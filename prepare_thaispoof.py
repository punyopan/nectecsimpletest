import argparse
import os
import random
import subprocess
from pathlib import Path
from typing import List, Tuple

def extract_zip(zip_path: Path, extract_to: Path):
    """Extracts a zip file using the system unzip command."""
    print(f"📦 Extracting {zip_path.name}...")
    try:
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(extract_to)], check=True)
    except FileNotFoundError:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

def prepare_dataset(input_dir: str, output_dir: str, train_ratio: float = 0.8, 
                    dev_ratio: float = 0.1, mms_only: bool = False, mms_weight: int = 1):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    audio_dir = output_path / "audio"
    protocol_dir = output_path / "protocols"
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    zip_files = list(input_path.glob("*.zip"))
    if not zip_files:
        print(f"❌ No .zip files found in {input_path}")
        return

    print(f"Found {len(zip_files)} zip files. Starting extraction...")
    for zip_file in zip_files:
        if mms_only and "genuine" not in zip_file.name.lower() and "mms" not in zip_file.name.lower():
            # If mms_only is True, skip extracting any spoof zips that aren't MMS
            continue
        extract_zip(zip_file, audio_dir)

    print("\n🔍 Scanning extracted audio files...")
    all_files: List[Tuple[str, str, bool]] = []  # (rel_path, label, is_mms)
    
    for audio_file in audio_dir.rglob("*.*"):
        if audio_file.suffix.lower() not in [".wav", ".flac"]:
            continue
            
        lower_path = str(audio_file).lower()
        is_mms = "mms" in lower_path

        if "genuine" in lower_path or "bonafide" in lower_path:
            label = "bonafide"
        else:
            if mms_only and not is_mms:
                continue
            label = "spoof"
            
        rel_path = audio_file.relative_to(audio_dir)
        all_files.append((str(rel_path), label, is_mms))

    print(f"Found {len(all_files)} total audio files.")

    print("🔀 Shuffling and splitting dataset...")
    random.seed(42)
    random.shuffle(all_files)

    total = len(all_files)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    train_data = all_files[:train_end]
    dev_data = all_files[train_end:dev_end]
    eval_data = all_files[dev_end:]

    # Apply MMS prioritization strictly to the Training set to avoid data leakage
    train_data_final = []
    for path, label, is_mms in train_data:
        train_data_final.append((path, label))
        if label == "spoof" and is_mms and mms_weight > 1:
            for _ in range(mms_weight - 1):
                train_data_final.append((path, label))
                
    if mms_weight > 1:
        random.shuffle(train_data_final)
        print(f"📈 Upsampled MMS spoofs in train set by {mms_weight}x")

    dev_data_final = [(p, l) for p, l, _ in dev_data]
    eval_data_final = [(p, l) for p, l, _ in eval_data]

    print(f"Split sizes -> Train: {len(train_data_final)} | Dev: {len(dev_data_final)} | Eval: {len(eval_data_final)}")

    def write_protocol(data: List[Tuple[str, str]], filename: str):
        filepath = protocol_dir / filename
        with open(filepath, "w") as f:
            for rel_path, label in data:
                key = str(Path(rel_path).with_suffix(''))
                f.write(f"{key},{label}\n")
        print(f"✅ Generated protocol: {filepath}")

    write_protocol(train_data_final, "train_protocol.txt")
    write_protocol(dev_data_final, "dev_protocol.txt")
    write_protocol(eval_data_final, "eval_protocol.txt")
    
    print("\n🎉 Dataset preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ThaiSpoof dataset for AASIST training.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the downloaded .zip files")
    parser.add_argument("--output_dir", required=True, help="Directory to extract audio and save protocols")
    parser.add_argument("--mms_only", action="store_true", help="Only use MMS for spoofed audio, ignore TTS, pitchshift, etc.")
    parser.add_argument("--mms_weight", type=int, default=1, help="Multiply MMS samples in the training set by this factor to prioritize them")
    args = parser.parse_args()
    
    prepare_dataset(args.input_dir, args.output_dir, mms_only=args.mms_only, mms_weight=args.mms_weight)
