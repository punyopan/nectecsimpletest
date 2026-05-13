"""
Discover ThaiSpoof dataset structure.
Run after downloading ThaiSpoof_main from Google Drive.

Auto-detects: folder structure, protocol format, file extensions,
and reports bonafide/spoof counts per split.

Usage:
    python discover_dataset.py --path /path/to/ThaiSpoof_main
"""
import argparse
import os
from pathlib import Path
from collections import Counter


def discover(root_path: str):
    root = Path(root_path)
    if not root.exists():
        print(f"❌ Path does not exist: {root}")
        return

    print(f"\n{'='*60}")
    print(f"  ThaiSpoof Dataset Discovery")
    print(f"  Root: {root}")
    print(f"{'='*60}\n")

    # ---- 1. Folder structure ----
    print("📁 Folder structure:")
    for item in sorted(root.rglob("*")):
        if item.is_dir():
            # Count files in this directory
            files = list(item.glob("*"))
            n_files = len([f for f in files if f.is_file()])
            n_dirs = len([f for f in files if f.is_dir()])
            rel = item.relative_to(root)
            indent = "  " * len(rel.parts)
            print(f"  {indent}📂 {item.name}/ "
                  f"({n_files} files, {n_dirs} subdirs)")

    # ---- 2. Audio files ----
    print("\n🎵 Audio files:")
    audio_exts = Counter()
    audio_files = []
    for ext in [".wav", ".flac", ".mp3", ".ogg", ".m4a"]:
        found = list(root.rglob(f"*{ext}"))
        if found:
            audio_exts[ext] = len(found)
            audio_files.extend(found)
    for ext, count in audio_exts.most_common():
        print(f"  {ext}: {count} files")
    if not audio_files:
        print("  ⚠️  No audio files found!")

    # Show sample filenames
    if audio_files:
        print("\n  Sample filenames:")
        for f in sorted(audio_files)[:5]:
            print(f"    {f.relative_to(root)}")

    # ---- 3. Protocol/metadata files ----
    print("\n📋 Protocol/metadata files:")
    meta_files = []
    for ext in [".txt", ".csv", ".tsv", ".json"]:
        found = list(root.rglob(f"*{ext}"))
        meta_files.extend(found)
    for f in sorted(meta_files):
        rel = f.relative_to(root)
        size = f.stat().st_size
        print(f"  {rel} ({size} bytes)")

    # ---- 4. Parse protocol files ----
    print("\n🔍 Protocol file analysis:")
    for f in sorted(meta_files):
        if f.suffix in [".txt", ".csv", ".tsv"]:
            _analyze_protocol(f, root)

    # ---- 5. Generate config snippet ----
    if audio_files and meta_files:
        _suggest_config(root, audio_exts, meta_files)


def _analyze_protocol(filepath, root):
    """Analyze a protocol file to detect format."""
    rel = filepath.relative_to(root)
    print(f"\n  --- {rel} ---")
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  ❌ Cannot read: {e}")
        return

    print(f"  Lines: {len(lines)}")
    if not lines:
        return

    # Show first 3 lines
    print("  First 3 lines:")
    for line in lines[:3]:
        print(f"    {line.rstrip()}")

    # Detect separator
    first_line = lines[0].strip()
    for sep_name, sep_char in [("tab", "\t"), ("comma", ","),
                                ("space", " "), ("pipe", "|")]:
        cols = first_line.split(sep_char)
        if len(cols) >= 3:
            print(f"  Separator: '{sep_name}' → {len(cols)} columns")
            break

    # Count labels
    label_counts = Counter()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check last column for bonafide/spoof
        parts = line.split()
        if parts:
            last = parts[-1].lower()
            if last in ["bonafide", "spoof", "genuine", "fake",
                         "real", "synthetic"]:
                label_counts[last] += 1
            # Also check common column positions
            for col in parts:
                col_lower = col.lower()
                if col_lower in ["bonafide", "spoof"]:
                    label_counts[col_lower] += 1
                    break

    if label_counts:
        print(f"  Labels found: {dict(label_counts)}")
    else:
        print("  Labels: could not auto-detect (check file manually)")


def _suggest_config(root, audio_exts, meta_files):
    """Suggest a config snippet for main.py"""
    primary_ext = audio_exts.most_common(1)[0][0] if audio_exts else ".wav"

    print(f"\n{'='*60}")
    print("  📝 Suggested Config Snippet")
    print(f"{'='*60}")
    print(f"""
To use ThaiSpoof with this training pipeline, you'll need to create
protocol files in ASVspoof format (5 columns, space-separated):

    SPEAKER_ID UTTERANCE_ID - ATTACK_TYPE LABEL

Where:
  SPEAKER_ID  = speaker identifier
  UTTERANCE_ID = filename without extension
  -           = placeholder
  ATTACK_TYPE = attack name or "-" for bonafide
  LABEL       = "bonafide" or "spoof"

Then set these in your config:
  "database_path": "{root}",
  "audio_ext": "{primary_ext}",
  "audio_subdir": "<subfolder containing audio>"

Or use the gen_universal_spoof_list() parser with fmt_config
to directly parse ThaiSpoof's native format.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover ThaiSpoof dataset structure")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to ThaiSpoof_main directory")
    args = parser.parse_args()
    discover(args.path)
