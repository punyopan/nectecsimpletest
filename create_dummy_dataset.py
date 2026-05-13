"""
Create a tiny dummy ASVspoof2019-format dataset for smoke testing.
Works with both S0 (AASIST) and S2 (AASIST_SSL) configs.

Usage:
    python create_dummy_dataset.py [--num_samples 20] [--output_dir ./LA]
"""
import argparse
import os
import numpy as np
import soundfile as sf


def create_dummy_dataset(output_dir: str = "./LA", num_samples: int = 20):
    sr = 16000
    duration = 4.0
    n_samples = int(sr * duration)
    track = "LA"
    attack_types = ["A01", "A02", "A03", "A04", "A05", "A06"]

    splits = {
        "train": {
            "audio_dir": f"ASVspoof2019_{track}_train/flac",
            "protocol_name": f"ASVspoof2019.{track}.cm.train.trn.txt",
        },
        "dev": {
            "audio_dir": f"ASVspoof2019_{track}_dev/flac",
            "protocol_name": f"ASVspoof2019.{track}.cm.dev.trl.txt",
        },
        "eval": {
            "audio_dir": f"ASVspoof2019_{track}_eval/flac",
            "protocol_name": f"ASVspoof2019.{track}.cm.eval.trl.txt",
        },
    }

    protocol_dir = os.path.join(output_dir,
                                f"ASVspoof2019_{track}_cm_protocols")
    os.makedirs(protocol_dir, exist_ok=True)

    for split_name, split_info in splits.items():
        audio_dir = os.path.join(output_dir, split_info["audio_dir"])
        os.makedirs(audio_dir, exist_ok=True)

        protocol_path = os.path.join(protocol_dir,
                                     split_info["protocol_name"])
        lines = []

        for i in range(num_samples):
            utt_id = f"{track}_D_{split_name}_{i:04d}"
            if i < num_samples // 2:
                label = "bonafide"
                attack = "-"
            else:
                label = "spoof"
                attack = attack_types[i % len(attack_types)]

            audio = np.random.randn(n_samples).astype(np.float32) * 0.01
            flac_path = os.path.join(audio_dir, f"{utt_id}.flac")
            sf.write(flac_path, audio, sr)

            speaker_id = f"{track}_{i:04d}"
            line = f"{speaker_id} {utt_id} - {attack} {label}"
            lines.append(line)

        with open(protocol_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[{split_name}] Created {num_samples} dummy files in "
              f"{audio_dir}")

    # Create fake ASV scores file
    asv_score_dir = os.path.join(output_dir,
                                 f"ASVspoof2019_{track}_asv_scores")
    os.makedirs(asv_score_dir, exist_ok=True)
    asv_score_path = os.path.join(
        asv_score_dir,
        f"ASVspoof2019.{track}.asv.eval.gi.trl.scores.txt")

    asv_lines = []
    for i in range(num_samples):
        speaker_id = f"{track}_{i:04d}"
        if i < num_samples // 2:
            score = np.random.uniform(0.5, 1.0)
            asv_lines.append(f"{speaker_id} target {score:.4f}")
        else:
            score = np.random.uniform(-1.0, 0.0)
            asv_lines.append(f"{speaker_id} nontarget {score:.4f}")

    # Add spoof entries
    for i in range(num_samples // 2, num_samples):
        speaker_id = f"{track}_{i:04d}"
        score = np.random.uniform(-1.5, -0.5)
        asv_lines.append(f"{speaker_id} spoof {score:.4f}")

    with open(asv_score_path, "w") as f:
        f.write("\n".join(asv_lines) + "\n")
    print(f"[asv_scores] Created: {asv_score_path}")

    print(f"\n✅ Dummy dataset created at: {output_dir}")
    print(f"   Total files: {num_samples * 3} audio + 3 protocols + "
          f"1 ASV scores")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create dummy ASVspoof2019 dataset for smoke testing")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./LA")
    args = parser.parse_args()
    create_dummy_dataset(args.output_dir, args.num_samples)
