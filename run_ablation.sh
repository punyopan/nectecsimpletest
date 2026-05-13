#!/bin/bash
# Run all 4 ablation studies sequentially
# Usage: bash run_ablation.sh

set -e

echo "============================================"
echo "  Ablation Study: S0 → S2 → S3 → S4"
echo "============================================"

echo ""
echo "[1/4] S0: AASIST Baseline (100 epochs)"
python main.py --config config/S0_baseline.conf --comment S0_baseline
echo "✅ S0 complete"

echo ""
echo "[2/4] S2: AASIST + WavLM-Large (20 epochs)"
python main.py --config config/S2_ssl.conf --comment S2_ssl
echo "✅ S2 complete"

echo ""
echo "[3/4] S3: S2 + AFSS (20 epochs)"
python main.py --config config/S3_afss.conf --comment S3_afss
echo "✅ S3 complete"

echo ""
echo "[4/4] S4: AASIST3 Full = S3 + Codec (20 epochs)"
python main.py --config config/S4_aasist3_full.conf --comment S4_full
echo "✅ S4 complete"

echo ""
echo "============================================"
echo "  All ablations complete!"
echo "  Results in: ./exp_result/"
echo "============================================"
