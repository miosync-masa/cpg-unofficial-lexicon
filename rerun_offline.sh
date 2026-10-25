#!/bin/sh
# Rerun every offline step from the stored LLM codings and embedding vectors. No LLM or embedding API is called.
# Usage: ./rerun_offline.sh            (PYTHON=/path/to/python to use another interpreter)
set -e
cd "$(dirname "$0")"
P=${PYTHON:-python3}
# keep the current outputs before anything is overwritten
B=backups/results_$(date +%Y-%m-%d_%H%M%S)
mkdir -p backups && cp -R results "$B" && echo "backed up results/ to $B"

# v1 statistics recomputed from the stored per-item pulls
$P -I analysis/reproduce_stats.py > /dev/null
# first coding run, recording-status run: agreement and consensus
F=coding/results/first_run
$P coding/scripts/aggregate_codes.py $F/codes_gpt.csv $F/codes_gemini.csv $F/codes_claude.csv --out $F > /dev/null
R=coding/results/recording
$P coding/scripts/aggregate_recording.py $R/recording_gpt.csv $R/recording_gemini.csv $R/recording_claude.csv \
  --first $F/consensus_codes.csv --out $R > /dev/null
$P analysis/build_items_with_consensus.py > /dev/null
$P analysis/analyze_recording.py > /dev/null
# NNPM conditions A/B (bge-m3, Gemini, OpenAI): summary
E=embeddings/results
$P -I embeddings/scripts/analyze_replication.py $E/nnpm_BAAI_bge_m3.csv $E/nnpm_gemini_embedding_001.csv \
  $E/nnpm_text_embedding_3_large.csv --out $E/replication_summary.json > /dev/null 2>&1
# T3 exclusions, T1 clean hosts and controls, condition C from the saved vectors, T1 analysis
$P embeddings/scripts/check_hosts.py > /dev/null
$P embeddings/scripts/build_clean.py > /dev/null
for b in openai gemini bge-m3; do
  $P -I embeddings/scripts/nnpm_replicate.py data $E --backend $b --condition-c embeddings/controls --from-vectors > /dev/null
done
$P analysis/analyze_t1.py > /dev/null
# conditions D and E (controls without the covert target): D cleaned again from the logged responses, pulls from the
# saved vectors (the E control lists in data/controls_E_*.json need the embedding API to regenerate)
$P embeddings/scripts/generate_controls_D.py --offline > /dev/null
for b in openai gemini bge-m3; do
  $P embeddings/scripts/nnpm_target_blind.py --backend $b --from-vectors > /dev/null
done
$P analysis/analyze_target_blind.py > /dev/null
# T2 codebook v1.1 aggregation, T4 integrated analysis
V=coding/results/v1_1
$P coding/scripts/aggregate_v1_1.py $V/v1_1_gpt.csv $V/v1_1_gemini.csv $V/v1_1_claude.csv --out $V > /dev/null
$P analysis/analyze_final.py > /dev/null
# source audit: majority labels, kappa, spot-check list and sensitivity from the stored coder outputs
$P analysis/analyze_source_audit.py > /dev/null
echo "offline rerun done"
