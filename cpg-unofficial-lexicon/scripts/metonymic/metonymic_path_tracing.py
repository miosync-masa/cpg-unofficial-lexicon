#!/usr/bin/env python3
"""
Metonymic CPG Path Tracing Experiment
======================================
Detection method: Kurisu's Method 3 (Phrasal Differential)

Theory:
  Metonymic covert paths are hosted at PHRASE level, not WORD level.
  If "Netflix and chill" carries a sexual covert path via metonymic
  enchantment, then:
    emb("Netflix and chill") - emb("Netflix") = differential vector
  This differential should point toward the sexual cluster MORE than
  control phrase differentials.

Design:
  - Candidate pairs: phrases with known Metonymic CPG + their head word
  - Control pairs: phrases WITHOUT Metonymic CPG + their head word
  - Target anchors: semantic cluster the covert path points toward
  - Measure: cos_sim(differential, target_centroid) for each pair
  - Test: Mann-Whitney U, candidate differentials > control differentials

Based on HHP Eq.3 (leakage analysis) extended to phrase-level hosts.

Authors: Masamichi Iizumi (theory), Torami (implementation), Kurisu (method design)
Date: 2026-03-13
Status: Pilot experiment — Miosync internal
"""

import os
import json
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "text-embedding-3-large"
DIMENSIONS = 3072

OUTPUT_DIR = Path("metonymic_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Experiment Groups
# -------------------------------------------------------------------

# === GROUP 1: English Sexual Metonymic ===
# Candidate: phrases with metonymic sexual covert path
# Control: similar phrases WITHOUT sexual covert path
# Target: sexual anchor cluster

GROUP1_EN_SEXUAL = {
    "name": "English Sexual Metonymic",
    "language": "en",
    "target_type": "sexual",
    "candidates": [
        # (phrase, head_word, expected_mechanism)
        ("Netflix and chill", "Netflix", "metonymic: activity sequence → sex"),
        ("come up for coffee", "coffee", "metonymic: invitation sequence → sex"),
        ("spend the night", "night", "metonymic: temporal frame → sex"),
        ("go back to my place", "place", "metonymic: location move → sex"),
        ("see my etchings", "etchings", "metonymic: classic invitation → sex"),
        ("sleep together", "sleep", "metonymic: bed-sharing → sex"),
        ("hook up", "hook", "metonymic: connection → sex"),
        ("get lucky", "lucky", "metonymic: fortune → sex"),
    ],
    "controls": [
        # Phrases with same head words or similar structure, NO sexual path
        # NOTE: v2 — removed metonymic-contaminated controls
        #   "come up for air" (sexual breathlessness connotation)
        #   "sleep soundly" (sleep carries sexual host from 'sleep together')
        #   "Netflix and popcorn" (Netflix itself partially contaminated by 'chill')
        ("watch a movie and relax", "movie", "literal: entertainment"),
        ("come up for breakfast", "breakfast", "literal: morning meal"),
        ("spend the day", "day", "literal: time use"),
        ("go back to my desk", "desk", "literal: return to work"),
        ("see my photos", "photos", "literal: viewing images"),
        ("rest peacefully", "rest", "literal: relaxation"),
        ("hook up the printer", "hook", "literal: connect device"),
        ("get lucky at poker", "lucky", "literal: gambling fortune"),
    ],
    "target_anchors": [
        "sexual intercourse", "erotic", "intimacy", "lovemaking",
        "seduction", "arousal", "passionate", "desire"
    ],
    "neutral_anchors": [
        "furniture", "mathematics", "transportation", "geology",
        "accounting", "plumbing", "agriculture", "architecture"
    ],
}

# === GROUP 2: Japanese Sexual Metonymic ===
GROUP2_JA_SEXUAL = {
    "name": "Japanese Sexual Metonymic",
    "language": "ja",
    "target_type": "sexual",
    "candidates": [
        ("床を共にする", "床", "metonymic: 寝具共有 → 性行為"),
        ("手を出す", "手", "metonymic: 身体接触の開始 → 性的接触"),
        ("一線を越える", "一線", "metonymic: 境界侵犯 → 性的関係"),
        ("体の関係", "体", "metonymic: 身体 → 性的関係"),
        ("深い関係", "深い", "metonymic: 関係の深度 → 性的関係"),
        ("大人の関係", "大人", "metonymic: 成熟 → 性的関係"),
        ("最後まで", "最後", "metonymic: 行為の終点 → 性行為完遂"),
        ("お泊まり", "泊まり", "metonymic: 宿泊 → 性行為"),
    ],
    "controls": [
        ("床に就く", "床", "literal: 就寝"),
        ("手を洗う", "手", "literal: 衛生行為"),
        ("一線を画す", "一線", "literal/metaphorical: 区別する"),
        ("体の調子", "体", "literal: 健康状態"),
        ("深い眠り", "深い", "literal: 睡眠の質"),
        ("大人の対応", "大人", "literal: 成熟した振る舞い"),
        ("最後まで読む", "最後", "literal: 読書完遂"),
        ("お泊まり保育", "泊まり", "literal: 保育行事"),
    ],
    "target_anchors": [
        "性行為", "エロティック", "性的興奮", "情事",
        "誘惑", "欲望", "肉体関係", "愛撫"
    ],
    "neutral_anchors": [
        "家具", "数学", "交通", "地質学",
        "会計", "配管", "農業", "建築"
    ],
}

# === GROUP 3: English Drinking Metonymic ===
GROUP3_EN_DRINKING = {
    "name": "English Drinking Metonymic",
    "language": "en",
    "target_type": "drinking/intoxication",
    "candidates": [
        ("grab a drink", "drink", "metonymic: single act → drinking session"),
        ("hit the bar", "bar", "metonymic: arrival → drinking"),
        ("crack open a cold one", "cold", "metonymic: opening act → drinking"),
        ("happy hour", "happy", "metonymic: time frame → drinking"),
        ("liquid lunch", "liquid", "metonymic: substance → drinking instead of eating"),
        ("nightcap", "night", "metonymic: end-of-day → final drink"),
    ],
    "controls": [
        ("grab a bite", "bite", "literal: eating"),
        ("hit the gym", "gym", "literal: exercise"),
        ("crack open a book", "book", "literal: reading"),
        ("happy ending", "happy", "literal/other: conclusion"),
        ("liquid assets", "liquid", "literal: finance"),
        ("night shift", "night", "literal: work schedule"),
    ],
    "target_anchors": [
        "alcohol", "drunk", "intoxication", "beer",
        "wine", "cocktail", "hangover", "inebriated"
    ],
    "neutral_anchors": [
        "furniture", "mathematics", "transportation", "geology",
        "accounting", "plumbing", "agriculture", "architecture"
    ],
}

# === GROUP 4: Japanese Drinking Metonymic ===
GROUP4_JA_DRINKING = {
    "name": "Japanese Drinking Metonymic",
    "language": "ja",
    "target_type": "drinking/intoxication",
    "candidates": [
        ("一杯やる", "一杯", "metonymic: 一杯で飲酒全体を指示"),
        ("飲みに行く", "飲み", "metonymic: 飲む行為 → 飲み会全体"),
        ("ちょっと付き合って", "付き合い", "metonymic: 同行要請 → 飲酒誘い"),
        ("二軒目行こう", "二軒目", "metonymic: 次の店 → 飲み続行"),
        ("もう一杯だけ", "一杯", "metonymic: 追加の一杯 → まだ終わらない"),
        ("乾杯しよう", "乾杯", "metonymic: 開始の儀礼 → 飲み会"),
    ],
    "controls": [
        ("一杯のコーヒー", "一杯", "literal: 一杯の飲料"),
        ("食べに行く", "食べ", "literal: 食事"),
        ("ちょっと手伝って", "手伝い", "literal: 手助け要請"),
        ("二階に行こう", "二階", "literal: 場所移動"),
        ("もう一回だけ", "一回", "literal: 繰り返し"),
        ("出発しよう", "出発", "literal: 行動開始"),
    ],
    "target_anchors": [
        "飲酒", "酔い", "アルコール", "ビール",
        "酒", "居酒屋", "二日酔い", "泥酔"
    ],
    "neutral_anchors": [
        "家具", "数学", "交通", "地質学",
        "会計", "配管", "農業", "建築"
    ],
}

# === GROUP 5: Ritual/Type E × Metonymic (Cross-cultural) ===
GROUP5_RITUAL = {
    "name": "Ritual Type-E Metonymic (Cross-cultural)",
    "language": "en",
    "target_type": "respect/reverence",
    "candidates": [
        # Bodily actions with enchanted meaning (Type E × Metonymic)
        ("bow deeply", "bow", "Type E metonymic: 身体前傾 → 敬意"),
        ("shake hands", "shake", "Type E metonymic: 手の接触 → 合意/信頼"),
        ("raise a glass", "glass", "Type E metonymic: グラス挙上 → 祝福"),
        ("tip your hat", "hat", "Type E metonymic: 帽子操作 → 敬意"),
        ("kneel before", "kneel", "Type E metonymic: 膝つき → 服従/敬意"),
        ("stand in silence", "silence", "Type E metonymic: 静止 → 追悼"),
        ("clasp hands in prayer", "prayer", "Type E metonymic: 合掌 → 祈り/敬意"),
        ("salute", "salute", "Type E metonymic: 挙手 → 敬意/忠誠"),
    ],
    "controls": [
        ("bend forward", "bend", "literal: physical movement"),
        ("shake vigorously", "shake", "literal: physical action"),
        ("raise a flag", "flag", "literal: lifting object"),
        ("tip the bucket", "bucket", "literal: tilting container"),
        ("kneel to tie shoes", "kneel", "literal: practical action"),
        ("stand in line", "line", "literal: queuing"),
        ("clasp a necklace", "necklace", "literal: fastening"),
        ("wave goodbye", "wave", "literal/conventional: farewell"),
    ],
    "target_anchors": [
        "respect", "reverence", "honor", "deference",
        "devotion", "homage", "veneration", "dignity"
    ],
    "neutral_anchors": [
        "furniture", "mathematics", "transportation", "geology",
        "accounting", "plumbing", "agriculture", "architecture"
    ],
}

ALL_GROUPS = [
    GROUP1_EN_SEXUAL,
    GROUP2_JA_SEXUAL,
    GROUP3_EN_DRINKING,
    GROUP4_JA_DRINKING,
    GROUP5_RITUAL,
]

# -------------------------------------------------------------------
# Embedding functions
# -------------------------------------------------------------------
def get_embeddings(texts: list[str], model: str = MODEL, dimensions: int = DIMENSIONS) -> list[list[float]]:
    """Get embeddings from OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        input=texts,
        model=model,
        dimensions=dimensions,
    )
    return [item.embedding for item in response.data]


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -------------------------------------------------------------------
# Core analysis: Method 3 (Phrasal Differential)
# -------------------------------------------------------------------
def run_phrasal_differential(group: dict) -> dict:
    """
    Kurisu's Method 3: Phrasal Differential Analysis

    For each (phrase, head_word) pair:
      1. Get emb(phrase) and emb(head_word)
      2. differential = emb(phrase) - emb(head_word)
      3. Compute cos_sim(differential, target_centroid)
      4. This measures how much the phrase structure
         activates the covert path beyond the head word alone.

    Also compute:
      - Direct phrasal pull (HHP-style, for comparison)
      - Head word pull (baseline)
    """
    print(f"\n{'='*60}")
    print(f"  {group['name']}")
    print(f"  Target: {group['target_type']}")
    print(f"{'='*60}")

    # --- Collect all texts to embed in one batch ---
    all_texts = []

    # Candidate phrases and head words
    cand_phrases = [c[0] for c in group["candidates"]]
    cand_heads = [c[1] for c in group["candidates"]]

    # Control phrases and head words
    ctrl_phrases = [c[0] for c in group["controls"]]
    ctrl_heads = [c[1] for c in group["controls"]]

    # Anchors
    target_anchors = group["target_anchors"]
    neutral_anchors = group["neutral_anchors"]

    all_texts = (
        cand_phrases + cand_heads +
        ctrl_phrases + ctrl_heads +
        target_anchors + neutral_anchors
    )

    # --- Get embeddings ---
    print(f"  Embedding {len(all_texts)} texts...")
    all_embs = get_embeddings(all_texts)

    # --- Parse back ---
    idx = 0
    n_cand = len(cand_phrases)
    n_ctrl = len(ctrl_phrases)
    n_target = len(target_anchors)
    n_neutral = len(neutral_anchors)

    cand_phrase_embs = [np.array(e) for e in all_embs[idx:idx+n_cand]]; idx += n_cand
    cand_head_embs = [np.array(e) for e in all_embs[idx:idx+n_cand]]; idx += n_cand
    ctrl_phrase_embs = [np.array(e) for e in all_embs[idx:idx+n_ctrl]]; idx += n_ctrl
    ctrl_head_embs = [np.array(e) for e in all_embs[idx:idx+n_ctrl]]; idx += n_ctrl
    target_embs = [np.array(e) for e in all_embs[idx:idx+n_target]]; idx += n_target
    neutral_embs = [np.array(e) for e in all_embs[idx:idx+n_neutral]]; idx += n_neutral

    # --- Compute centroids ---
    target_centroid = np.mean(target_embs, axis=0)
    neutral_centroid = np.mean(neutral_embs, axis=0)

    # --- Compute differentials and metrics ---
    def analyze_pairs(phrase_embs, head_embs, pair_info):
        results = []
        for i, (p_emb, h_emb) in enumerate(zip(phrase_embs, head_embs)):
            # Differential vector (Method 3)
            diff_vec = p_emb - h_emb

            # Differential pull: how much does phrase structure push toward target?
            diff_target_sim = cos_sim(diff_vec, target_centroid)
            diff_neutral_sim = cos_sim(diff_vec, neutral_centroid)
            diff_pull = diff_target_sim - diff_neutral_sim

            # Direct phrase pull (HHP-style, for comparison)
            phrase_target_sim = cos_sim(p_emb, target_centroid)
            phrase_neutral_sim = cos_sim(p_emb, neutral_centroid)
            phrase_pull = phrase_target_sim - phrase_neutral_sim

            # Head word pull (baseline)
            head_target_sim = cos_sim(h_emb, target_centroid)
            head_neutral_sim = cos_sim(h_emb, neutral_centroid)
            head_pull = head_target_sim - head_neutral_sim

            # Pull amplification: how much does phrase add over head?
            pull_amplification = phrase_pull - head_pull

            results.append({
                "phrase": pair_info[i][0],
                "head_word": pair_info[i][1],
                "mechanism": pair_info[i][2],
                "diff_pull": diff_pull,
                "phrase_pull": phrase_pull,
                "head_pull": head_pull,
                "pull_amplification": pull_amplification,
                "diff_target_sim": diff_target_sim,
                "diff_neutral_sim": diff_neutral_sim,
            })
        return results

    cand_results = analyze_pairs(cand_phrase_embs, cand_head_embs, group["candidates"])
    ctrl_results = analyze_pairs(ctrl_phrase_embs, ctrl_head_embs, group["controls"])

    # --- Statistical test ---
    from scipy.stats import mannwhitneyu

    cand_diff_pulls = [r["diff_pull"] for r in cand_results]
    ctrl_diff_pulls = [r["diff_pull"] for r in ctrl_results]

    cand_amplifications = [r["pull_amplification"] for r in cand_results]
    ctrl_amplifications = [r["pull_amplification"] for r in ctrl_results]

    # Test 1: Differential pull (Method 3 core)
    try:
        u_diff, p_diff = mannwhitneyu(cand_diff_pulls, ctrl_diff_pulls, alternative="greater")
    except ValueError:
        u_diff, p_diff = 0, 1.0

    # Test 2: Pull amplification (phrase_pull - head_pull)
    try:
        u_amp, p_amp = mannwhitneyu(cand_amplifications, ctrl_amplifications, alternative="greater")
    except ValueError:
        u_amp, p_amp = 0, 1.0

    # Hedges' g for both
    def hedges_g(group1, group2):
        n1, n2 = len(group1), len(group2)
        m1, m2 = np.mean(group1), np.mean(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        if s_pooled == 0:
            return 0.0
        g = (m1 - m2) / s_pooled
        correction = 1 - 3 / (4*(n1+n2-2) - 1)
        return g * correction

    g_diff = hedges_g(cand_diff_pulls, ctrl_diff_pulls)
    g_amp = hedges_g(cand_amplifications, ctrl_amplifications)

    # --- Print results ---
    print(f"\n  --- Candidate items (Metonymic CPG) ---")
    for r in cand_results:
        print(f"    {r['phrase']:30s} | diff_pull={r['diff_pull']:+.4f} | "
              f"phrase_pull={r['phrase_pull']:+.4f} | head_pull={r['head_pull']:+.4f} | "
              f"amplification={r['pull_amplification']:+.4f}")

    print(f"\n  --- Control items (No Metonymic CPG) ---")
    for r in ctrl_results:
        print(f"    {r['phrase']:30s} | diff_pull={r['diff_pull']:+.4f} | "
              f"phrase_pull={r['phrase_pull']:+.4f} | head_pull={r['head_pull']:+.4f} | "
              f"amplification={r['pull_amplification']:+.4f}")

    print(f"\n  --- Statistical Summary ---")
    print(f"  [Differential Pull (Method 3)]")
    print(f"    Candidate mean: {np.mean(cand_diff_pulls):+.4f}")
    print(f"    Control mean:   {np.mean(ctrl_diff_pulls):+.4f}")
    print(f"    Difference:     {np.mean(cand_diff_pulls) - np.mean(ctrl_diff_pulls):+.4f}")
    print(f"    Mann-Whitney U: {u_diff:.1f}, p = {p_diff:.4f}")
    print(f"    Hedges' g:      {g_diff:.3f}")
    print(f"    Significant:    {'YES ✅' if p_diff < 0.05 else 'NO ❌'}")

    print(f"\n  [Pull Amplification (phrase_pull - head_pull)]")
    print(f"    Candidate mean: {np.mean(cand_amplifications):+.4f}")
    print(f"    Control mean:   {np.mean(ctrl_amplifications):+.4f}")
    print(f"    Difference:     {np.mean(cand_amplifications) - np.mean(ctrl_amplifications):+.4f}")
    print(f"    Mann-Whitney U: {u_amp:.1f}, p = {p_amp:.4f}")
    print(f"    Hedges' g:      {g_amp:.3f}")
    print(f"    Significant:    {'YES ✅' if p_amp < 0.05 else 'NO ❌'}")

    # --- Compile output ---
    output = {
        "group_name": group["name"],
        "language": group["language"],
        "target_type": group["target_type"],
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "n_candidates": n_cand,
        "n_controls": n_ctrl,
        "timestamp": datetime.now().isoformat(),
        "differential_pull": {
            "candidate_mean": float(np.mean(cand_diff_pulls)),
            "control_mean": float(np.mean(ctrl_diff_pulls)),
            "difference": float(np.mean(cand_diff_pulls) - np.mean(ctrl_diff_pulls)),
            "mann_whitney_U": float(u_diff),
            "p_value": float(p_diff),
            "hedges_g": float(g_diff),
            "significant": p_diff < 0.05,
        },
        "pull_amplification": {
            "candidate_mean": float(np.mean(cand_amplifications)),
            "control_mean": float(np.mean(ctrl_amplifications)),
            "difference": float(np.mean(cand_amplifications) - np.mean(ctrl_amplifications)),
            "mann_whitney_U": float(u_amp),
            "p_value": float(p_amp),
            "hedges_g": float(g_amp),
            "significant": p_amp < 0.05,
        },
        "candidate_results": cand_results,
        "control_results": ctrl_results,
    }

    return output


# -------------------------------------------------------------------
# CSV export
# -------------------------------------------------------------------
def export_csv(output: dict, filepath: Path):
    """Export item-level results to CSV."""
    rows = []
    for r in output["candidate_results"]:
        rows.append({**r, "group": "candidate"})
    for r in output["control_results"]:
        rows.append({**r, "group": "control"})

    fieldnames = [
        "group", "phrase", "head_word", "mechanism",
        "diff_pull", "phrase_pull", "head_pull", "pull_amplification",
        "diff_target_sim", "diff_neutral_sim"
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  METONYMIC CPG PATH TRACING EXPERIMENT")
    print("  Method: Kurisu's Method 3 (Phrasal Differential)")
    print(f"  Model: {MODEL} ({DIMENSIONS}d)")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_outputs = []

    for group in ALL_GROUPS:
        output = run_phrasal_differential(group)
        all_outputs.append(output)

        # Export CSV
        safe_name = group["name"].lower().replace(" ", "_").replace("/", "_")
        csv_path = OUTPUT_DIR / f"{safe_name}.csv"
        export_csv(output, csv_path)
        print(f"  → CSV saved: {csv_path}")

    # --- Save combined JSON ---
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = OUTPUT_DIR / "metonymic_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\n  → Combined JSON saved: {json_path}")

    # --- Grand summary ---
    print("\n" + "=" * 60)
    print("  GRAND SUMMARY")
    print("=" * 60)
    print(f"  {'Group':<45s} | {'Diff Pull p':>11s} | {'g':>6s} | {'Amp p':>11s} | {'g':>6s} | Sig?")
    print(f"  {'-'*45}-+-{'-'*11}-+-{'-'*6}-+-{'-'*11}-+-{'-'*6}-+------")
    for o in all_outputs:
        dp = o["differential_pull"]
        pa = o["pull_amplification"]
        sig = "✅" if dp["significant"] or pa["significant"] else "❌"
        print(f"  {o['group_name']:<45s} | p={dp['p_value']:>8.4f} | {dp['hedges_g']:>+5.2f} | "
              f"p={pa['p_value']:>8.4f} | {pa['hedges_g']:>+5.2f} | {sig}")

    print(f"\n  Total groups: {len(all_outputs)}")
    n_sig_diff = sum(1 for o in all_outputs if o["differential_pull"]["significant"])
    n_sig_amp = sum(1 for o in all_outputs if o["pull_amplification"]["significant"])
    print(f"  Significant (diff pull): {n_sig_diff}/{len(all_outputs)}")
    print(f"  Significant (amplification): {n_sig_amp}/{len(all_outputs)}")
    print(f"\n  All results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
