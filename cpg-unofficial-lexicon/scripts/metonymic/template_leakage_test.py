#!/usr/bin/env python3
"""
Template-Level Leakage Test & Sensitivity Analysis
====================================================
Proposed by: Kurisu (theoretical design), Okabe (structural analysis)
Implemented by: Torami

Three experiments:
  1. Sensitivity Analysis: EN Sexual with contaminated controls removed
  2. Template Leakage Test: "Netflix and X" with 10 unrelated fillers
  3. Host-Word Template Leakage: "sleep _" and "come up for _" variants

Theoretical basis:
  Metonymic CPG paths are hosted at phrase level. When a phrase like
  "Netflix and chill" establishes a covert path, the template structure
  "Netflix and _" may retain residual target-pull regardless of filler.
  This is "template-level leakage" — distinct from HHP's word-level leakage.
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "text-embedding-3-large"
DIMENSIONS = 3072

OUTPUT_DIR = Path("metonymic_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Shared utilities
# -------------------------------------------------------------------
def get_embeddings(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=texts, model=MODEL, dimensions=DIMENSIONS)
    return [item.embedding for item in response.data]

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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

SEXUAL_TARGET_ANCHORS = [
    "sexual intercourse", "erotic", "intimacy", "lovemaking",
    "seduction", "arousal", "passionate", "desire"
]
SEXUAL_NEUTRAL_ANCHORS = [
    "furniture", "mathematics", "transportation", "geology",
    "accounting", "plumbing", "agriculture", "architecture"
]

# ===================================================================
# EXPERIMENT 1: Sensitivity Analysis
# EN Sexual with contaminated controls removed
# ===================================================================
def run_sensitivity_analysis():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Sensitivity Analysis")
    print("  EN Sexual — Contaminated controls removed")
    print("=" * 60)

    candidates = [
        ("Netflix and chill", "Netflix"),
        ("come up for coffee", "coffee"),
        ("spend the night", "night"),
        ("go back to my place", "place"),
        ("see my etchings", "etchings"),
        ("sleep together", "sleep"),
        ("hook up", "hook"),
        ("get lucky", "lucky"),
    ]

    # Original controls (v1) — marking contaminated ones
    # CONTAMINATED: come up for air, sleep soundly, Netflix and popcorn
    clean_controls = [
        ("spend the day", "day"),
        ("go back to my desk", "desk"),
        ("see my photos", "photos"),
        ("hook up the printer", "hook"),
        ("get lucky at poker", "lucky"),
    ]

    all_texts = []
    for phrase, head in candidates + clean_controls:
        all_texts.extend([phrase, head])
    all_texts.extend(SEXUAL_TARGET_ANCHORS + SEXUAL_NEUTRAL_ANCHORS)

    print(f"  Embedding {len(all_texts)} texts...")
    all_embs = get_embeddings(all_texts)

    idx = 0
    n_cand = len(candidates)
    n_ctrl = len(clean_controls)

    def parse_pairs(n):
        nonlocal idx
        phrase_embs, head_embs = [], []
        for _ in range(n):
            phrase_embs.append(np.array(all_embs[idx])); idx += 1
            head_embs.append(np.array(all_embs[idx])); idx += 1
        return phrase_embs, head_embs

    cand_p, cand_h = parse_pairs(n_cand)
    ctrl_p, ctrl_h = parse_pairs(n_ctrl)

    target_embs = [np.array(all_embs[idx + i]) for i in range(len(SEXUAL_TARGET_ANCHORS))]
    idx += len(SEXUAL_TARGET_ANCHORS)
    neutral_embs = [np.array(all_embs[idx + i]) for i in range(len(SEXUAL_NEUTRAL_ANCHORS))]
    idx += len(SEXUAL_NEUTRAL_ANCHORS)

    target_centroid = np.mean(target_embs, axis=0)
    neutral_centroid = np.mean(neutral_embs, axis=0)

    def compute_diff_pull(phrase_embs, head_embs):
        pulls = []
        for p, h in zip(phrase_embs, head_embs):
            diff = p - h
            pull = cos_sim(diff, target_centroid) - cos_sim(diff, neutral_centroid)
            pulls.append(pull)
        return pulls

    cand_pulls = compute_diff_pull(cand_p, cand_h)
    ctrl_pulls = compute_diff_pull(ctrl_p, ctrl_h)

    from scipy.stats import mannwhitneyu
    try:
        u, p = mannwhitneyu(cand_pulls, ctrl_pulls, alternative="greater")
    except ValueError:
        u, p = 0, 1.0

    g = hedges_g(cand_pulls, ctrl_pulls)

    print(f"\n  Candidates (n={n_cand}):")
    for i, (phrase, head) in enumerate(candidates):
        print(f"    {phrase:30s} | diff_pull={cand_pulls[i]:+.4f}")

    print(f"\n  Clean Controls (n={n_ctrl}):")
    for i, (phrase, head) in enumerate(clean_controls):
        print(f"    {phrase:30s} | diff_pull={ctrl_pulls[i]:+.4f}")

    print(f"\n  --- Results ---")
    print(f"  Candidate mean:   {np.mean(cand_pulls):+.4f}")
    print(f"  Clean Ctrl mean:  {np.mean(ctrl_pulls):+.4f}")
    print(f"  Difference:       {np.mean(cand_pulls) - np.mean(ctrl_pulls):+.4f}")
    print(f"  Mann-Whitney U:   {u:.1f}")
    print(f"  p-value:          {p:.4f}")
    print(f"  Hedges' g:        {g:.3f}")
    print(f"  Significant:      {'YES ✅' if p < 0.05 else 'NO ❌'}")

    print(f"\n  --- Comparison ---")
    print(f"  Original (all controls):   p = 0.1172, g = 0.691")
    print(f"  Sensitivity (clean only):  p = {p:.4f}, g = {g:.3f}")

    return {
        "experiment": "sensitivity_analysis",
        "candidate_mean": float(np.mean(cand_pulls)),
        "control_mean": float(np.mean(ctrl_pulls)),
        "difference": float(np.mean(cand_pulls) - np.mean(ctrl_pulls)),
        "mann_whitney_U": float(u),
        "p_value": float(p),
        "hedges_g": float(g),
        "significant": bool(p < 0.05),
        "n_candidates": n_cand,
        "n_controls": n_ctrl,
        "removed_controls": [
            "come up for air (metonymic: sexual breathlessness)",
            "sleep soundly (host-word contamination from 'sleep together')",
            "Netflix and popcorn (template contamination from 'Netflix and chill')",
        ],
    }


# ===================================================================
# EXPERIMENT 2: Template-Level Leakage
# "Netflix and X" with 10 unrelated fillers
# ===================================================================
def run_template_leakage():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Template-Level Leakage")
    print("  'Netflix and X' — 10 unrelated fillers")
    print("=" * 60)

    # The covert-path phrase
    source_phrase = "Netflix and chill"

    # Template fillers — deliberately unrelated to sex
    template_fillers = [
        ("Netflix and homework", "homework"),
        ("Netflix and laundry", "laundry"),
        ("Netflix and yoga", "yoga"),
        ("Netflix and taxes", "taxes"),
        ("Netflix and gardening", "gardening"),
        ("Netflix and cooking", "cooking"),
        ("Netflix and cleaning", "cleaning"),
        ("Netflix and studying", "studying"),
        ("Netflix and knitting", "knitting"),
        ("Netflix and ironing", "ironing"),
    ]

    # Baseline: completely unrelated phrases with same structure "X and Y"
    baseline_phrases = [
        ("bread and butter", "butter"),
        ("salt and pepper", "pepper"),
        ("pen and paper", "paper"),
        ("lock and key", "key"),
        ("needle and thread", "thread"),
        ("soap and water", "water"),
        ("hammer and nail", "nail"),
        ("cup and saucer", "saucer"),
        ("broom and dustpan", "dustpan"),
        ("fork and knife", "knife"),
    ]

    # Also embed Netflix alone and the source phrase
    all_texts = [source_phrase, "Netflix"]
    for phrase, _ in template_fillers:
        all_texts.append(phrase)
    for phrase, _ in baseline_phrases:
        all_texts.append(phrase)
    all_texts.extend(SEXUAL_TARGET_ANCHORS + SEXUAL_NEUTRAL_ANCHORS)

    print(f"  Embedding {len(all_texts)} texts...")
    all_embs = get_embeddings(all_texts)

    idx = 0
    source_emb = np.array(all_embs[idx]); idx += 1
    netflix_emb = np.array(all_embs[idx]); idx += 1

    template_embs = [np.array(all_embs[idx + i]) for i in range(len(template_fillers))]
    idx += len(template_fillers)
    baseline_embs = [np.array(all_embs[idx + i]) for i in range(len(baseline_phrases))]
    idx += len(baseline_phrases)

    target_embs = [np.array(all_embs[idx + i]) for i in range(len(SEXUAL_TARGET_ANCHORS))]
    idx += len(SEXUAL_TARGET_ANCHORS)
    neutral_embs = [np.array(all_embs[idx + i]) for i in range(len(SEXUAL_NEUTRAL_ANCHORS))]

    target_centroid = np.mean(target_embs, axis=0)
    neutral_centroid = np.mean(neutral_embs, axis=0)

    def pull_index(emb):
        return cos_sim(emb, target_centroid) - cos_sim(emb, neutral_centroid)

    source_pull = pull_index(source_emb)
    netflix_pull = pull_index(netflix_emb)
    template_pulls = [pull_index(e) for e in template_embs]
    baseline_pulls = [pull_index(e) for e in baseline_embs]

    print(f"\n  --- Source ---")
    print(f"    'Netflix and chill':  pull = {source_pull:+.4f}")
    print(f"    'Netflix' (alone):    pull = {netflix_pull:+.4f}")
    print(f"    Leakage (chill → template): {source_pull - netflix_pull:+.4f}")

    print(f"\n  --- 'Netflix and X' template fills ---")
    for i, (phrase, _) in enumerate(template_fillers):
        leakage = template_pulls[i] - netflix_pull
        print(f"    {phrase:35s} | pull={template_pulls[i]:+.4f} | leakage={leakage:+.4f}")

    print(f"\n  --- Baseline 'X and Y' (no Netflix) ---")
    for i, (phrase, _) in enumerate(baseline_phrases):
        print(f"    {phrase:35s} | pull={baseline_pulls[i]:+.4f}")

    from scipy.stats import mannwhitneyu
    try:
        u, p = mannwhitneyu(template_pulls, baseline_pulls, alternative="greater")
    except ValueError:
        u, p = 0, 1.0

    g = hedges_g(template_pulls, baseline_pulls)

    print(f"\n  --- Statistical Summary ---")
    print(f"  Template mean:    {np.mean(template_pulls):+.4f}")
    print(f"  Baseline mean:    {np.mean(baseline_pulls):+.4f}")
    print(f"  Difference:       {np.mean(template_pulls) - np.mean(baseline_pulls):+.4f}")
    print(f"  Mann-Whitney U:   {u:.1f}")
    print(f"  p-value:          {p:.4f}")
    print(f"  Hedges' g:        {g:.3f}")
    print(f"  Significant:      {'YES ✅' if p < 0.05 else 'NO ❌'}")
    print(f"\n  Netflix alone pull:   {netflix_pull:+.4f}")
    print(f"  Template mean pull:   {np.mean(template_pulls):+.4f}")
    print(f"  Mean template leakage (template - Netflix alone): {np.mean(template_pulls) - netflix_pull:+.4f}")

    return {
        "experiment": "template_leakage_netflix",
        "source_phrase": source_phrase,
        "source_pull": float(source_pull),
        "head_word_pull": float(netflix_pull),
        "template_mean": float(np.mean(template_pulls)),
        "baseline_mean": float(np.mean(baseline_pulls)),
        "mean_template_leakage": float(np.mean(template_pulls) - netflix_pull),
        "mann_whitney_U": float(u),
        "p_value": float(p),
        "hedges_g": float(g),
        "significant": bool(p < 0.05),
        "template_items": [
            {"phrase": template_fillers[i][0], "pull": float(template_pulls[i]),
             "leakage": float(template_pulls[i] - netflix_pull)}
            for i in range(len(template_fillers))
        ],
        "baseline_items": [
            {"phrase": baseline_phrases[i][0], "pull": float(baseline_pulls[i])}
            for i in range(len(baseline_phrases))
        ],
    }


# ===================================================================
# EXPERIMENT 3: Host-Word Template Leakage
# "sleep _" and "come up for _" variants
# ===================================================================
def run_host_word_leakage():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Host-Word Template Leakage")
    print("  'sleep _' and 'come up for _' variants")
    print("=" * 60)

    hosts = {
        "sleep": {
            "source": "sleep together",
            "templates": [
                "sleep early", "sleep well", "sleep late",
                "sleep alone", "sleep outside", "sleep deeply",
                "sleep lightly", "sleep poorly", "sleep forever",
                "sleep downstairs",
            ],
            "baselines": [
                "rest early", "rest well", "rest late",
                "rest alone", "rest outside", "rest deeply",
                "rest lightly", "rest poorly", "rest forever",
                "rest downstairs",
            ],
        },
        "come up for": {
            "source": "come up for coffee",
            "templates": [
                "come up for breakfast", "come up for dinner",
                "come up for water", "come up for tea",
                "come up for lunch", "come up for juice",
                "come up for milk", "come up for snacks",
                "come up for dessert", "come up for soup",
            ],
            "baselines": [
                "go out for breakfast", "go out for dinner",
                "go out for water", "go out for tea",
                "go out for lunch", "go out for juice",
                "go out for milk", "go out for snacks",
                "go out for dessert", "go out for soup",
            ],
        },
    }

    all_results = {}

    for host_name, config in hosts.items():
        print(f"\n  --- Host: '{host_name}' ---")
        print(f"  Source phrase: '{config['source']}'")

        all_texts = [config["source"], host_name]
        all_texts.extend(config["templates"])
        all_texts.extend(config["baselines"])
        all_texts.extend(SEXUAL_TARGET_ANCHORS + SEXUAL_NEUTRAL_ANCHORS)

        print(f"  Embedding {len(all_texts)} texts...")
        embs = get_embeddings(all_texts)

        idx = 0
        source_emb = np.array(embs[idx]); idx += 1
        host_emb = np.array(embs[idx]); idx += 1

        n_t = len(config["templates"])
        template_embs = [np.array(embs[idx + i]) for i in range(n_t)]; idx += n_t
        baseline_embs = [np.array(embs[idx + i]) for i in range(n_t)]; idx += n_t

        target_embs = [np.array(embs[idx + i]) for i in range(len(SEXUAL_TARGET_ANCHORS))]
        idx += len(SEXUAL_TARGET_ANCHORS)
        neutral_embs = [np.array(embs[idx + i]) for i in range(len(SEXUAL_NEUTRAL_ANCHORS))]

        target_centroid = np.mean(target_embs, axis=0)
        neutral_centroid = np.mean(neutral_embs, axis=0)

        def pull_index(emb):
            return cos_sim(emb, target_centroid) - cos_sim(emb, neutral_centroid)

        source_pull = pull_index(source_emb)
        host_pull = pull_index(host_emb)
        template_pulls = [pull_index(e) for e in template_embs]
        baseline_pulls = [pull_index(e) for e in baseline_embs]

        from scipy.stats import mannwhitneyu
        try:
            u, p = mannwhitneyu(template_pulls, baseline_pulls, alternative="greater")
        except ValueError:
            u, p = 0, 1.0
        g = hedges_g(template_pulls, baseline_pulls)

        print(f"  Source pull:     {source_pull:+.4f}")
        print(f"  Host-word pull:  {host_pull:+.4f}")

        print(f"\n  Template fills ('{host_name} _'):")
        for i, t in enumerate(config["templates"]):
            print(f"    {t:35s} | pull={template_pulls[i]:+.4f}")

        print(f"\n  Baseline fills:")
        for i, b in enumerate(config["baselines"]):
            print(f"    {b:35s} | pull={baseline_pulls[i]:+.4f}")

        print(f"\n  Template mean:  {np.mean(template_pulls):+.4f}")
        print(f"  Baseline mean:  {np.mean(baseline_pulls):+.4f}")
        print(f"  Difference:     {np.mean(template_pulls) - np.mean(baseline_pulls):+.4f}")
        print(f"  Mann-Whitney p: {p:.4f}")
        print(f"  Hedges' g:      {g:.3f}")
        print(f"  Significant:    {'YES ✅' if p < 0.05 else 'NO ❌'}")

        all_results[host_name] = {
            "source_phrase": config["source"],
            "source_pull": float(source_pull),
            "host_pull": float(host_pull),
            "template_mean": float(np.mean(template_pulls)),
            "baseline_mean": float(np.mean(baseline_pulls)),
            "difference": float(np.mean(template_pulls) - np.mean(baseline_pulls)),
            "mann_whitney_p": float(p),
            "hedges_g": float(g),
            "significant": bool(p < 0.05),
        }

    return all_results


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("  TEMPLATE-LEVEL LEAKAGE & SENSITIVITY ANALYSIS")
    print(f"  Model: {MODEL} ({DIMENSIONS}d)")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # Exp 1: Sensitivity
    results["sensitivity_analysis"] = run_sensitivity_analysis()

    # Exp 2: Netflix template leakage
    results["template_leakage_netflix"] = run_template_leakage()

    # Exp 3: Host-word template leakage
    results["host_word_leakage"] = run_host_word_leakage()

    # Save
    json_path = OUTPUT_DIR / "template_leakage_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  → All results saved: {json_path}")

    # Grand summary
    print("\n" + "=" * 60)
    print("  GRAND SUMMARY")
    print("=" * 60)

    sa = results["sensitivity_analysis"]
    print(f"  1. Sensitivity Analysis (EN Sexual, clean controls only)")
    print(f"     Original: p=0.1172, g=0.691")
    print(f"     Clean:    p={sa['p_value']:.4f}, g={sa['hedges_g']:.3f} {'✅' if sa['significant'] else '❌'}")

    tl = results["template_leakage_netflix"]
    print(f"\n  2. Netflix Template Leakage")
    print(f"     'Netflix and X' mean pull: {tl['template_mean']:+.4f}")
    print(f"     Baseline 'X and Y' mean:   {tl['baseline_mean']:+.4f}")
    print(f"     p={tl['p_value']:.4f}, g={tl['hedges_g']:.3f} {'✅' if tl['significant'] else '❌'}")

    hw = results["host_word_leakage"]
    for host_name, r in hw.items():
        print(f"\n  3. Host-Word Leakage: '{host_name}'")
        print(f"     Template mean: {r['template_mean']:+.4f}")
        print(f"     Baseline mean: {r['baseline_mean']:+.4f}")
        print(f"     p={r['mann_whitney_p']:.4f}, g={r['hedges_g']:.3f} {'✅' if r['significant'] else '❌'}")


if __name__ == "__main__":
    main()
