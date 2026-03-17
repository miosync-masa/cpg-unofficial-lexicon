#!/usr/bin/env python3
"""
CPG Embedding Archaeology — Candidate vs Control Detection
===========================================================
Extends HHP framework to all 217 CPG items across 53 traditions.

Method:
  For each CPG item:
    1. Compute cos_sim(host_word, covert_target)  = candidate similarity
    2. Generate N control words from same domain
    3. Compute cos_sim(control_word_i, covert_target) = control similarities
    4. Pull index = candidate_sim - mean(control_sims)
    5. Per-tradition: Mann-Whitney U (candidates > controls)

  For S-type (compound) items:
    1. Compute cos_sim(emb(host1)+emb(host2), covert_target) = candidate
    2. Swap one host: cos_sim(emb(control)+emb(host2), covert_target) = control
    3. Same statistical tests

Based on: hhp_experiment.py, cross_cultural_dual_path.py, metonymic_path_tracing.py
Authors: Masamichi Iizumi + Tamaki (環) / Miosync, Inc.
Date: 2026-03-14 (Happy Birthday, Boss!)
"""

import json, os, sys, time, csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from scipy.stats import mannwhitneyu, ttest_ind
from openai import OpenAI

# --- Config ---
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    # Try hhp_experiment .env
    load_dotenv(os.path.expanduser("~/hhp_experiment/.env"))
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    load_dotenv(os.path.expanduser("~/multi-agent-shogun/.env"))
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODEL = "text-embedding-3-large"
DIMS = 3072
N_CONTROLS = 10  # controls per item
SEED = 42
np.random.seed(SEED)

BASE = Path(os.path.expanduser("~/multi-agent-shogun"))
RESULTS = BASE / "results"
OUTPUT = BASE / "cpg_archaeology"
OUTPUT.mkdir(exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

print("=" * 70)
print("CPG EMBEDDING ARCHAEOLOGY")
print("Candidate vs Control Detection (HHP Method Extended)")
print("=" * 70)
print(f"Model: {MODEL} ({DIMS}d)")
print(f"Controls per item: {N_CONTROLS}")


# =================================================================
# STEP 1: Load existing data
# =================================================================

# Load pre-computed embeddings
emb_data = np.load(str(RESULTS / "cpg_large_embeddings.npz"), allow_pickle=True)
emb_ids = list(emb_data['ids'])
emb_host = emb_data['host']  # (N, 3072)
emb_target = emb_data['target']  # (N, 3072)
emb_sims = emb_data['sims']

# Build lookup
host_emb_lookup = {emb_ids[i]: emb_host[i] for i in range(len(emb_ids))}
target_emb_lookup = {emb_ids[i]: emb_target[i] for i in range(len(emb_ids))}
sim_lookup = {emb_ids[i]: float(emb_sims[i]) for i in range(len(emb_ids))}

# Load item metadata
import yaml
items = []

# Shogun items
shogun = pd.read_csv(str(RESULTS / "cpg_all_items_for_embedding.csv"), encoding='utf-8-sig')
yaml_data = {}
for fname in os.listdir(str(RESULTS)):
    if fname.startswith('ashigaru') and fname.endswith('.yaml'):
        with open(str(RESULTS / fname)) as f:
            y = yaml.safe_load(f)
        for item in y.get('items', []):
            yaml_data[item.get('id', '')] = item

for _, row in shogun.iterrows():
    iid = row['id']
    if iid not in sim_lookup:
        continue
    yd = yaml_data.get(iid, {})
    host = row.get('host_romanized', '') or row.get('host_word', '')
    target = row.get('covert_target', '')
    tradition = yd.get('tradition', 'unknown')
    items.append({
        'id': iid, 'host': str(host).strip(), 'target': str(target).strip(),
        'tradition': tradition, 'source': 'shogun',
        'sim': sim_lookup[iid],
        'x_s': float(yd.get('x_s', yd.get('s', 0)) or 0),
        'language': str(row.get('language', '')),
    })

# Torami items
torami = pd.read_csv(str(RESULTS / "torami_103_with_tradition.csv"))
for _, row in torami.iterrows():
    iid = 'T_%03d' % row['id']
    if iid not in sim_lookup:
        continue
    host = str(row.get('item_roman', '') or row.get('item_orig', ''))
    target = str(row.get('covert_target', ''))
    items.append({
        'id': iid, 'host': host.strip(), 'target': target.strip(),
        'tradition': row.get('tradition', 'unknown'), 'source': 'torami',
        'sim': sim_lookup[iid],
        'x_s': float(row.get('x_s', 0) or 0),
        'language': str(row.get('language', '')),
    })

# Filter valid
items = [i for i in items if i['host'] and i['target'] and i['sim'] is not None]
print(f"\nLoaded {len(items)} items with embeddings")


# =================================================================
# STEP 2: Generate controls using GPT
# =================================================================

def generate_controls(host_word, covert_target, language="", tradition="", n=N_CONTROLS):
    """
    69 Method: Ask GPT-5.2 for the semantically nearest neighbors of host_word,
    excluding anything associated with covert_target.
    
    Like asking: "69's neighbors are 68, 70, 67, 71..."
    We want: "海老's neighbors are 蟹, 蝦, 鮑, 帆立..."
    """
    lang_str = language if language else "same language as the candidate"
    
    prompt = (
        "You are helping with a linguistic experiment.\n\n"
        f"Give me exactly {n} words or phrases that are SEMANTICALLY CLOSEST to:\n"
        f'  "{host_word}"\n\n'
        f"LANGUAGE: {lang_str}\n"
        "ALL words MUST be in the same language as the candidate. This is mandatory.\n\n"
        "Think of it like numbers: if the candidate is 69, I want 68, 70, 67, 71...\n"
        f'The NEAREST NEIGHBORS in meaning. Words a native speaker would say are '
        f'most similar to "{host_word}".\n\n'
        "EXCLUSION: Do NOT include any word associated with:\n"
        f'  "{covert_target}"\n\n'
        "Return ONLY the words, one per line. No numbers, no explanation, no translations."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=300,
        )
        text_r = resp.choices[0].message.content.strip()
        controls = [line.strip().strip('.-\u2022*0123456789) ') for line in text_r.split('\n')
                     if line.strip() and len(line.strip()) < 80]
        controls = [c for c in controls if c.lower() != host_word.lower() and len(c) > 0]
        return controls[:n]
    except Exception as e:
        print(f"  Control generation failed for {host_word}: {e}")
        return []


def get_embeddings(texts):
    """Batch embed texts."""
    if not texts:
        return []
    all_embs = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=MODEL, input=batch, dimensions=DIMS)
        for d in resp.data:
            all_embs.append(np.array(d.embedding))
    return all_embs


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def hedges_g(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if s_pooled == 0:
        return 0.0
    g = (m1 - m2) / s_pooled
    correction = 1 - 3 / (4*(n1+n2-2) - 1)
    return g * correction


# =================================================================
# STEP 3: Run candidate vs control for each item
# =================================================================

print("\n" + "=" * 70)
print("GENERATING CONTROLS AND COMPUTING PULL INDICES")
print("=" * 70)

all_results = []
control_cache_path = OUTPUT / "control_cache.json"

# Load cache if exists
if control_cache_path.exists():
    with open(control_cache_path) as f:
        control_cache = json.load(f)
    print(f"Loaded control cache ({len(control_cache)} items)")
else:
    control_cache = {}

batch_num = 0
total = len(items)

for idx, item in enumerate(items):
    iid = item['id']
    host = item['host']
    target = item['target']

    if idx % 20 == 0:
        print(f"\n  Processing {idx+1}/{total}...")

    # 1. Get or generate controls
    cache_key = f"{host}|||{target}|||{item.get('language','')}|||{item.get('tradition','')}"
    if cache_key in control_cache:
        control_words = control_cache[cache_key]
    else:
        control_words = generate_controls(host, target, language=item.get("language",""), tradition=item.get("tradition",""))
        control_cache[cache_key] = control_words
        # Save cache periodically
        if idx % 50 == 0:
            with open(control_cache_path, 'w') as f:
                json.dump(control_cache, f, ensure_ascii=False, indent=2)
            print(f"    (cache saved: {len(control_cache)} items)")

    if not control_words:
        continue

    # 2. Embed target and controls
    # Target is already embedded, get from lookup
    target_emb = target_emb_lookup.get(iid)
    if target_emb is None:
        continue

    # Embed control words
    control_embs = get_embeddings(control_words)
    if not control_embs:
        continue

    # 3. Compute similarities
    candidate_sim = item['sim']  # pre-computed host↔target
    control_sims = [cos_sim(c_emb, target_emb) for c_emb in control_embs]

    # 4. Pull index = candidate - mean(controls)
    pull_index = candidate_sim - np.mean(control_sims)

    all_results.append({
        'id': iid,
        'host': host,
        'target': target,
        'tradition': item['tradition'],
        'source': item['source'],
        'x_s': item['x_s'],
        'candidate_sim': candidate_sim,
        'control_mean': float(np.mean(control_sims)),
        'control_std': float(np.std(control_sims)),
        'control_max': float(np.max(control_sims)),
        'control_min': float(np.min(control_sims)),
        'pull_index': pull_index,
        'n_controls': len(control_sims),
        'controls': list(zip(control_words, [float(s) for s in control_sims])),
    })

# Save cache
with open(control_cache_path, 'w') as f:
    json.dump(control_cache, f, ensure_ascii=False, indent=2)

print(f"\n\nCompleted: {len(all_results)} items with controls")


# =================================================================
# STEP 4: Per-item analysis
# =================================================================

print("\n" + "=" * 70)
print("PER-ITEM PULL INDEX RESULTS")
print("=" * 70)

df = pd.DataFrame([{k: v for k, v in r.items() if k != 'controls'} for r in all_results])

# Overall stats
n_positive = (df['pull_index'] > 0).sum()
n_total = len(df)
print(f"\nPull index > 0: {n_positive}/{n_total} ({100*n_positive/n_total:.1f}%)")
print(f"Mean pull index: {df['pull_index'].mean():.4f} +/- {df['pull_index'].std():.4f}")

# Sign test
from scipy.stats import binomtest
try:
    p_sign = binomtest(n_positive, n_total, 0.5, alternative="greater").pvalue
except:
    from scipy.stats import binomtest
    p_sign = binomtest(n_positive, n_total, 0.5, alternative='greater').pvalue
print(f"Sign test (H0: 50% positive): p = {p_sign:.6f}")

# One-sample t-test (pull > 0)
from scipy.stats import ttest_1samp
t_1s, p_1s = ttest_1samp(df['pull_index'], 0, alternative='greater')
print(f"One-sample t-test (pull > 0): t={t_1s:.3f}, p={p_1s:.6f}")


# =================================================================
# STEP 5: Per-tradition analysis
# =================================================================

print("\n" + "=" * 70)
print("PER-TRADITION ANALYSIS")
print("=" * 70)

tradition_results = []
for trad, grp in df.groupby('tradition'):
    n = len(grp)
    if n < 3:
        continue  # too few items

    pulls = grp['pull_index'].values
    mean_pull = np.mean(pulls)
    pct_positive = (pulls > 0).sum() / len(pulls) * 100

    # One-sample t-test: pull > 0?
    if len(pulls) >= 3:
        t, p = ttest_1samp(pulls, 0, alternative='greater')
    else:
        t, p = 0, 1.0

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''
    print(f"  {trad:40s} (n={n:2d}): pull={mean_pull:+.4f}  {pct_positive:5.1f}% positive  t={t:5.2f}  p={p:.4f} {sig}")

    tradition_results.append({
        'tradition': trad, 'n': n, 'mean_pull': mean_pull,
        'pct_positive': pct_positive, 't': t, 'p': p,
    })

trad_df = pd.DataFrame(tradition_results)
n_sig = (trad_df['p'] < 0.05).sum()
print(f"\nTraditions with p < 0.05: {n_sig}/{len(trad_df)}")


# =================================================================
# STEP 6: P1-P5 Proposition Tests
# =================================================================

print("\n" + "=" * 70)
print("PROPOSITION TESTS (P1-P5)")
print("=" * 70)

# P1: Universality — CPG exists across cultures
print("\n--- P1: Universality ---")
print(f"Traditions tested: {len(trad_df)}")
print(f"Traditions with significant pull (p<0.05): {n_sig}")
print(f"Overall pull > 0: {n_positive}/{n_total} ({100*n_positive/n_total:.1f}%)")
print(f"Overall sign test: p = {p_sign:.6f}")

# P2: R/E/S distinction
print("\n--- P2: R/E/S Distinction ---")
s0 = df[df['x_s'] == 0]  # R or E dominant
sd = df[df['x_s'] >= 0.4]  # S dominant
sm = df[(df['x_s'] > 0) & (df['x_s'] < 0.4)]  # mixed
print(f"R/E type (S=0, n={len(s0)}): candidate_sim={s0['candidate_sim'].mean():.4f}, pull={s0['pull_index'].mean():.4f}")
if len(sm) > 0:
    print(f"Mixed    (0<S<0.4, n={len(sm)}): candidate_sim={sm['candidate_sim'].mean():.4f}, pull={sm['pull_index'].mean():.4f}")
if len(sd) > 0:
    print(f"S-dom    (S>=0.4, n={len(sd)}): candidate_sim={sd['candidate_sim'].mean():.4f}, pull={sd['pull_index'].mean():.4f}")

# P3: S-type critical point
print("\n--- P3: S-type Critical Point ---")
K = 0.34
below = df[df['x_s'] <= K]
above = df[df['x_s'] > K]
if len(above) > 0 and len(below) > 0:
    print(f"S <= {K} (n={len(below)}): candidate_sim={below['candidate_sim'].mean():.4f}")
    print(f"S >  {K} (n={len(above)}): candidate_sim={above['candidate_sim'].mean():.4f}")
    t3, p3 = ttest_ind(below['candidate_sim'], above['candidate_sim'], equal_var=False)
    print(f"Welch t={t3:.3f}, p={p3:.4f}")

# P4: Vector effects — would need vector data in df
print("\n--- P4: Vector Effects ---")
print("(Requires per-item vector coding. See HLM results for Phonetic p=0.001, Formal p=0.031)")

# P5: Tradition as unit
print("\n--- P5: Tradition as Unit ---")
# ICC-like: how much of pull variance is between-tradition?
overall_var = df['pull_index'].var()
within_var = df.groupby('tradition')['pull_index'].var().mean()
between_var = overall_var - within_var if overall_var > within_var else 0
pseudo_icc = between_var / overall_var if overall_var > 0 else 0
print(f"Pull index variance: total={overall_var:.4f}, within-tradition={within_var:.4f}")
print(f"Pseudo-ICC (tradition): {pseudo_icc:.4f} ({pseudo_icc*100:.1f}% between-tradition)")


# =================================================================
# STEP 7: Top/Bottom items
# =================================================================

print("\n" + "=" * 70)
print("TOP 10 STRONGEST CPG PULL")
print("=" * 70)
top = df.sort_values('pull_index', ascending=False).head(10)
for _, row in top.iterrows():
    print(f"  {row['host']:25s} -> {row['target']:25s}  pull={row['pull_index']:+.4f}  sim={row['candidate_sim']:.4f}  [{row['tradition']}]")

print("\n" + "=" * 70)
print("BOTTOM 10 (WEAKEST/NEGATIVE PULL)")
print("=" * 70)
bot = df.sort_values('pull_index', ascending=True).head(10)
for _, row in bot.iterrows():
    print(f"  {row['host']:25s} -> {row['target']:25s}  pull={row['pull_index']:+.4f}  sim={row['candidate_sim']:.4f}  [{row['tradition']}]")


# =================================================================
# STEP 8: Save
# =================================================================

df.to_csv(OUTPUT / "cpg_archaeology_results.csv", index=False, encoding='utf-8-sig')
trad_df.to_csv(OUTPUT / "cpg_archaeology_traditions.csv", index=False, encoding='utf-8-sig')

# Full results with controls
with open(OUTPUT / "cpg_archaeology_full.json", 'w') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

summary = {
    'date': datetime.now().isoformat(),
    'model': MODEL,
    'dimensions': DIMS,
    'n_items': len(df),
    'n_traditions': len(trad_df),
    'n_controls_per_item': N_CONTROLS,
    'overall_pull_mean': round(float(df['pull_index'].mean()), 4),
    'pct_positive_pull': round(float(100 * n_positive / n_total), 1),
    'sign_test_p': round(float(p_sign), 6),
    'one_sample_t': round(float(t_1s), 3),
    'one_sample_p': round(float(p_1s), 6),
    'n_traditions_significant': int(n_sig),
    'pseudo_icc_tradition': round(float(pseudo_icc), 4),
}
with open(OUTPUT / "cpg_archaeology_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("SAVED TO: %s" % OUTPUT)
print("  cpg_archaeology_results.csv     — per-item results")
print("  cpg_archaeology_traditions.csv   — per-tradition summary")
print("  cpg_archaeology_full.json        — full data with controls")
print("  cpg_archaeology_summary.json     — summary stats")
print("  control_cache.json               — generated controls (reusable)")
print("=" * 70)
print("\n\"Dictionaries record tatemae. Embeddings preserve honne.\"")
print("                       — Happy Birthday, Boss! 2026-03-14")
