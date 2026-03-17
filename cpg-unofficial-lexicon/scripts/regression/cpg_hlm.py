import json, numpy as np, os, warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("statsmodels not found, installing...")
    import subprocess
    subprocess.check_call(['pip3', 'install', 'statsmodels', 'pandas'])
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

import pandas as pd

BASE = os.path.expanduser('~/multi-agent-shogun/results')

# Load full embeddings
with open(os.path.join(BASE, 'cpg_embeddings_full.json')) as f:
    data = json.load(f)

items = data['items']
valid = [i for i in items if i.get('r') is not None]

# Build DataFrame
rows = []
for v in valid:
    region = v['id'].split('_')[0]
    # Parse vectors into individual dummies
    vecs = [x.strip() for x in v.get('vector','').split('+') if x.strip() and x.strip() != 'Formal?']
    rows.append({
        'id': v['id'],
        'sim': v['embedding_similarity'],
        'R': v['r'],
        'E': v['e'],
        'S': v['s'],
        'Z': v['z_register'] if v.get('z_register') else 2,
        'region': region,
        'has_Phonetic': 1 if 'Phonetic' in vecs else 0,
        'has_Formal': 1 if 'Formal' in vecs else 0,
        'has_Metonymic': 1 if 'Metonymic' in vecs else 0,
        'has_Behavioral': 1 if 'Behavioral' in vecs else 0,
        'has_Structural': 1 if 'Structural' in vecs else 0,
        'has_Morphological': 1 if 'Morphological' in vecs else 0,
        'has_Chromatic': 1 if 'Chromatic' in vecs else 0,
    })

df = pd.DataFrame(rows)
print("DataFrame: %d rows, %d columns" % df.shape)
print("Regions:", df['region'].value_counts().to_dict())
print()

# ================================================================
# MODEL A: OLS (flat) — baseline for comparison
# ================================================================
X_ols = df[['R', 'E', 'Z']].copy()
X_ols = sm.add_constant(X_ols)
ols = sm.OLS(df['sim'], X_ols).fit()

print("=" * 65)
print("MODEL A: OLS (flat) — sim ~ R + E + Z")
print("=" * 65)
print("R2=%.4f  AdjR2=%.4f  AIC=%.1f" % (ols.rsquared, ols.rsquared_adj, ols.aic))
for name in ['const', 'R', 'E', 'Z']:
    b = ols.params[name]
    se = ols.bse[name]
    t = ols.tvalues[name]
    p = ols.pvalues[name]
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '.' if p<0.1 else ''
    print("  %-12s  b=%8.4f  se=%7.4f  t=%6.2f  p=%.4f %s" % (name, b, se, t, p, sig))

# ================================================================
# MODEL B: HLM — Random intercept for region
# sim ~ R + E + Z + (1 | region)
# ================================================================
print("\n" + "=" * 65)
print("MODEL B: HLM Random Intercept — sim ~ R + E + Z + (1|region)")
print("=" * 65)

X_hlm = df[['R', 'E', 'Z']].copy()
X_hlm = sm.add_constant(X_hlm)

hlm_b = MixedLM(df['sim'], X_hlm, groups=df['region']).fit(reml=True)
print(hlm_b.summary().tables[1])
print("\nRandom effects (region intercept variance):")
print(hlm_b.cov_re)
print("Log-likelihood: %.2f" % hlm_b.llf)
print("AIC: %.1f" % hlm_b.aic)

# ICC (Intraclass Correlation)
var_region = float(hlm_b.cov_re.iloc[0, 0])
var_resid = hlm_b.scale
icc = var_region / (var_region + var_resid)
print("ICC (region): %.4f" % icc)
print("  → %.1f%% of variance in similarity is between-region" % (icc * 100))

# ================================================================
# MODEL C: HLM with Z at Level 2 (region-level predictor)
# sim ~ R + E + (1|region), with Z as region-level mean
# ================================================================
print("\n" + "=" * 65)
print("MODEL C: HLM — R + E as L1, Z_mean as L2")
print("=" * 65)

# Compute region-level mean Z
z_means = df.groupby('region')['Z'].mean()
df['Z_region_mean'] = df['region'].map(z_means)
df['Z_centered'] = df['Z'] - df['Z_region_mean']

X_c = df[['R', 'E', 'Z_region_mean', 'Z_centered']].copy()
X_c = sm.add_constant(X_c)

hlm_c = MixedLM(df['sim'], X_c, groups=df['region']).fit(reml=True)
print(hlm_c.summary().tables[1])
print("\nRandom effects variance: %.6f" % float(hlm_c.cov_re.iloc[0, 0]))
print("AIC: %.1f" % hlm_c.aic)

# ================================================================
# MODEL D: HLM Full — R + E + Z + Vectors + (1|region)
# ================================================================
print("\n" + "=" * 65)
print("MODEL D: HLM Full — R + E + Z + Vectors + (1|region)")
print("=" * 65)

vec_cols = ['has_Phonetic', 'has_Formal', 'has_Metonymic', 'has_Behavioral',
            'has_Structural', 'has_Morphological']
X_d = df[['R', 'E', 'Z'] + vec_cols].copy()
X_d = sm.add_constant(X_d)

hlm_d = MixedLM(df['sim'], X_d, groups=df['region']).fit(reml=True)
print(hlm_d.summary().tables[1])
print("\nRandom effects variance: %.6f" % float(hlm_d.cov_re.iloc[0, 0]))
print("AIC: %.1f" % hlm_d.aic)

# ================================================================
# MODEL E: HLM with Random Slope for E (does E effect vary by region?)
# ================================================================
print("\n" + "=" * 65)
print("MODEL E: HLM Random Slope — sim ~ R + E + Z + (1 + E|region)")
print("=" * 65)

try:
    X_e = df[['R', 'E', 'Z']].copy()
    X_e = sm.add_constant(X_e)
    hlm_e = MixedLM(df['sim'], X_e, groups=df['region'],
                     exog_re=sm.add_constant(df[['E']])).fit(reml=True)
    print(hlm_e.summary().tables[1])
    print("\nRandom effects covariance:")
    print(hlm_e.cov_re)
    print("AIC: %.1f" % hlm_e.aic)
except Exception as ex:
    print("Random slope model failed: %s" % str(ex))
    print("(Expected — small group sizes may prevent convergence)")

# ================================================================
# S-COMPONENT IN HLM
# ================================================================
print("\n" + "=" * 65)
print("S-COMPONENT in HLM CONTEXT")
print("=" * 65)

X_s = df[['R', 'E', 'S', 'Z']].copy()
X_s = sm.add_constant(X_s)
hlm_s = MixedLM(df['sim'], X_s, groups=df['region']).fit(reml=True)

for name in ['const', 'R', 'E', 'S', 'Z']:
    b = hlm_s.params[name]
    se = hlm_s.bse[name]
    z = hlm_s.tvalues[name]
    p = hlm_s.pvalues[name]
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '.' if p<0.1 else ''
    print("  %-12s  b=%8.4f  se=%7.4f  z=%6.2f  p=%.4f %s" % (name, b, se, z, p, sig))

print("AIC: %.1f" % hlm_s.aic)

# ================================================================
# MODEL COMPARISON TABLE
# ================================================================
print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)
print("%-35s  %8s  %8s" % ("Model", "AIC", "LogLik"))
print("-" * 55)
print("%-35s  %8.1f  %8.2f" % ("A: OLS (flat)", ols.aic, ols.llf))
print("%-35s  %8.1f  %8.2f" % ("B: HLM random intercept", hlm_b.aic, hlm_b.llf))
print("%-35s  %8.1f  %8.2f" % ("C: HLM Z as L2", hlm_c.aic, hlm_c.llf))
print("%-35s  %8.1f  %8.2f" % ("D: HLM full + vectors", hlm_d.aic, hlm_d.llf))
print("%-35s  %8.1f  %8.2f" % ("S: HLM with S component", hlm_s.aic, hlm_s.llf))

# ================================================================
# REGION RANDOM EFFECTS (BLUPs)
# ================================================================
print("\n" + "=" * 65)
print("REGION RANDOM EFFECTS (BLUPs from Model B)")
print("=" * 65)
re = hlm_b.random_effects
for region in sorted(re.keys()):
    intercept = float(re[region].iloc[0])
    n = len(df[df['region'] == region])
    mean_sim = df[df['region'] == region]['sim'].mean()
    mean_z = df[df['region'] == region]['Z'].mean()
    print("%6s (n=%2d, Z_mean=%.1f): random_intercept=%+.4f  mean_sim=%.4f" %
          (region, n, mean_z, intercept, mean_sim))

# ================================================================
# SAVE
# ================================================================
results = {
    'models': {
        'A_OLS': {'R2': round(ols.rsquared, 4), 'AIC': round(ols.aic, 1)},
        'B_HLM_RI': {'AIC': round(hlm_b.aic, 1), 'ICC': round(icc, 4)},
        'C_HLM_Z_L2': {'AIC': round(hlm_c.aic, 1)},
        'D_HLM_Full': {'AIC': round(hlm_d.aic, 1)},
        'S_HLM_S': {'AIC': round(hlm_s.aic, 1)},
    },
    'ICC': round(icc, 4),
    'region_effects': {r: round(float(re[r].iloc[0]), 4) for r in re},
}
with open(os.path.join(BASE, 'cpg_hlm_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to cpg_hlm_results.json")
