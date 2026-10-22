import json, numpy as np, os, csv, warnings, yaml
warnings.filterwarnings('ignore')
from scipy.optimize import differential_evolution
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

BASE = os.path.expanduser('~/multi-agent-shogun')
RESULTS = os.path.join(BASE, 'results')

print("=" * 70)
print("CPG TRADITION-LEVEL HLM (T_T fix, N=217, 3072d LARGE)")
print("=" * 70)

# ================================================================
# STEP 1: Load torami CSV (103 items with tradition)
# ================================================================
torami = pd.read_csv(os.path.join(RESULTS, 'torami_103_with_tradition.csv'))
print("Torami: %d items, traditions: %s" % (len(torami), torami['tradition'].nunique()))
print("  ", torami['tradition'].value_counts().to_dict())

# ================================================================
# STEP 2: Load shogun YAML tradition names + embeddings
# ================================================================
# Load pre-computed large embeddings
emb_data = np.load(os.path.join(RESULTS, 'cpg_large_embeddings.npz'), allow_pickle=True)
emb_ids = list(emb_data['ids'])
emb_sims = dict(zip(emb_ids, emb_data['sims']))

# Build shogun tradition lookup from YAMLs
shogun_traditions = {}
for fname in os.listdir(RESULTS):
    if fname.startswith('ashigaru') and fname.endswith('.yaml'):
        with open(os.path.join(RESULTS, fname)) as f:
            y = yaml.safe_load(f)
        for item in y.get('items', []):
            iid = item.get('id', '')
            trad = item.get('tradition', '')
            shogun_traditions[iid] = trad

# Load shogun CSV
shogun = pd.read_csv(os.path.join(RESULTS, 'cpg_all_items_for_embedding.csv'), encoding='utf-8-sig')
print("Shogun: %d items" % len(shogun))

# Add tradition from YAML
shogun['tradition'] = shogun['id'].map(shogun_traditions).fillna('unknown')
# Add tradition_type based on region
def get_tradition_type(row):
    s = row.get('s', 0) or 0
    try:
        s = float(s)
    except:
        s = 0
    if s >= 0.4:
        return 'COMPOUND'
    elif s > 0:
        return 'MIXED'
    return 'UNITARY'

shogun['tradition_type'] = shogun.apply(get_tradition_type, axis=1)

# Fill r,e,s from YAMLs
yaml_lookup = {}
for fname in os.listdir(RESULTS):
    if fname.startswith('ashigaru') and fname.endswith('.yaml'):
        with open(os.path.join(RESULTS, fname)) as f:
            y = yaml.safe_load(f)
        for item in y.get('items', []):
            iid = item.get('id', '')
            r = item.get('x_r', item.get('r', None))
            e = item.get('x_e', item.get('e', None))
            s = item.get('x_s', item.get('s', None))
            xt = item.get('x_type', None)
            if xt and isinstance(xt, list) and len(xt) == 3:
                r, e, s = xt[0], xt[1], xt[2]
            if r is not None and e is not None and s is not None:
                yaml_lookup[iid] = (float(r), float(e), float(s))

for idx, row in shogun.iterrows():
    iid = row['id']
    r, e, s = row.get('r'), row.get('e'), row.get('s')
    try:
        r = float(r) if r and str(r).strip() else None
    except:
        r = None
    try:
        e = float(e) if e and str(e).strip() else None
    except:
        e = None
    try:
        s = float(s) if s and str(s).strip() else None
    except:
        s = None
    if (r is None or e is None or s is None) and iid in yaml_lookup:
        r, e, s = yaml_lookup[iid]
    shogun.at[idx, 'r'] = r
    shogun.at[idx, 'e'] = e
    shogun.at[idx, 's'] = s

# Add embedding similarity
shogun['sim'] = shogun['id'].map(emb_sims)
torami['sim'] = torami['id'].astype(str).apply(lambda x: emb_sims.get('T_%03d' % int(x), None) if x.isdigit() else emb_sims.get(x, None))

# ================================================================
# STEP 3: Unify columns and merge
# ================================================================
# Standardize column names
shogun_std = shogun.rename(columns={'r': 'x_r', 'e': 'x_e', 's': 'x_s', 'z_register': 'z_level'}).copy()
shogun_std['source'] = 'shogun'

# Parse vector dummies for shogun
for vec in ['Phonetic','Formal','Metonymic','Behavioral','Structural','Morphological','Chromatic']:
    col = 'has_' + vec
    if col not in shogun_std.columns:
        shogun_std[col] = shogun_std['vector'].fillna('').str.contains(vec).astype(int)

torami_std = torami.copy()
torami_std['source'] = 'torami'

# Common columns
common_cols = ['id', 'tradition', 'tradition_type', 'x_r', 'x_e', 'x_s',
               'has_Phonetic', 'has_Formal', 'has_Metonymic', 'has_Behavioral',
               'has_Structural', 'has_Morphological', 'has_Chromatic',
               'z_level', 'sim', 'source']

# Ensure columns exist
for col in common_cols:
    if col not in shogun_std.columns:
        shogun_std[col] = None
    if col not in torami_std.columns:
        torami_std[col] = None

# Ensure z_level is numeric
shogun_std['z_level'] = pd.to_numeric(shogun_std['z_level'], errors='coerce').fillna(2)
torami_std['z_level'] = pd.to_numeric(torami_std['z_level'], errors='coerce').fillna(2)

df = pd.concat([shogun_std[common_cols], torami_std[common_cols]], ignore_index=True)

# Convert numeric
for col in ['x_r', 'x_e', 'x_s', 'sim', 'z_level']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing critical data
df = df.dropna(subset=['x_r', 'x_e', 'x_s', 'sim'])
n = len(df)

Y = df['sim'].values
R = df['x_r'].values
E = df['x_e'].values
S = df['x_s'].values
Z = df['z_level'].values
P = df['has_Phonetic'].values.astype(float)
F = df['has_Formal'].values.astype(float)
SS_tot = np.sum((Y - np.mean(Y))**2)

print("\nMerged: %d items" % n)
print("Traditions: %d unique" % df['tradition'].nunique())
print(df['tradition'].value_counts().to_string())
print("\nTradition types:", df['tradition_type'].value_counts().to_dict())

# ================================================================
# MODEL 1: OLS
# ================================================================
X1 = np.column_stack([np.ones(n), R, E, Z])
b1 = np.linalg.lstsq(X1, Y, rcond=None)[0]
R2_ols = 1 - np.sum((Y - X1@b1)**2) / SS_tot
from scipy.stats import t as tdist
MSE1 = np.sum((Y - X1@b1)**2) / (n-4)
var1 = MSE1 * np.linalg.inv(X1.T @ X1)
se1 = np.sqrt(np.diag(var1))
t1 = b1 / se1

print("\n" + "=" * 70)
print("MODEL 1: OLS sim ~ R + E + Z  (N=%d)" % n)
print("=" * 70)
print("R2=%.4f  AdjR2=%.4f" % (R2_ols, 1-(1-R2_ols)*(n-1)/(n-4)))
for nm, b, se, t in zip(['const','R','E','Z'], b1, se1, t1):
    p = 2*(1-tdist.cdf(abs(t), n-4))
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '.' if p<0.1 else ''
    print("  %-10s b=%8.4f se=%7.4f t=%6.2f p=%.4f %s" % (nm, b, se, t, p, sig))

# ================================================================
# MODEL 2: HLM with tradition (THE KEY CHANGE!)
# ================================================================
X_hlm = df[['x_r','x_e','z_level']].copy()
X_hlm = sm.add_constant(X_hlm)
hlm = MixedLM(df['sim'], X_hlm, groups=df['tradition']).fit(reml=True)

var_re = float(hlm.cov_re.iloc[0,0])
icc = var_re / (var_re + hlm.scale)

print("\n" + "=" * 70)
print("MODEL 2: HLM + (1|tradition)  (N=%d, groups=%d)" % (n, df['tradition'].nunique()))
print("=" * 70)
print(hlm.summary().tables[1])
print("\nICC = %.4f (%.1f%% between-tradition)" % (icc, icc*100))
print("  (was 0.2095 with region grouping)")

# HLM Full with vectors
vec_cols = ['has_Phonetic','has_Formal','has_Metonymic','has_Behavioral','has_Structural','has_Morphological']
X_d = df[['x_r','x_e','z_level'] + vec_cols].astype(float).copy()
X_d = sm.add_constant(X_d)
hlm_d = MixedLM(df['sim'], X_d, groups=df['tradition']).fit(reml=True)

print("\n" + "=" * 70)
print("MODEL 2b: HLM Full + Vectors + (1|tradition)")
print("=" * 70)
print(hlm_d.summary().tables[1])

# ================================================================
# MODEL 3: Gate x Heaviside
# ================================================================
def gc_step(p, R, E, Z, S, P, F):
    a0,a1,a2,a3,a4,a5,alpha,K = p
    gate = a0 + a1*R + a2*E + a3*Z + a4*P + a5*F
    ch = np.where(S > K, 1.0 - alpha, 1.0)
    return gate * ch

def loss_step(p):
    return np.sum((Y - gc_step(p, R, E, Z, S, P, F))**2)

bounds_step = [(-0.5,1),(-0.5,0.5),(-0.5,0.5),(-0.1,0.2),(-0.3,0.3),(-0.3,0.3),(0.01,0.99),(0.01,0.99)]
res_h = differential_evolution(loss_step, bounds_step, seed=42, maxiter=2000, tol=1e-12)
ph = res_h.x
Yh = gc_step(ph, R, E, Z, S, P, F)
R2_h = 1 - np.sum((Y - Yh)**2) / SS_tot

print("\n" + "=" * 70)
print("MODEL 3: Gate x Heaviside (N=%d)" % n)
print("=" * 70)
print("R2=%.4f  (was 0.1100 with region)" % R2_h)
print("K_crit=%.4f  alpha=%.4f  (was K=0.3283)" % (ph[7], ph[6]))
print("  S < %.4f: Channel=1.000" % ph[7])
print("  S > %.4f: Channel=%.3f" % (ph[7], 1-ph[6]))

# ================================================================
# MODEL 4: Tradition-centered Heaviside
# ================================================================
trad_means = df.groupby('tradition')['sim'].mean()
grand_mean = Y.mean()
offsets = df['tradition'].map(trad_means).values - grand_mean
Yc = Y - offsets

def loss_step_c(p):
    return np.sum((Yc - gc_step(p, R, E, Z, S, P, F))**2)

res_rc = differential_evolution(loss_step_c, bounds_step, seed=42, maxiter=2000, tol=1e-12)
prc = res_rc.x
Yrc = gc_step(prc, R, E, Z, S, P, F) + offsets
R2_rc = 1 - np.sum((Y - Yrc)**2) / SS_tot

print("\n" + "=" * 70)
print("MODEL 4: Tradition-centered Heaviside (N=%d)" % n)
print("=" * 70)
print("R2=%.4f  (was 0.2546 with region)" % R2_rc)
print("K_crit=%.4f  alpha=%.4f" % (prc[7], prc[6]))

# ================================================================
# TRADITION-LEVEL RANDOM EFFECTS
# ================================================================
print("\n" + "=" * 70)
print("TRADITION RANDOM EFFECTS (BLUPs)")
print("=" * 70)
re = hlm.random_effects
for trad in sorted(re.keys()):
    intercept = float(re[trad].iloc[0])
    grp = df[df['tradition'] == trad]
    n_t = len(grp)
    mean_sim = grp['sim'].mean()
    mean_s = grp['x_s'].mean()
    ttype = grp['tradition_type'].mode().iloc[0] if len(grp['tradition_type'].mode()) > 0 else '?'
    print("  %-25s (n=%2d, S_mean=%.2f, %s): intercept=%+.4f  sim=%.4f" %
          (trad, n_t, mean_s, ttype, intercept, mean_sim))

# ================================================================
# TRADITION TYPE ANALYSIS (COMPOUND vs UNITARY)
# ================================================================
print("\n" + "=" * 70)
print("TRADITION TYPE: COMPOUND vs UNITARY")
print("=" * 70)
for ttype in ['COMPOUND', 'UNITARY', 'MIXED']:
    subset = df[df['tradition_type'] == ttype]
    if len(subset) > 0:
        print("  %s (n=%d): sim=%.4f +/- %.4f" %
              (ttype, len(subset), subset['sim'].mean(), subset['sim'].std()))

# Welch t-test COMPOUND vs UNITARY
comp = df[df['tradition_type']=='COMPOUND']['sim']
unit = df[df['tradition_type']=='UNITARY']['sim']
if len(comp) > 0 and len(unit) > 0:
    from scipy.stats import ttest_ind
    t, p = ttest_ind(comp, unit, equal_var=False)
    d = (comp.mean() - unit.mean()) / np.sqrt((comp.std()**2 + unit.std()**2) / 2)
    print("  COMPOUND vs UNITARY: t=%.3f  p=%.4f  d=%.3f" % (t, p, d))

# ================================================================
# S ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("S-COMPONENT (tradition-level HLM context)")
print("=" * 70)
s0 = df[df['x_s']==0]; snz = df[df['x_s']>0]; sd = df[df['x_s']>=0.4]
print("S=0    (n=%d): sim=%.4f" % (len(s0), s0['sim'].mean()))
print("S>0    (n=%d): sim=%.4f" % (len(snz), snz['sim'].mean()))
print("S>=0.4 (n=%d): sim=%.4f" % (len(sd), sd['sim'].mean()))

# ================================================================
# FINAL COMPARISON
# ================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON: region vs tradition grouping")
print("=" * 70)
print("%-45s  region    tradition" % "Metric")
print("-" * 70)
print("%-45s  %.4f    %.4f" % ("OLS R2", 0.0846, R2_ols))
print("%-45s  %.4f    %.4f" % ("ICC", 0.2095, icc))
print("%-45s  %.4f    %.4f" % ("Gate x Heaviside R2", 0.1100, R2_h))
print("%-45s  %.4f    %.4f" % ("Region/Tradition + Heaviside R2", 0.2546, R2_rc))
print("%-45s  %.4f    %.4f" % ("K_crit", 0.3283, ph[7]))
print("%-45s  %.4f    " % ("Formal p (HLM)", 0.038))

# Save
results = {
    'n': n, 'n_traditions': int(df['tradition'].nunique()),
    'ols_R2': round(R2_ols, 4),
    'icc_tradition': round(icc, 4),
    'icc_region_old': 0.2095,
    'heaviside_R2': round(R2_h, 4),
    'tradition_heaviside_R2': round(R2_rc, 4),
    'K_crit': round(float(ph[7]), 4),
    'alpha': round(float(ph[6]), 4),
}
with open(os.path.join(RESULTS, 'cpg_tradition_hlm_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to cpg_tradition_hlm_results.json")
