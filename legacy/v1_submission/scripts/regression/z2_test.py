import json, numpy as np, warnings, os
warnings.filterwarnings('ignore')
from scipy.optimize import differential_evolution
import pandas as pd, yaml
import statsmodels.api as sm

RESULTS = os.path.expanduser('~/multi-agent-shogun/results')
emb = np.load(RESULTS + '/cpg_large_embeddings.npz', allow_pickle=True)
emb_sims = dict(zip(emb['ids'], emb['sims']))

shogun = pd.read_csv(RESULTS + '/cpg_all_items_for_embedding.csv', encoding='utf-8-sig')
yaml_lookup, trad_lookup = {}, {}
for fname in os.listdir(RESULTS):
    if fname.startswith('ashigaru') and fname.endswith('.yaml'):
        with open(RESULTS + '/' + fname) as f:
            y = yaml.safe_load(f)
        for item in y.get('items', []):
            iid = item.get('id', '')
            r, e, s = item.get('x_r'), item.get('x_e'), item.get('x_s')
            xt = item.get('x_type')
            if xt and isinstance(xt, list) and len(xt) == 3:
                r, e, s = xt
            if r is not None:
                yaml_lookup[iid] = (float(r), float(e), float(s))
            trad_lookup[iid] = item.get('tradition', 'unknown')

torami = pd.read_csv(RESULTS + '/torami_103_with_tradition.csv')

rows = []
for _, row in shogun.iterrows():
    iid = row['id']
    r, e, s = row.get('r'), row.get('e'), row.get('s')
    try: r = float(r)
    except: r = None
    try: e = float(e)
    except: e = None
    try: s = float(s)
    except: s = None
    if r is None and iid in yaml_lookup:
        r, e, s = yaml_lookup[iid]
    sim = emb_sims.get(iid)
    if r is None or sim is None:
        continue
    vecs = str(row.get('vector', ''))
    z = int(row['z_register']) if pd.notna(row.get('z_register')) else 2
    rows.append({'sim': sim, 'R': r, 'E': e, 'S': s, 'Z': z, 'Z2': z * z,
                 'P': 1 if 'Phonetic' in vecs else 0,
                 'F': 1 if 'Formal' in vecs else 0,
                 'trad': trad_lookup.get(iid, 'unknown')})

for _, row in torami.iterrows():
    iid = 'T_%03d' % row['id']
    sim = emb_sims.get(iid)
    r, e, s = row.get('x_r'), row.get('x_e'), row.get('x_s')
    try: r, e, s = float(r), float(e), float(s)
    except: continue
    if sim is None:
        continue
    z = int(row['z_level']) if pd.notna(row.get('z_level')) else 2
    rows.append({'sim': sim, 'R': r, 'E': e, 'S': s, 'Z': z, 'Z2': z * z,
                 'P': int(row.get('has_Phonetic', 0)),
                 'F': int(row.get('has_Formal', 0)),
                 'trad': row.get('tradition', 'unknown')})

df = pd.DataFrame(rows)
n = len(df)
Y = df['sim'].values
df = df.dropna(subset=["R","E","Z","Z2","sim"])
df = df[~df[["R","E","Z","Z2","sim"]].isin([np.inf,-np.inf]).any(axis=1)]
n = len(df)
Y = df["sim"].values
SS = np.sum((Y - np.mean(Y))**2)

print("=" * 65)
print("E = mc2 TEST: Z vs Z2 (N=%d)" % n)
print("=" * 65)

# OLS comparisons
for label, cols in [('Z', ['R','E','Z']), ('Z2', ['R','E','Z2']), ('Z+Z2', ['R','E','Z','Z2'])]:
    X = sm.add_constant(df[cols].astype(float))
    res = sm.OLS(Y, X).fit()
    print("\nsim ~ %s:  R2=%.4f  AdjR2=%.4f" % ('+'.join(cols), res.rsquared, res.rsquared_adj))
    for name in res.params.index:
        b = res.params[name]
        se = res.bse[name]
        t = res.tvalues[name]
        p = res.pvalues[name]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''
        print("  %-8s b=%8.4f se=%7.4f t=%6.2f p=%.4f %s" % (name, b, se, t, p, sig))

# Gate x Heaviside with tradition offset
R_ = df['R'].values.astype(float)
E_ = df['E'].values.astype(float)
S_ = df['S'].values.astype(float)
Z_ = df['Z'].values.astype(float)
Z2_ = df['Z2'].values.astype(float)
P_ = df['P'].values.astype(float)
F_ = df['F'].values.astype(float)

tm = df.groupby('trad')['sim'].transform('mean')
off = tm.values - Y.mean()
Yc = Y - off

def gc(p, zv):
    a0, a1, a2, a3, a4, a5, alpha, K = p
    gate = a0 + a1*R_ + a2*E_ + a3*zv + a4*P_ + a5*F_
    ch = np.where(S_ > K, 1.0 - alpha, 1.0)
    return gate * ch

bnds = [(-0.5, 1), (-0.5, 0.5), (-0.5, 0.5), (-0.1, 0.2),
        (-0.3, 0.3), (-0.3, 0.3), (0.01, 0.99), (0.01, 0.99)]

print("\n" + "=" * 65)
print("TRADITION + HEAVISIDE: Z vs Z2")
print("=" * 65)

results = {}
for label, zv in [('Z', Z_), ('Z2', Z2_)]:
    res = differential_evolution(
        lambda p: np.sum((Yc - gc(p, zv))**2),
        bnds, seed=42, maxiter=2000, tol=1e-12)
    Yp = gc(res.x, zv) + off
    R2 = 1 - np.sum((Y - Yp)**2) / SS
    adj = 1 - (1 - R2) * (n - 1) / (n - 9)
    results[label] = {'R2': R2, 'K': res.x[7], 'alpha': res.x[6], 'Z_coeff': res.x[3]}
    print("  Gate(%s) x Heaviside + tradition:" % label)
    print("    R2=%.4f  K=%.4f  alpha=%.4f  Z_coeff=%.4f" % (R2, res.x[7], res.x[6], res.x[3]))

# Z+Z2 combined in Gate
def gc_both(p, Z, Z2):
    a0, a1, a2, a3, a4, a5, a6, alpha, K = p
    gate = a0 + a1*R_ + a2*E_ + a3*Z + a6*Z2 + a4*P_ + a5*F_
    ch = np.where(S_ > K, 1.0 - alpha, 1.0)
    return gate * ch

bnds9 = [(-0.5, 1), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
         (-0.3, 0.3), (-0.3, 0.3), (-0.5, 0.5), (0.01, 0.99), (0.01, 0.99)]
res3 = differential_evolution(
    lambda p: np.sum((Yc - gc_both(p, Z_, Z2_))**2),
    bnds9, seed=42, maxiter=2000, tol=1e-12)
Yp3 = gc_both(res3.x, Z_, Z2_) + off
R2_3 = 1 - np.sum((Y - Yp3)**2) / SS
print("\n  Gate(Z+Z2) x Heaviside + tradition:")
print("    R2=%.4f  K=%.4f  Z_coeff=%.4f  Z2_coeff=%.4f" %
      (R2_3, res3.x[8], res3.x[3], res3.x[6]))

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)
print("  Gate(Z)    + tradition:  R2=%.4f" % results['Z']['R2'])
print("  Gate(Z2)   + tradition:  R2=%.4f  (delta=%+.4f)" %
      (results['Z2']['R2'], results['Z2']['R2'] - results['Z']['R2']))
print("  Gate(Z+Z2) + tradition:  R2=%.4f  (delta=%+.4f)" %
      (R2_3, R2_3 - results['Z']['R2']))

delta = results['Z2']['R2'] - results['Z']['R2']
print("\n" + "=" * 65)
if delta > 0.01:
    print("VERDICT: E = mc2 CONFIRMED! Z2 beats Z by %.4f" % delta)
elif delta > 0.005:
    print("VERDICT: E = mc2 SUGGESTIVE. Z2 slightly better by %.4f" % delta)
elif R2_3 > max(results['Z']['R2'], results['Z2']['R2']) + 0.005:
    print("VERDICT: BOTH Z and Z2 matter. Combined model best.")
else:
    print("VERDICT: Z and Z2 similar. Current Z resolution (1-3) may be too coarse.")
    print("  Z has only 3 levels. Z2 = {1,4,9}. Almost linear transform.")
    print("  Need finer velocity scale (0-10) to distinguish Z from Z2.")
print("=" * 65)
