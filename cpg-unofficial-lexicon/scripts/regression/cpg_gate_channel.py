import json, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize, curve_fit
import pandas as pd

BASE = os.path.expanduser('~/multi-agent-shogun/results')

with open(os.path.join(BASE, 'cpg_embeddings_full.json')) as f:
    data = json.load(f)

valid = [i for i in data['items'] if i.get('r') is not None]

rows = []
for v in valid:
    region = v['id'].split('_')[0]
    vecs = [x.strip() for x in v.get('vector','').split('+') if x.strip() and x.strip() != 'Formal?']
    rows.append({
        'id': v['id'], 'sim': v['embedding_similarity'],
        'R': v['r'], 'E': v['e'], 'S': v['s'],
        'Z': v['z_register'] if v.get('z_register') else 2,
        'region': region,
        'has_Phonetic': 1 if 'Phonetic' in vecs else 0,
        'has_Formal': 1 if 'Formal' in vecs else 0,
    })

df = pd.DataFrame(rows)
Y = df['sim'].values
R = df['R'].values
E = df['E'].values
S = df['S'].values
Z = df['Z'].values
P = df['has_Phonetic'].values
F = df['has_Formal'].values
n = len(Y)

print("=" * 65)
print("GATE x CHANNEL MODEL (PCC/SCC Separation)")
print("Tamaki's SSOC Architecture Applied to CPG")
print("=" * 65)
print("N = %d items, 10 cultural regions" % n)

# ================================================================
# MODEL 0: OLS baseline (for comparison)
# ================================================================
X0 = np.column_stack([np.ones(n), R, E, Z])
b0 = np.linalg.lstsq(X0, Y, rcond=None)[0]
Y0 = X0 @ b0
SS_tot = np.sum((Y - np.mean(Y))**2)
R2_ols = 1 - np.sum((Y - Y0)**2) / SS_tot

# ================================================================
# MODEL 1: Gate (linear PCC) x Channel (Hill function SCC)
#
# sim = Gate(R, E, Z, Phonetic, Formal) * Channel(S)
#
# Gate = a0 + a1*R + a2*E + a3*Z + a4*Phonetic + a5*Formal
# Channel = 1 - alpha * S^hill_n / (S^hill_n + K^hill_n)
#
# When S=0: Channel=1 (no SCC effect)
# When S>>K: Channel approaches 1-alpha (SCC suppresses similarity)
# ================================================================

def gate_channel_model(params, R, E, Z, S, P, F):
    a0, a1, a2, a3, a4, a5, alpha, K, hill_n = params
    gate = a0 + a1*R + a2*E + a3*Z + a4*P + a5*F
    S_safe = np.maximum(S, 1e-10)
    K_safe = max(K, 1e-10)
    hill_n_safe = max(hill_n, 0.1)
    channel = 1.0 - alpha * (S_safe**hill_n_safe) / (S_safe**hill_n_safe + K_safe**hill_n_safe)
    return gate * channel

def loss_gc(params):
    pred = gate_channel_model(params, R, E, Z, S, P, F)
    return np.sum((Y - pred)**2)

# Initial guess
x0 = [0.15, 0.05, 0.10, 0.02, 0.08, 0.05, 0.3, 0.3, 2.0]
bounds = [
    (-0.5, 1.0),   # a0
    (-0.5, 0.5),   # a1
    (-0.5, 0.5),   # a2
    (-0.1, 0.2),   # a3
    (-0.3, 0.3),   # a4
    (-0.3, 0.3),   # a5
    (0.01, 0.99),  # alpha (SCC suppression strength)
    (0.01, 0.99),  # K (Hill half-activation)
    (0.5, 10.0),   # hill_n (cooperativity)
]

from scipy.optimize import differential_evolution
result = differential_evolution(loss_gc, bounds, seed=42, maxiter=1000, tol=1e-10)
params_best = result.x

Y_gc = gate_channel_model(params_best, R, E, Z, S, P, F)
SS_res_gc = np.sum((Y - Y_gc)**2)
R2_gc = 1 - SS_res_gc / SS_tot
p_gc = len(params_best)
R2_adj_gc = 1 - (1 - R2_gc) * (n - 1) / (n - p_gc - 1)

a0, a1, a2, a3, a4, a5, alpha, K, hill_n = params_best

print("\n" + "=" * 65)
print("MODEL 1: Gate x Channel (PCC x SCC)")
print("=" * 65)
print("R2 = %.4f  (OLS baseline: %.4f)" % (R2_gc, R2_ols))
print("Adj R2 = %.4f" % R2_adj_gc)
print("Improvement: +%.1f%% absolute, %.1fx relative" % ((R2_gc - R2_ols)*100, R2_gc/R2_ols if R2_ols > 0 else 0))
print("\n--- Gate (PCC: linear) ---")
print("  a0 (intercept) = %.4f" % a0)
print("  a1 (R)         = %.4f" % a1)
print("  a2 (E)         = %.4f" % a2)
print("  a3 (Z)         = %.4f" % a3)
print("  a4 (Phonetic)  = %.4f" % a4)
print("  a5 (Formal)    = %.4f" % a5)
print("\n--- Channel (SCC: Hill function) ---")
print("  alpha (suppression) = %.4f" % alpha)
print("  K (half-activation) = %.4f" % K)
print("  hill_n (cooperativity) = %.4f" % hill_n)
print("\n  Interpretation:")
print("  When S=0:   Channel = 1.000 (no SCC effect)")
print("  When S=K:   Channel = %.3f (half suppression)" % (1 - alpha * 0.5))
print("  When S=0.6: Channel = %.3f" % (1 - alpha * (0.6**hill_n) / (0.6**hill_n + K**hill_n)))
print("  When S=0.8: Channel = %.3f" % (1 - alpha * (0.8**hill_n) / (0.8**hill_n + K**hill_n)))

# ================================================================
# MODEL 2: Additive (no multiplication) — to test if Gate*Channel
# is genuinely better than Gate+Channel
# ================================================================
def additive_model(params, R, E, Z, S, P, F):
    a0, a1, a2, a3, a4, a5, b_s = params
    return a0 + a1*R + a2*E + a3*Z + a4*P + a5*F + b_s*S

def loss_add(params):
    pred = additive_model(params, R, E, Z, S, P, F)
    return np.sum((Y - pred)**2)

x0_add = [0.15, 0.05, 0.10, 0.02, 0.08, 0.05, -0.05]
res_add = minimize(loss_add, x0_add, method='Nelder-Mead')
Y_add = additive_model(res_add.x, R, E, Z, S, P, F)
R2_add = 1 - np.sum((Y - Y_add)**2) / SS_tot

print("\n" + "=" * 65)
print("MODEL 2: Additive (Gate + S linear)")
print("=" * 65)
print("R2 = %.4f" % R2_add)
print("S coefficient = %.4f" % res_add.x[6])

# ================================================================
# MODEL 3: Gate x Channel with region random effects
# (manual implementation — add region mean as offset)
# ================================================================
region_means = df.groupby('region')['sim'].mean()
grand_mean = Y.mean()
region_offsets = df['region'].map(region_means) - grand_mean
Y_centered = Y - region_offsets.values

# Fit Gate x Channel on region-centered data
def loss_gc_centered(params):
    pred = gate_channel_model(params, R, E, Z, S, P, F)
    return np.sum((Y_centered - pred)**2)

result_c = differential_evolution(loss_gc_centered, bounds, seed=42, maxiter=1000, tol=1e-10)
params_c = result_c.x
Y_gc_c = gate_channel_model(params_c, R, E, Z, S, P, F)

# Total model: region_offset + Gate*Channel
Y_total = Y_gc_c + region_offsets.values
SS_res_total = np.sum((Y - Y_total)**2)
R2_total = 1 - SS_res_total / SS_tot

a0c, a1c, a2c, a3c, a4c, a5c, alpha_c, K_c, hill_nc = params_c

print("\n" + "=" * 65)
print("MODEL 3: Region-centered Gate x Channel (HLM + PCC*SCC)")
print("=" * 65)
print("R2 = %.4f  (OLS: %.4f, GxC alone: %.4f)" % (R2_total, R2_ols, R2_gc))
print("\n--- Channel (SCC) after region centering ---")
print("  alpha = %.4f" % alpha_c)
print("  K     = %.4f" % K_c)
print("  hill_n = %.4f" % hill_nc)

# ================================================================
# S-COMPONENT DEEP DIVE: Channel function visualization
# ================================================================
print("\n" + "=" * 65)
print("CHANNEL FUNCTION: S -> Channel(S)")
print("(How S-component modulates embedding similarity)")
print("=" * 65)

s_range = np.arange(0, 1.01, 0.1)
for s_val in s_range:
    ch = 1.0 - alpha * (max(s_val,1e-10)**hill_n) / (max(s_val,1e-10)**hill_n + K**hill_n)
    bar = '#' * int(ch * 50)
    print("  S=%.1f: Channel=%.3f  %s" % (s_val, ch, bar))

# ================================================================
# RESIDUAL ANALYSIS: Which items are best/worst predicted?
# ================================================================
residuals = Y - Y_gc
df['residual'] = residuals
df['predicted'] = Y_gc

print("\n" + "=" * 65)
print("BEST PREDICTED (|residual| < 0.02)")
print("=" * 65)
good = df[df['residual'].abs() < 0.02].sort_values('residual', key=abs)
for _, row in good.head(10).iterrows():
    print("  %s: actual=%.4f pred=%.4f resid=%+.4f S=%.1f" %
          (row['id'], row['sim'], row['predicted'], row['residual'], row['S']))

print("\n" + "=" * 65)
print("WORST PREDICTED (largest |residual|)")
print("=" * 65)
worst = df.sort_values('residual', key=abs, ascending=False)
for _, row in worst.head(10).iterrows():
    print("  %s: actual=%.4f pred=%.4f resid=%+.4f S=%.1f region=%s" %
          (row['id'], row['sim'], row['predicted'], row['residual'], row['S'], row['region']))

# ================================================================
# COMPARISON TABLE
# ================================================================
print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)
print("%-45s  R2      AdjR2" % "Model")
print("-" * 65)
print("%-45s  %.4f  %.4f" % ("OLS: sim ~ R+E+Z", R2_ols, 1-(1-R2_ols)*(n-1)/(n-4)))
print("%-45s  %.4f  " % ("Additive: sim ~ R+E+Z+Phonetic+Formal+S", R2_add))
print("%-45s  %.4f  %.4f" % ("Gate x Channel (PCC*SCC)", R2_gc, R2_adj_gc))
print("%-45s  %.4f  " % ("Region-centered Gate x Channel", R2_total))

print("\n" + "=" * 65)
print("PHYSICAL INTERPRETATION")
print("=" * 65)
print("""
Gate (PCC) controls WHICH covert paths are possible:
  - R component: meaning->word direction (taboo avoidance)
  - E component: word->meaning direction (cultural assignment)
  - Z register: public vs private transmission
  - Phonetic/Formal vectors: specific encoding channels

Channel (SCC) controls HOW MUCH emergence occurs:
  - S component activates the Hill function
  - Below threshold K=%.2f: minimal emergence (unitary CPG)
  - Above threshold: cooperative emergence (compound CPG)
  - hill_n=%.1f: cooperativity coefficient
    (>1 = ultrasensitive switch, <1 = gradual)

The product Gate*Channel means:
  - PCC alone cannot create emergence (needs SCC)
  - SCC alone cannot operate without structure (needs PCC)
  - BOTH are required for full CPG function
  = "PCC without SCC is a dictionary. SCC without PCC is noise."
""" % (K, hill_n))

# Save
results = {
    'gate_channel': {
        'R2': round(R2_gc, 4), 'R2_adj': round(R2_adj_gc, 4),
        'gate': {'a0': round(a0,4), 'a1_R': round(a1,4), 'a2_E': round(a2,4),
                 'a3_Z': round(a3,4), 'a4_Phonetic': round(a4,4), 'a5_Formal': round(a5,4)},
        'channel': {'alpha': round(alpha,4), 'K': round(K,4), 'hill_n': round(hill_n,4)},
    },
    'comparison': {
        'OLS_R2': round(R2_ols, 4),
        'additive_R2': round(R2_add, 4),
        'gate_channel_R2': round(R2_gc, 4),
        'region_centered_R2': round(R2_total, 4),
    },
    'improvement_over_OLS': round((R2_gc / R2_ols if R2_ols > 0 else 0), 2),
}
with open(os.path.join(BASE, 'cpg_gate_channel_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to cpg_gate_channel_results.json")
