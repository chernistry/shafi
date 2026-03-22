#!/usr/bin/env python3
"""Analyze submission score history to find patterns, sensitivities, and projections."""

import itertools

# ── Raw data ────────────────────────────────────────────────────────────
submissions = [
    # (name, Det, Asst, G, T, F, Total_reported, date)
    ("v1",  0.814, 0.607, 0.638, 0.996, 1.041, 0.497, "Mar 11"),
    ("v2",  0.943, 0.673, 0.772, 0.996, 1.046, 0.694, "Mar 11"),
    ("v3",  0.943, 0.707, 0.766, 0.996, 1.046, 0.696, "Mar 12"),
    ("v4",  0.943, 0.667, 0.758, 0.996, 1.047, 0.680, "Mar 12"),
    ("v5",  0.943, 0.667, 0.801, 0.996, 1.047, 0.718, "Mar 12"),
    ("v6",  0.971, 0.693, 0.801, 0.996, 1.047, 0.742, "Mar 12"),
    ("v7",  0.971, 0.647, 0.608, 0.996, 1.047, 0.554, "Mar 12"),
    ("v8",  0.971, 0.687, 0.801, 0.996, 1.047, 0.740, "Mar 12"),
    ("v9",  0.971, 0.700, 0.654, 0.994, 1.047, 0.606, "Mar 12"),
    ("v10", 0.943, 0.693, 0.772, 0.995, 1.047, 0.698, "Mar 16"),
    ("v11", 0.957, 0.680, 0.689, 0.995, 1.043, 0.625, "Mar 17"),
    ("v12", 0.829, 0.667, 0.613, 0.993, 1.034, 0.491, "Mar 19"),
    ("v13", 0.943, 0.620, 0.650, 0.993, 1.029, 0.562, "Mar 19"),
    ("v14", 0.829, 0.673, 0.613, 0.993, 1.034, 0.492, "Mar 19"),
    ("v15", 0.929, 0.607, 0.593, 0.996, 1.045, 0.514, "Mar 19"),
]


def compute_total(det, asst, g, t, f):
    return (0.7 * det + 0.3 * asst) * g * t * f


# ═══════════════════════════════════════════════════════════════════════
# 1. FORMULA VERIFICATION
# ═══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("1. FORMULA VERIFICATION: Total = (0.7*Det + 0.3*Asst) * G * T * F")
print("=" * 72)
for name, det, asst, g, t, f, total_rep, date in submissions:
    calc = compute_total(det, asst, g, t, f)
    diff = calc - total_rep
    flag = " <-- MISMATCH" if abs(diff) > 0.005 else ""
    print(f"  {name:>3s}: reported={total_rep:.3f}  computed={calc:.3f}  diff={diff:+.3f}{flag}")

# ═══════════════════════════════════════════════════════════════════════
# 2. SENSITIVITY ANALYSIS (marginal return at v6 baseline)
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("2. SENSITIVITY ANALYSIS (partial derivatives at v6 baseline)")
print("=" * 72)

v6 = dict(det=0.971, asst=0.693, g=0.801, t=0.996, f=1.047)
base = compute_total(**v6)
delta = 0.01  # +1 percentage point

sensitivities = {}
for comp in ["det", "asst", "g", "t", "f"]:
    tweaked = dict(v6)
    tweaked[comp] += delta
    new_total = compute_total(**tweaked)
    marginal = (new_total - base) / delta
    sensitivities[comp] = marginal
    print(f"  d(Total)/d({comp:>4s}) = {marginal:.4f}   "
          f"(+0.01 in {comp} => +{new_total - base:.4f} total)")

ranked = sorted(sensitivities.items(), key=lambda x: -x[1])
print(f"\n  Ranking by marginal return: {' > '.join(f'{k}({v:.3f})' for k, v in ranked)}")

# ═══════════════════════════════════════════════════════════════════════
# 3. WHAT IMPROVEMENTS NEEDED TO HIT TARGETS FROM V6?
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("3. SINGLE-COMPONENT IMPROVEMENTS NEEDED FROM v6 (Total=0.742)")
print("=" * 72)

targets = [0.85, 0.90, 0.95]
for target in targets:
    print(f"\n  --- Target Total = {target:.2f} ---")
    for comp in ["det", "asst", "g", "t", "f"]:
        # Binary search for needed value
        lo, hi = v6[comp], 2.0
        for _ in range(100):
            mid = (lo + hi) / 2
            tweaked = dict(v6)
            tweaked[comp] = mid
            if compute_total(**tweaked) < target:
                lo = mid
            else:
                hi = mid
        needed = (lo + hi) / 2
        improvement = needed - v6[comp]
        feasible = "FEASIBLE" if needed <= 1.0 or comp == "f" else "INFEASIBLE (>1.0)"
        if comp in ("det", "asst", "g", "t") and needed > 1.0:
            feasible = "INFEASIBLE (>1.0)"
        elif comp == "f" and needed > 1.2:
            feasible = "UNLIKELY"
        print(f"    {comp:>4s}: need {needed:.3f} (+{improvement:.3f})  {feasible}")

# ═══════════════════════════════════════════════════════════════════════
# 4. CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("4. CORRELATION ANALYSIS ACROSS 15 SUBMISSIONS")
print("=" * 72)

import math

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x - mx)*(y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx)**2 for x in xs))
    dy = math.sqrt(sum((y - my)**2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)

components = ["Det", "Asst", "G", "T", "F", "Total"]
vectors = {
    "Det":   [s[1] for s in submissions],
    "Asst":  [s[2] for s in submissions],
    "G":     [s[3] for s in submissions],
    "T":     [s[4] for s in submissions],
    "F":     [s[5] for s in submissions],
    "Total": [s[6] for s in submissions],
}

print("\n  Pairwise Pearson correlations:")
print(f"  {'':>6s}", end="")
for c in components:
    print(f"  {c:>6s}", end="")
print()
for c1 in components:
    print(f"  {c1:>6s}", end="")
    for c2 in components:
        r = pearson(vectors[c1], vectors[c2])
        print(f"  {r:>6.3f}", end="")
    print()

print("\n  Correlations with Total (sorted):")
for c in ["Det", "Asst", "G", "T", "F"]:
    r = pearson(vectors[c], vectors["Total"])
    print(f"    {c:>4s} vs Total: r = {r:+.3f}")

# ═══════════════════════════════════════════════════════════════════════
# 5. OPTIMAL IMPROVEMENT ALLOCATION
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("5. OPTIMAL IMPROVEMENT ALLOCATION (from v6 baseline)")
print("=" * 72)

budget = 0.05  # 5 percentage points to allocate
steps = 50
print(f"\n  Budget: {budget:.2f} total improvement to distribute across Det, Asst, G")
print(f"  (T and F are largely external/fixed, so we focus on Det, Asst, G)\n")

best_alloc = None
best_total = 0
# Grid search over allocations of budget across det, asst, g
for d_steps in range(steps + 1):
    for a_steps in range(steps + 1 - d_steps):
        g_steps = steps - d_steps - a_steps
        d_add = budget * d_steps / steps
        a_add = budget * a_steps / steps
        g_add = budget * g_steps / steps
        t = compute_total(
            min(v6["det"] + d_add, 1.0),
            min(v6["asst"] + a_add, 1.0),
            min(v6["g"] + g_add, 1.0),
            v6["t"], v6["f"]
        )
        if t > best_total:
            best_total = t
            best_alloc = (d_add, a_add, g_add)

print(f"  Best allocation of +{budget:.2f}:")
print(f"    Det  += {best_alloc[0]:.3f}  (to {v6['det'] + best_alloc[0]:.3f})")
print(f"    Asst += {best_alloc[1]:.3f}  (to {v6['asst'] + best_alloc[1]:.3f})")
print(f"    G    += {best_alloc[2]:.3f}  (to {v6['g'] + best_alloc[2]:.3f})")
print(f"    => Total = {best_total:.4f}  (up from {base:.4f}, delta = +{best_total - base:.4f})")

# Also show what happens if we put all budget into one component
print(f"\n  Single-component allocation comparison (+{budget:.2f} each):")
for comp_name, comp_key in [("Det", "det"), ("Asst", "asst"), ("G", "g")]:
    tweaked = dict(v6)
    tweaked[comp_key] = min(tweaked[comp_key] + budget, 1.0)
    t = compute_total(**tweaked)
    print(f"    All into {comp_name:>4s}: Total = {t:.4f}  (delta = +{t - base:.4f})")

# ═══════════════════════════════════════════════════════════════════════
# 6. PROJECTION: 1792-dim embeddings improving G by +0.15 to +0.32
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("6. PROJECTION: 1792-dim embeddings (G improvement +0.15 to +0.32)")
print("=" * 72)

g_improvements = [0.15, 0.20, 0.25, 0.30, 0.32]
print(f"\n  From v6 baseline (G={v6['g']:.3f}, Total={base:.4f}):")
for dg in g_improvements:
    new_g = min(v6["g"] + dg, 1.0)
    t = compute_total(v6["det"], v6["asst"], new_g, v6["t"], v6["f"])
    print(f"    G += {dg:.2f} => G={new_g:.3f}, Total={t:.4f}  (delta = +{t - base:.4f})")

g_lo = min(v6["g"] + 0.15, 1.0)
g_hi = min(v6["g"] + 0.32, 1.0)
t_lo = compute_total(v6["det"], v6["asst"], g_lo, v6["t"], v6["f"])
t_hi = compute_total(v6["det"], v6["asst"], g_hi, v6["t"], v6["f"])
print(f"\n  PROJECTED TOTAL RANGE: {t_lo:.4f} to {t_hi:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 7. PROJECTION: Bug fixes (Det +0.02, Asst +0.01) combined with embeddings
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("7. COMBINED PROJECTION: Bug fixes + Embedding improvements")
print("=" * 72)

det_fix = 0.02
asst_fix = 0.01
new_det = min(v6["det"] + det_fix, 1.0)
new_asst = min(v6["asst"] + asst_fix, 1.0)

# Bug fixes alone
t_bugfix = compute_total(new_det, new_asst, v6["g"], v6["t"], v6["f"])
print(f"\n  Bug fixes alone (Det +{det_fix}, Asst +{asst_fix}):")
print(f"    Det={new_det:.3f}, Asst={new_asst:.3f}, G={v6['g']:.3f}")
print(f"    Total = {t_bugfix:.4f}  (delta = +{t_bugfix - base:.4f})")

# Bug fixes + embedding improvements
print(f"\n  Bug fixes + Embedding improvements:")
for dg in g_improvements:
    new_g = min(v6["g"] + dg, 1.0)
    t = compute_total(new_det, new_asst, new_g, v6["t"], v6["f"])
    print(f"    G += {dg:.2f} => Det={new_det:.3f}, Asst={new_asst:.3f}, G={new_g:.3f}, "
          f"Total={t:.4f}  (delta = +{t - base:.4f})")

t_combo_lo = compute_total(new_det, new_asst, g_lo, v6["t"], v6["f"])
t_combo_hi = compute_total(new_det, new_asst, g_hi, v6["t"], v6["f"])
print(f"\n  COMBINED PROJECTED RANGE: {t_combo_lo:.4f} to {t_combo_hi:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("EXECUTIVE SUMMARY")
print("=" * 72)
print(f"""
  Current best (v6): Total = {base:.4f}

  KEY FINDINGS:
  1. G (grounding) is the highest-leverage component:
     - Highest sensitivity ({sensitivities['g']:.3f} per unit)
     - Highest empirical correlation with Total
     - Biggest observed swings between submissions (0.593 to 0.801)

  2. The biggest score regressions (v6->v7, v6->v9, v10->v12) are ALL
     driven by G dropping. G is the dominant source of variance.

  3. Optimal strategy: focus almost entirely on G improvement.
     All-into-G beats any mixed allocation.

  4. With 1792-dim embeddings (G +0.15 to +0.32):
     Projected total: {t_lo:.3f} to {t_hi:.3f}

  5. Adding bug fixes (Det +0.02, Asst +0.01) on top:
     Projected total: {t_combo_lo:.3f} to {t_combo_hi:.3f}

  6. To reach 0.85 from v6, we need G ~ 0.918 (+0.117) -- achievable
     with the embedding upgrade alone.

  7. Det is already near ceiling (0.971). Asst improvements have only
     30% weight. T and F are near-constant multipliers we can't control.
     G is the one lever with both high sensitivity AND large room to grow.
""")
