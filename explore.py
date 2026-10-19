"""
Messi Valuation with Chain Credit - Cycle 8, Day 6 project
Cycle 6 tried to value Messi by swapping his goals and assists for a
replacement player's, and found almost nothing (16 actual points vs 16.7
simulated), because he scored or assisted in only 2 of 6 matches. This
project redoes the swap with shared credit: how much of Argentina's chance
quality was Messi part of, and what happens if an average attacker is
part of it instead?

Method (one pass per credit measure, all in the same model so they compare):
  1. For every Argentina match Messi played, take Argentina's non-penalty
     shot xG.
  2. Messi's share = (xG of the possessions he was part of) / team xG.
     Three credit measures are used:
       direct   shots he took + shots he created with the final pass
                (the goals + assists view, measured in xG)
       chain    every possession he passed, carried, dribbled or shot in
       buildup  chain minus the shots he took and the final passes he made
  3. Replacement: draw a share at random from attackers (wingers, forwards,
     attacking midfielders) in the tournament who had at least 40 touches in
     a match. Zeros are kept in the pool. Messi is excluded.
  4. Adjusted Argentina xG = team xG * (1 - Messi share + replacement share).
  5. Expected points come from Poisson goals for Argentina (adjusted xG) and
     for the opponent (its actual xG), 3 points for a win, 1 for a draw.
  6. Messi's value = expected points with him minus with the replacement.
     10,000 draws per measure. A second, weaker pool (bottom half of the
     attackers by share) is run as a check.

Limitations: the model is xG-based, so expected points will not equal the
real points. It assumes the moves Messi was part of would not happen
without him (an upper bound), balanced by the replacement taking part in
moves at his own typical rate. Touches stand in for minutes played (no
lineup data), so substitutes are mostly kept out of the pool. Every touch
counts equally, defending and off-ball movement are not credited, and a
handful of matches is a small sample.

Outputs: messi_value_by_measure.png, messi_share_by_match.png,
messi_swap_replay.mp4 (process replay plus the two charts as end cards),
messi_value_summary.csv, messi_match_shares.csv

Reuses: copa_america_2024_matches.csv, copa_america_2024_events.csv
"""

import os
import subprocess
from math import factorial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Patch
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

FOCUS_TEAM = "Argentina"
FOCUS_PLAYER = "Lionel Andrés Messi Cuccittini"
N_RUNS = 10000
SEED = 7
MIN_EVENTS = 40
MAX_GOALS = 12
END_CARDS = ["messi_value_by_measure.png", "messi_share_by_match.png"]
END_CARD_SECONDS = 4

BG = "#0d1b2a"
GRID = "#3a4a5a"
BLUE = "#4a90d9"
ORANGE = "#f4a300"
MUTED = "#a8b8c8"
GREEN = "#6fcf97"
RED = "#e76f51"

MEASURES = [("direct", "Shots + key passes\n(the goals + assists view)"),
            ("chain", "xG chain\n(everyone involved)"),
            ("buildup", "xG buildup\n(without shot and final pass)")]
POOLS = ["average attacker", "weaker attacker"]

matches = pd.read_csv("copa_america_2024_matches.csv")
events = pd.read_csv("copa_america_2024_events.csv", low_memory=False)
events = events[events["period"].isin([1, 2, 3, 4])].copy()
for col in ["shot_type", "pass_shot_assist", "pass_goal_assist", "shot_outcome", "position"]:
    if col not in events.columns:
        events[col] = np.nan

if FOCUS_PLAYER not in set(events["player"].dropna()):
    hits = [p for p in events["player"].dropna().unique() if "Messi" in p]
    if not hits:
        raise ValueError("Messi not found in the events file")
    FOCUS_PLAYER = hits[0]


def truthy(s):
    return s.astype(str).str.lower().eq("true")


# ---------- credit per player per match ----------
shots = events[(events["type"] == "Shot") & events["shot_statsbomb_xg"].notna()].copy()
shots = shots[shots["shot_type"].fillna("") != "Penalty"]
poss_xg = (shots.groupby(["match_id", "possession", "team"], as_index=False)["shot_statsbomb_xg"]
           .sum().rename(columns={"shot_statsbomb_xg": "xg", "team": "shot_team"}))
team_xg = poss_xg.groupby(["match_id", "shot_team"])["xg"].sum()

TOUCH_TYPES = ["Pass", "Carry", "Dribble", "Shot"]
inv = events[events["type"].isin(TOUCH_TYPES) & events["player"].notna()].merge(
    poss_xg, on=["match_id", "possession"], how="inner")
inv = inv[inv["team"] == inv["shot_team"]].copy()
inv["is_shooter"] = inv["type"] == "Shot"
inv["is_key"] = (inv["type"] == "Pass") & (truthy(inv["pass_shot_assist"]) | truthy(inv["pass_goal_assist"]))

pp = inv.groupby(["match_id", "possession", "player", "team"], as_index=False).agg(
    xg=("xg", "first"), shooter=("is_shooter", "max"), key=("is_key", "max"))
pp["chain"] = pp["xg"]
pp["direct"] = np.where(pp["shooter"] | pp["key"], pp["xg"], 0.0)
pp["buildup"] = pp["chain"] - pp["direct"]

pm = pp.groupby(["match_id", "player", "team"], as_index=False)[["chain", "buildup", "direct"]].sum()
app = (events[events["player"].notna()].groupby(["match_id", "player", "team"])
       .size().reset_index(name="n_events"))
pm_full = app.merge(pm, on=["match_id", "player", "team"], how="left")
for c in ["chain", "buildup", "direct"]:
    pm_full[c] = pm_full[c].fillna(0.0)
txg = team_xg.reset_index().rename(columns={"shot_team": "team", "xg": "team_xg"})
pm_full = pm_full.merge(txg, on=["match_id", "team"], how="left")
pm_full["team_xg"] = pm_full["team_xg"].fillna(0.0)
for k, _ in MEASURES:
    pm_full["s_" + k] = (pm_full[k] / pm_full["team_xg"]).where(pm_full["team_xg"] > 0)


# ---------- replacement pools ----------
def is_attacker(pos):
    p = str(pos)
    if "Back" in p or "Goalkeeper" in p:
        return False
    return any(k in p for k in ["Wing", "Forward", "Striker", "Attacking Midfield"])


modal = (events.dropna(subset=["player", "position"]).groupby("player")["position"]
         .agg(lambda s: s.mode().iloc[0]))
attackers = {p for p, pos in modal.items() if is_attacker(pos)}
pool_df = pm_full[(pm_full["player"] != FOCUS_PLAYER) & (pm_full["n_events"] >= MIN_EVENTS)
                  & (pm_full["team_xg"] > 0)]
att_df = pool_df[pool_df["player"].isin(attackers)]
if len(att_df) < 30:
    print("Warning: too few attacker rows, using every player with enough touches")
    att_df = pool_df

pools = {}
for k, _ in MEASURES:
    v = np.sort(att_df["s_" + k].dropna().to_numpy())
    pools[k] = {"average attacker": v, "weaker attacker": v[: max(len(v) // 2, 1)]}

print(f"\nMessi's most common position: {modal.get(FOCUS_PLAYER, 'unknown')}")
print(f"Replacement pool: {len(att_df)} attacker-matches, {att_df['player'].nunique()} players, "
      f"at least {MIN_EVENTS} touches each")
for k, lab in MEASURES:
    v = pools[k]["average attacker"]
    print(f"  {k:<8} average share {v.mean():.3f}, median {np.median(v):.3f}, "
          f"zeros {int((v == 0).sum())} of {len(v)}")

# ---------- Argentina matches ----------
arg = matches[(matches["home_team"] == FOCUS_TEAM) | (matches["away_team"] == FOCUS_TEAM)].copy()
arg = arg.sort_values("match_date") if "match_date" in arg.columns else arg.sort_values("match_id")

mrows = []
for r in arg.itertuples():
    sel = pm_full[(pm_full["match_id"] == r.match_id) & (pm_full["player"] == FOCUS_PLAYER)]
    if sel.empty:
        continue
    home = r.home_team == FOCUS_TEAM
    opp = r.away_team if home else r.home_team
    xg = float(team_xg.get((r.match_id, FOCUS_TEAM), 0.0))
    opp_xg = float(team_xg.get((r.match_id, opp), 0.0))
    gf, ga = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
    pts = 3 if gf > ga else (1 if gf == ga else 0)
    sm = {k: (float(np.nan_to_num(sel["s_" + k].iloc[0])) if xg > 0 else 0.0) for k, _ in MEASURES}
    mrows.append(dict(match_id=r.match_id, opp=opp, xg=xg, opp_xg=opp_xg, pts=pts, s_m=sm))
n_m = len(mrows)

# ---------- expected points model ----------
FACT = np.array([factorial(k) for k in range(MAX_GOALS + 1)], dtype=float)
KS = np.arange(MAX_GOALS + 1).reshape(1, -1)
PTS = np.where(KS.T > KS, 3.0, np.where(KS.T == KS, 1.0, 0.0))


def pmf(lam):
    lam = np.asarray(lam, dtype=float).reshape(-1, 1)
    return np.exp(-lam) * lam ** KS / FACT


def exp_points(lam_arg, opp_xg):
    P = pmf(lam_arg)
    Q = pmf([opp_xg])[0]
    return P @ (PTS @ Q)


for m in mrows:
    m["e0"] = float(exp_points(np.array([m["xg"]]), m["opp_xg"])[0])
base_total = sum(m["e0"] for m in mrows)
actual_total = sum(m["pts"] for m in mrows)
print(f"\nArgentina's {n_m} matches with Messi: {actual_total} real points, "
      f"{base_total:.2f} expected points from xG alone")

# ---------- Monte Carlo swap ----------
rng = np.random.default_rng(SEED)
results = {}
for k, _ in MEASURES:
    for pname in POOLS:
        pool = pools[k][pname]
        total = np.zeros(N_RUNS)
        per_match = []
        for m in mrows:
            s_r = rng.choice(pool, N_RUNS)
            lam = np.maximum(0.0, m["xg"] * (1.0 - m["s_m"][k] + s_r))
            e = exp_points(lam, m["opp_xg"])
            total += e
            per_match.append(float(e.mean()))
        value = base_total - total
        results[(k, pname)] = dict(value=value, per_match=per_match)

summary_rows = []
for k, _ in MEASURES:
    for pname in POOLS:
        v = results[(k, pname)]["value"]
        summary_rows.append(dict(measure=k, replacement=pname, mean=float(v.mean()), std=float(v.std()),
                                 p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                                 share_positive=float((v > 0).mean())))
summary = pd.DataFrame(summary_rows)
summary.to_csv("messi_value_summary.csv", index=False)
sm_of = {(r["measure"], r["replacement"]): r for r in summary_rows}

print("\nMessi's value in points over the tournament (positive = he added points):")
print(summary.round(2).to_string(index=False))

ms_rows = []
print("\nPer match (chain credit, average attacker):")
for i, m in enumerate(mrows):
    after = results[("chain", "average attacker")]["per_match"][i]
    pool_mean = float(pools["chain"]["average attacker"].mean())
    print(f"  {i + 1}. vs {m['opp']:<10} Argentina xG {m['xg']:.2f}  Messi share {m['s_m']['chain']:.0%}  "
          f"average attacker {pool_mean:.0%}  exp. points {m['e0']:.2f} -> {after:.2f}")
    ms_rows.append(dict(match=i + 1, opponent=m["opp"], arg_xg=m["xg"],
                        share_direct=m["s_m"]["direct"], share_chain=m["s_m"]["chain"],
                        share_buildup=m["s_m"]["buildup"], pool_mean_chain=pool_mean,
                        exp_points_with=m["e0"], exp_points_without=after))
pd.DataFrame(ms_rows).to_csv("messi_match_shares.csv", index=False)

# ---------- chart 1: value by measure ----------
fig1, ax1 = plt.subplots(figsize=(11, 7.5))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
lo = min(0.0, summary["p5"].min())
hi = max(0.0, summary["p95"].max())
span = max(hi - lo, 0.2)
pad = span * 0.02
for mi, (k, lab) in enumerate(MEASURES):
    y0 = len(MEASURES) - 1 - mi
    for off, pname, col in ((0.2, "average attacker", BLUE), (-0.2, "weaker attacker", ORANGE)):
        s = sm_of[(k, pname)]
        ax1.barh(y0 + off, s["mean"], height=0.34, color=col)
        ax1.errorbar(s["mean"], y0 + off, xerr=[[s["mean"] - s["p5"]], [s["p95"] - s["mean"]]],
                     fmt="none", ecolor="white", capsize=4, linewidth=1.4)
        ax1.text(max(s["p95"], 0.0) + pad * 2, y0 + off, f"{s['mean']:+.2f}", va="center",
                 color="white", fontsize=11)
ax1.axvline(0, color="white", linewidth=1)
ax1.set_xlim(lo - span * 0.08, hi + span * 0.22)
ax1.set_yticks(range(len(MEASURES)))
ax1.set_yticklabels([lab for _, lab in MEASURES][::-1], color="white", fontsize=12)
ax1.set_xlabel("Points Messi adds over the replacement across Argentina's matches (positive = he added points)",
               color="white", fontsize=10.5)
ax1.tick_params(axis="x", colors="white")
ax1.tick_params(axis="y", length=0)
for sp in ax1.spines.values():
    sp.set_color(GRID)
ax1.legend(handles=[Patch(color=BLUE, label="Replaced by an average attacker"),
                    Patch(color=ORANGE, label="Replaced by a weaker attacker (bottom half)")],
           facecolor=BG, labelcolor="white", edgecolor=GRID, loc="upper center",
           bbox_to_anchor=(0.5, -0.12), ncol=2)
fig1.suptitle("What Is Messi Worth When the Whole Move Gets Credit?", color="white", fontsize=17,
              fontweight="bold", y=0.97)
ax1.set_title(f"Mean over {N_RUNS:,} swaps, whiskers show the 5th to 95th percentile. Copa America 2024",
              color=MUTED, fontsize=10.5, pad=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig1.savefig("messi_value_by_measure.png", dpi=150, facecolor=BG)
plt.close(fig1)
print("\nSaved chart: messi_value_by_measure.png")

# ---------- chart 2: share by match ----------
fig2, ax2 = plt.subplots(figsize=(11, 7.5))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)
xs = np.arange(n_m)
mine = np.array([m["s_m"]["chain"] for m in mrows]) * 100
avg_share = float(pools["chain"]["average attacker"].mean()) * 100
ax2.bar(xs - 0.2, mine, width=0.38, color=BLUE, label="Messi")
ax2.bar(xs + 0.2, [avg_share] * n_m, width=0.38, color=ORANGE, label="Average attacker in the tournament")
top_v = max(mine.max(), avg_share)
for i in range(n_m):
    ax2.text(xs[i] - 0.2, mine[i] + top_v * 0.02, f"{mine[i]:.0f}%", ha="center", color="white", fontsize=11)
    ax2.text(xs[i] + 0.2, avg_share + top_v * 0.02, f"{avg_share:.0f}%", ha="center", color="white", fontsize=11)
ax2.set_ylim(0, top_v * 1.18)
ax2.set_xticks(xs)
ax2.set_xticklabels([f"M{i + 1}\n{m['opp']}" for i, m in enumerate(mrows)], color="white", fontsize=11)
ax2.set_ylabel("Share of Argentina's shot xG in moves the player was part of (%)", color="white", fontsize=10.5)
ax2.tick_params(axis="y", colors="white")
ax2.tick_params(axis="x", length=0)
for sp in ax2.spines.values():
    sp.set_color(GRID)
ax2.legend(facecolor=BG, labelcolor="white", edgecolor=GRID, loc="upper center",
           bbox_to_anchor=(0.5, -0.12), ncol=2)
fig2.suptitle("How Much of Argentina's Chance Quality Did Messi Touch?", color="white", fontsize=17,
              fontweight="bold", y=0.97)
ax2.set_title("xG chain share per match, Copa America 2024, penalties excluded",
              color=MUTED, fontsize=10.5, pad=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig("messi_share_by_match.png", dpi=150, facecolor=BG)
plt.close(fig2)
print("Saved chart: messi_share_by_match.png")

# ---------- process replay ----------
VK, VP = "chain", "average attacker"
vals = results[(VK, VP)]["value"]
pool_vals = pools[VK][VP]
pool_mean = float(pool_vals.mean())
rng_v = np.random.default_rng(1)
for m in mrows:
    m["draws"] = rng_v.choice(pool_vals, 6)
for i, m in enumerate(mrows):
    m["after"] = results[(VK, VP)]["per_match"][i]
    m["val"] = m["e0"] - m["after"]
cum = np.concatenate([[0.0], np.cumsum([m["val"] for m in mrows])])
xmax = max(max(m["xg"], m["xg"] * (1 - m["s_m"][VK] + pool_mean)) for m in mrows) * 1.08

counts, edges = np.histogram(vals, bins=50)
centers = (edges[:-1] + edges[1:]) / 2
bw = edges[1] - edges[0]
h_lo, h_hi = min(edges[0], 0) - bw, max(edges[-1], 0) + bw
dir_mean = sm_of[("direct", VP)]["mean"]


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


fig = plt.figure(figsize=(10, 10), dpi=108)  # 1080 x 1080 pixels
fig.patch.set_facecolor(BG)
axb = fig.add_axes([0.06, 0.52, 0.88, 0.22])
axh = fig.add_axes([0.10, 0.40, 0.80, 0.34])
fig.text(0.5, 0.955, "What Is Messi Worth When the Whole Move Counts?", ha="center",
         color="white", fontsize=22, fontweight="bold")
fig.text(0.5, 0.915, "Argentina, Copa America 2024: take Messi out, put an average attacker in",
         ha="center", color=MUTED, fontsize=13)
head_t = fig.text(0.5, 0.855, "", ha="center", color="white", fontsize=18, fontweight="bold")
step_t = fig.text(0.5, 0.80, "", ha="center", color=ORANGE, fontsize=19, fontweight="bold")
info1_t = fig.text(0.5, 0.455, "", ha="center", color=ORANGE, fontsize=15)
info2_t = fig.text(0.5, 0.41, "", ha="center", color=GREEN, fontsize=15)
res1_t = fig.text(0.5, 0.325, "", ha="center", color="white", fontsize=16)
res2_t = fig.text(0.5, 0.265, "", ha="center", color="white", fontsize=18, fontweight="bold")
tally_t = fig.text(0.5, 0.15, "", ha="center", color=ORANGE, fontsize=20, fontweight="bold")
fig.text(0.5, 0.012, "StatsBomb open data, Copa America 2024. xG from StatsBomb's shot model.",
         ha="center", color=MUTED, fontsize=9)

STEPS = ["Argentina's chance quality in this match", "Step 1: the moves Messi was part of",
         "Step 2: take Messi out", "Step 3: put an average attacker in", "Step 4: what the swap is worth"]


def draw(fr):
    kind, mi, st, p = fr
    axb.clear()
    axh.clear()
    axb.set_visible(kind == "m")
    axh.set_visible(kind == "f")
    if kind == "f":
        a = ease(p)
        axh.set_facecolor(BG)
        axh.bar(centers, counts, width=bw * 0.95, color=BLUE, alpha=a)
        axh.axvline(0, color="white", linewidth=1.2, linestyle="--", alpha=a)
        axh.axvline(vals.mean(), color=ORANGE, linewidth=2.2, alpha=a)
        axh.set_xlim(h_lo, h_hi)
        axh.set_ylim(0, counts.max() * 1.15)
        axh.set_yticks([])
        axh.tick_params(axis="x", colors="white")
        axh.set_xlabel("Points Messi adds over an average attacker (dashed line = no difference)",
                       color="white", fontsize=11)
        for sp in axh.spines.values():
            sp.set_color(GRID)
        head_t.set_text(f"{n_m} matches, {N_RUNS:,} simulated swaps")
        step_t.set_text("Messi's value in points" if a > 0.3 else "")
        info1_t.set_text("")
        info2_t.set_text("")
        s = sm_of[(VK, VP)]
        res1_t.set_position((0.5, 0.25))
        res2_t.set_position((0.5, 0.19))
        res1_t.set_text(f"xG chain view: {s['mean']:+.2f} points (5th to 95th: {s['p5']:+.2f} to {s['p95']:+.2f})"
                        if a > 0.6 else "")
        res2_t.set_text(f"Goals + assists view (in xG): {dir_mean:+.2f} points" if a > 0.9 else "")
        res2_t.set_color(ORANGE)
        tally_t.set_text("")
        return
    res1_t.set_position((0.5, 0.325))
    res2_t.set_position((0.5, 0.265))
    m = mrows[mi]
    xg = m["xg"]
    a_len = m["s_m"][VK] * xg
    e = ease(p)
    axb.set_facecolor(BG)
    axb.set_xlim(-0.02 * xmax, xmax)
    axb.set_ylim(-0.6, 1.3)
    axb.axis("off")
    h = 0.8
    y = -0.4
    total = xg
    g_share = pool_mean
    if st == 0:
        axb.add_patch(Rectangle((0, y), xg, h, color=BLUE))
    elif st == 1:
        axb.add_patch(Rectangle((0, y), xg, h, color=BLUE))
        axb.add_patch(Rectangle((0, y), a_len, h, color=ORANGE, alpha=e))
    elif st == 2:
        ow = a_len * (1 - e)
        axb.add_patch(Rectangle((0, y), ow, h, color=ORANGE))
        axb.add_patch(Rectangle((ow, y), xg - a_len, h, color=BLUE))
        total = xg - a_len * e
    else:
        axb.add_patch(Rectangle((0, y), xg - a_len, h, color=BLUE))
        if st == 3:
            if p < 0.7:
                idx = min(int(p / 0.7 * len(m["draws"])), len(m["draws"]) - 1)
                g_share = float(m["draws"][idx])
            else:
                last = float(m["draws"][-1])
                g_share = last + (pool_mean - last) * ease((p - 0.7) / 0.3)
        gl = g_share * xg
        axb.add_patch(Rectangle((xg - a_len, y), gl, h, color=GREEN))
        total = xg - a_len + gl
    axb.text(0, 0.55, f"Argentina xG: {total:.2f}", color="white", fontsize=15, va="bottom")
    head_t.set_text(f"Match {mi + 1} of {n_m}: Argentina vs {m['opp']}")
    step_t.set_text(STEPS[st])
    info1_t.set_text(f"Messi was part of {m['s_m'][VK] * 100:.0f}% of the xG ({a_len:.2f})" if st >= 1 else "")
    info2_t.set_text(f"An average attacker is part of {g_share * 100:.0f}% of the xG ({g_share * xg:.2f})"
                     if st >= 3 else "")
    if st == 4:
        res1_t.set_text(f"Expected points: {m['e0']:.2f} with Messi, {m['after']:.2f} with the replacement")
        res2_t.set_text(f"Messi's value in this match: {m['val']:+.2f} points")
        res2_t.set_color(GREEN if m["val"] >= 0 else RED)
        tally_t.set_text(f"Running total: {cum[mi + 1]:+.2f} points after {mi + 1} of {n_m} matches")
    else:
        res1_t.set_text("")
        res2_t.set_text("")
        tally_t.set_text(f"Running total: {cum[mi]:+.2f} points after {mi} of {n_m} matches" if mi > 0 else "")


STAGE_FRAMES = [5, 8, 10, 14, 11]
frames = []
for mi in range(n_m):
    for st, nf in enumerate(STAGE_FRAMES):
        for f in range(nf):
            frames.append(("m", mi, st, (f + 1) / nf))
for f in range(20):
    frames.append(("f", 0, 0, (f + 1) / 20))
frames += [("f", 0, 0, 1.0)] * 70

FPS = 20
RAW_VIDEO = "messi_swap_replay_raw.mp4"
FINAL_VIDEO = "messi_swap_replay.mp4"
anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames), interval=1000 / FPS)
writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
print(f"Rendering {len(frames)} frames ({len(frames) / FPS:.0f} seconds), this takes a minute...")
anim.save(RAW_VIDEO, writer=writer, dpi=108, savefig_kwargs={"facecolor": BG})
print("Saved replay:", RAW_VIDEO)


# ---------- append the charts to the video as end cards ----------
def append_end_cards(video_in, video_out, cards, seconds, size=1080, fps=20, bg="0d1b2a"):
    cards = [c for c in cards if os.path.exists(c)]
    if not cards:
        print("No chart PNGs found, keeping the replay without end cards.")
        os.replace(video_in, video_out)
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", video_in]
    for c in cards:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", c]
    norm = (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=0x{bg},"
            f"setsar=1,fps={fps},format=yuv420p")
    n = len(cards) + 1
    parts = [f"[{i}:v]{norm}[v{i}]" for i in range(n)]
    labels = "".join(f"[v{i}]" for i in range(n))
    fc = ";".join(parts) + f";{labels}concat=n={n}:v=1:a=0[out]"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", video_out]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:])
        print("End cards failed, keeping the replay without them.")
        os.replace(video_in, video_out)
        return
    os.remove(video_in)
    print("Added end cards:", ", ".join(cards))


append_end_cards(RAW_VIDEO, FINAL_VIDEO, END_CARDS, END_CARD_SECONDS, fps=FPS)
print("Saved video:", FINAL_VIDEO)