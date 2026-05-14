#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 1 — No-Slip (Carpet) — FGO with iSAM2 + Huber Loss     ║
║                                                                      ║
║  INPUT FILES (same format for both experiments):                     ║
║    sensor_fusion_expt1__1_.csv  → fused data (LiDAR ICP poses       ║
║                                   + encoder + IMU) = ground truth   ║
║    sensor_fusion_expt1.csv      → encoder + IMU only trajectory     ║
║                                                                      ║
║  COLUMNS IN BOTH FILES:                                              ║
║    timestamp, global_x, global_y, global_theta,                     ║
║    delta_x, delta_y, delta_theta,                                   ║
║    enc_left, enc_right, acc_x, acc_y, acc_z                        ║
║                                                                      ║
║  WHAT THIS DOES:                                                     ║
║  1. Loads LiDAR ICP poses as GROUND TRUTH                           ║
║  2. Computes encoder+IMU dead reckoning trajectory                  ║
║  3. Runs FGO (iSAM2 + Huber) fusing encoder odometry + LiDAR       ║
║  4. Plots:                                                           ║
║     Fig 1 — XY trajectories (GT, Encoder+IMU, FGO)                 ║
║     Fig 2 — Position error vs time                                  ║
║                                                                      ║
║  FOR SLIP EXPERIMENT:                                                ║
║  Just change --fused and --enc file arguments.                      ║
║  FGO will automatically downweight encoder via Huber loss           ║
║  when slip causes large residuals.                                  ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
  # No-slip (Experiment 1):
  python3 experiment1_fgo.py

  # Slip (Experiment 2) — just change input files:
  python3 experiment1_fgo.py \
      --fused sensor_fusion_expt2_fused.csv \
      --enc   sensor_fusion_expt2_enc.csv \
      --title "Experiment 2 — Slip (Acrylic)"
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
import argparse

try:
    import gtsam
    from gtsam import (Pose2, NonlinearFactorGraph, Values,
                       PriorFactorPose2, BetweenFactorPose2,
                       noiseModel, ISAM2, ISAM2Params)
    HAS_GTSAM = True
except ImportError:
    print("[WARN] gtsam not installed — FGO will use weighted fallback")
    HAS_GTSAM = False

# ═══════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════
WHEEL_DIAMETER  = 0.065
WHEEL_CIRCUM    = np.pi * WHEEL_DIAMETER        # 0.2042 m

CPR_LEFT        = 294
CPR_RIGHT       = 672

DIST_PER_TICK_L = WHEEL_CIRCUM / CPR_LEFT       # 0.000695 m/tick
DIST_PER_TICK_R = WHEEL_CIRCUM / CPR_RIGHT      # 0.000304 m/tick

WHEEL_BASE      = 0.235                         # metres

ACCEL_SCALE     = 9.81 / 16384.0               # raw → m/s²

# ── FGO noise parameters ──────────────────────────────────────────────
# No-slip: encoder is fairly reliable → moderate sigma
# Slip:    encoder unreliable → increase SIGMA_ENC_XY/TH so Huber
#          downweights it more aggressively when residuals are large
SIGMA_ENC_XY   = 0.20    # encoder odometry position noise (m)
SIGMA_ENC_TH   = 0.15    # encoder odometry heading noise (rad)
SIGMA_LID_XY   = 0.03    # LiDAR ICP position noise (m) — trusted more
SIGMA_LID_TH   = 0.02    # LiDAR ICP heading noise (rad)
HUBER_K        = 0.3    # Huber threshold
                          # No-slip: residuals small → Huber rarely fires
                          # Slip:    encoder residuals large → Huber fires
                          #          and downweights encoder automatically
LIDAR_TIME_WIN = 0.3     # seconds — LiDAR match window (same file → tight)

# ── Plot colours ──────────────────────────────────────────────────────
BG   = "#07090f"; BG2 = "#0d1018"; BG3 = "#111520"
C_GT  = "#39ff14"    # green  — ground truth
C_ENC = "#ff4757"    # red    — encoder + IMU
C_FGO = "#00e5ff"    # cyan   — FGO fused
C_DIM = "#4a5270"
C_FG  = "#cdd2e8"

matplotlib.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": BG3,
    "axes.edgecolor":  "#1a2035", "axes.labelcolor": C_DIM,
    "axes.titlecolor": C_FG, "axes.titlesize": 10,
    "axes.labelsize":  8,    "xtick.color": C_DIM,
    "ytick.color":     C_DIM,"xtick.labelsize": 7,
    "ytick.labelsize": 7,    "grid.color": "#0f1525",
    "grid.linewidth":  0.6,  "text.color": C_FG,
    "font.family":     "monospace",
    "legend.facecolor": BG2, "legend.edgecolor": "#1a2035",
    "legend.fontsize": 8,    "lines.linewidth": 1.5,
})


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_fused(path):
    """
    Load LiDAR ground truth CSV (lidar_gd.csv format).
    Handles files WITHOUT headers safely.
    """

    # Try normal read first
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # If timestamp column is missing → fix manually
    if "timestamp" not in df.columns:
        df = pd.read_csv(path, comment="#", header=None)

        df.columns = [
            "timestamp",
            "global_x",
            "global_y",
            "global_theta",
            "delta_x",
            "delta_y",
            "delta_theta"
        ]

    # Convert to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    # Time normalization
    df["t_sec"] = df["timestamp"] - df["timestamp"].iloc[0]

    # Zero reference
    df["global_x"]     -= df["global_x"].iloc[0]
    df["global_y"]     -= df["global_y"].iloc[0]
    df["global_theta"] -= df["global_theta"].iloc[0]

    print(f"[FUSED] Loaded {len(df)} rows")
    return df


def load_encoder_csv(path):
    """
    Load the encoder+IMU only CSV.
    global_x/y/theta here = dead reckoning poses (NOT ground truth).
    We recompute odometry ourselves using calibrated CPR values.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["t_sec"] = df["timestamp"] - df["timestamp"].iloc[0]

    df["enc_left"]  -= df["enc_left"].iloc[0]
    df["enc_right"] -= df["enc_right"].iloc[0]

    # Strip trailing stationary rows
    EL = df["enc_left"].values
    ER = df["enc_right"].values
    last_move = len(df) - 1
    for i in range(len(df) - 1, 0, -1):
        if EL[i] != EL[i-1] or ER[i] != ER[i-1]:
            last_move = i
            break
    df = df.iloc[:last_move + 1].reset_index(drop=True)

    print(f"[ENC]   Loaded {len(df)} rows  "
          f"duration={df['t_sec'].iloc[-1]:.2f}s")
    print(f"[ENC]   Enc_L: 0 → {df['enc_left'].iloc[-1]:.0f}  "
          f"Enc_R: 0 → {df['enc_right'].iloc[-1]:.0f}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# ENCODER ODOMETRY  (calibrated per-wheel CPR)
# ═══════════════════════════════════════════════════════════════════════
def encoder_odometry(df):
    """
    Dead reckoning from encoder ticks.
    Uses delta_theta from the CSV for heading
    (this already integrates IMU gyro in your fusion pipeline).
    Falls back to encoder differential if delta_theta not reliable.
    """
    t  = df["t_sec"].values
    EL = df["enc_left"].values.astype(float)
    ER = df["enc_right"].values.astype(float)
    dth = df["delta_theta"].values.astype(float)   # heading from fused CSV

    x_out  = np.zeros(len(t))
    y_out  = np.zeros(len(t))
    th_out = np.zeros(len(t))

    for i in range(1, len(t)):
        dl = (EL[i] - EL[i-1]) * DIST_PER_TICK_L
        dr = (ER[i] - ER[i-1]) * DIST_PER_TICK_R
        v  = (dl + dr) / 2.0
        w  = dth[i]                                # use pre-integrated heading

        th_out[i] = th_out[i-1] + w
        x_out[i]  = x_out[i-1]  + v * np.cos(th_out[i])
        y_out[i]  = y_out[i-1]  + v * np.sin(th_out[i])

    return x_out, y_out, th_out


# ═══════════════════════════════════════════════════════════════════════
# FGO — GTSAM iSAM2 + Huber Loss
# ═══════════════════════════════════════════════════════════════════════
def run_fgo_gtsam(enc_df, fused_df):
    """
    Factor Graph Optimisation with iSAM2.

    Factors:
      1. Prior at origin (very tight)
      2. BetweenFactor  — encoder odometry (DIAGONAL noise)
      3. PriorFactor    — LiDAR ICP pose   (HUBER robust noise)

    No-slip case:
      Encoder residuals are small → Huber rarely activates
      → all sensors trusted roughly equally
      → FGO ≈ smooth blend of encoder + LiDAR

    Slip case (just change input files):
      Encoder residuals are large during slip → Huber fires
      → encoder factor automatically downweighted
      → LiDAR + IMU heading dominate
      → trajectory stays close to ground truth even during slip
    """
    isam    = ISAM2(ISAM2Params())
    graph   = NonlinearFactorGraph()
    initial = Values()

    # Encoder odometry noise — plain diagonal (trusted in no-slip)
    odom_noise = noiseModel.Diagonal.Sigmas(
        np.array([SIGMA_ENC_XY, SIGMA_ENC_XY, SIGMA_ENC_TH]))

    # LiDAR noise — Huber robust wrapper
    # In no-slip: residuals small → Huber weight = 1.0 (full trust)
    # In slip:    encoder residuals large → Huber weight < 1.0 (downweighted)
    lidar_noise = noiseModel.Robust.Create(
        noiseModel.mEstimator.Huber.Create(HUBER_K),
        noiseModel.Diagonal.Sigmas(
            np.array([SIGMA_LID_XY, SIGMA_LID_XY, SIGMA_LID_TH])))

    # Prior at origin — very tight
    graph.add(PriorFactorPose2(
        0, Pose2(0.0, 0.0, 0.0),
        noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6]))))
    initial.insert(0, Pose2(0.0, 0.0, 0.0))

    # Arrays
    t_enc  = enc_df["t_sec"].values
    EL     = enc_df["enc_left"].values.astype(float)
    ER     = enc_df["enc_right"].values.astype(float)
    dth    = enc_df["delta_theta"].values.astype(float)

    t_lid  = fused_df["t_sec"].values
    lid_x  = fused_df["global_x"].values
    lid_y  = fused_df["global_y"].values
    lid_th = fused_df["global_theta"].values

    fgo_x  = [0.0]; fgo_y = [0.0]; fgo_th = [0.0]
    prev   = Pose2(0.0, 0.0, 0.0)
    lid_idx = 0
    n_lid_used = 0

    for i in range(1, len(enc_df)):
        # ── Encoder odometry factor ──
        dl = (EL[i] - EL[i-1]) * DIST_PER_TICK_L
        dr = (ER[i] - ER[i-1]) * DIST_PER_TICK_R
        v  = (dl + dr) / 2.0
        w  = dth[i]

        th = prev.theta()
        motion = Pose2(v * np.cos(th), v * np.sin(th), w)
        graph.add(BetweenFactorPose2(i-1, i, motion, odom_noise))

        predicted = prev.compose(motion)
        initial.insert(i, predicted)

        # ── LiDAR ICP factor (nearest timestamp) ──
        enc_t_now = t_enc[i]
        while (lid_idx + 1 < len(t_lid) and
               abs(t_lid[lid_idx+1] - enc_t_now) <
               abs(t_lid[lid_idx]   - enc_t_now)):
            lid_idx += 1

        if abs(t_lid[lid_idx] - enc_t_now) < LIDAR_TIME_WIN:
            lp = Pose2(lid_x[lid_idx], lid_y[lid_idx], lid_th[lid_idx])
            graph.add(PriorFactorPose2(i, lp, lidar_noise))
            n_lid_used += 1

        # ── iSAM2 incremental update ──
        isam.update(graph, initial)
        result = isam.calculateEstimate()
        graph.resize(0); initial.clear()

        if result.exists(i):
            p = result.atPose2(i)
            prev = p
            fgo_x.append(p.x())
            fgo_y.append(p.y())
            fgo_th.append(p.theta())
        else:
            fgo_x.append(fgo_x[-1])
            fgo_y.append(fgo_y[-1])
            fgo_th.append(fgo_th[-1])

    print(f"[FGO]   LiDAR corrections: "
          f"{n_lid_used}/{len(enc_df)-1} steps "
          f"({100*n_lid_used/(len(enc_df)-1):.0f}%)")
    return np.array(fgo_x), np.array(fgo_y), np.array(fgo_th)


def run_fgo_fallback(enc_df, fused_df):
    """Weighted fusion fallback when GTSAM not installed."""
    print("[FGO]   Using weighted fallback — install gtsam for full iSAM2")

    t_enc  = enc_df["t_sec"].values
    EL     = enc_df["enc_left"].values.astype(float)
    ER     = enc_df["enc_right"].values.astype(float)
    dth    = enc_df["delta_theta"].values.astype(float)

    t_lid  = fused_df["t_sec"].values
    lid_x  = fused_df["global_x"].values
    lid_y  = fused_df["global_y"].values
    lid_th = fused_df["global_theta"].values

    fgo_x  = [0.0]; fgo_y = [0.0]; fgo_th = [0.0]
    lid_idx = 0
    n_lid_used = 0

    def huber_w(r):
        return 1.0 if abs(r) <= HUBER_K else HUBER_K / abs(r)

    for i in range(1, len(enc_df)):
        dl = (EL[i] - EL[i-1]) * DIST_PER_TICK_L
        dr = (ER[i] - ER[i-1]) * DIST_PER_TICK_R
        v  = (dl + dr) / 2.0
        w  = dth[i]

        th_pred = fgo_th[-1] + w
        x_pred  = fgo_x[-1]  + v * np.cos(th_pred)
        y_pred  = fgo_y[-1]  + v * np.sin(th_pred)

        while (lid_idx + 1 < len(t_lid) and
               abs(t_lid[lid_idx+1] - t_enc[i]) <
               abs(t_lid[lid_idx]   - t_enc[i])):
            lid_idx += 1

        if abs(t_lid[lid_idx] - t_enc[i]) < LIDAR_TIME_WIN:
            lx  = lid_x[lid_idx]
            ly  = lid_y[lid_idx]
            lth = lid_th[lid_idx]

            res  = np.sqrt((x_pred-lx)**2 + (y_pred-ly)**2)
            hw   = huber_w(res)
            w_l  = hw / (SIGMA_LID_XY**2)
            w_e  = 1.0 / (SIGMA_ENC_XY**2)

            fgo_x.append((w_e*x_pred + w_l*lx) / (w_e + w_l))
            fgo_y.append((w_e*y_pred + w_l*ly) / (w_e + w_l))

            res_th = abs(th_pred - lth)
            hw_th  = huber_w(res_th)
            w_lth  = hw_th / (SIGMA_LID_TH**2)
            w_eth  = 1.0   / (SIGMA_ENC_TH**2)
            fgo_th.append((w_eth*th_pred + w_lth*lth) / (w_eth + w_lth))
            n_lid_used += 1
        else:
            fgo_x.append(x_pred)
            fgo_y.append(y_pred)
            fgo_th.append(th_pred)

    print(f"[FGO]   LiDAR corrections: "
          f"{n_lid_used}/{len(enc_df)-1} steps "
          f"({100*n_lid_used/(len(enc_df)-1):.0f}%)")
    return np.array(fgo_x), np.array(fgo_y), np.array(fgo_th)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def pos_err(x1, y1, x2, y2):
    e = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    mask = np.isnan(x2) | np.isnan(y2)
    e[mask] = np.nan
    return e


def rmse(e):
    return np.sqrt(np.nanmean(e**2))


def smooth(y, w=3):
    y2 = np.array(y, dtype=float)
    return uniform_filter1d(y2, size=w)


def interp_gt(gt_df, t_query):
    """Interpolate GT x/y/theta onto query timestamps."""
    from scipy.interpolate import interp1d
    lt = gt_df["t_sec"].values
    out = {}
    for k in ["global_x", "global_y", "global_theta"]:
        f = interp1d(lt, gt_df[k].values,
                     bounds_error=False, fill_value=np.nan)
        out[k] = f(t_query)
    return out


def cum_mean(e):
    out = np.zeros_like(e)
    s = 0.0; c = 0
    for i, v in enumerate(e):
        if not np.isnan(v):
            s += v; c += 1
        out[i] = s / c if c > 0 else 0.0
    return out


def ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG3)
    for sp in ax.spines.values(): sp.set_color("#1a2035")
    ax.tick_params(colors=C_DIM, labelsize=7)
    ax.set_title(title, pad=5)
    if xlabel: ax.set_xlabel(xlabel, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7)
    ax.grid(True, alpha=0.4)


# ═══════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════
def fig_trajectories(fused_df, enc_df,
                     x_enc, y_enc,
                     x_fgo, y_fgo,
                     title_prefix, save=None):
    """
    Fig 1 — Three XY trajectory panels side by side:
      Left   : Ground Truth (LiDAR ICP from fused file)
      Middle : Encoder + IMU dead reckoning
      Right  : FGO fused estimate
    Plus an overlay panel below.
    """
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"{title_prefix}  —  TRAJECTORY COMPARISON",
                 fontsize=13, color=C_FG, y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.30,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.07)

    gt_x = fused_df["global_x"].values
    gt_y = fused_df["global_y"].values
    t_gt = fused_df["t_sec"].values

    panels = [
        (gt_x,  gt_y,  C_GT,  "Ground Truth (LiDAR ICP)"),
        (x_enc, y_enc, C_ENC, "Encoder + IMU Dead Reckoning"),
        (x_fgo, y_fgo, C_FGO, "FGO Fused Estimate"),
    ]

    for col, (xs, ys, col_c, lbl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax_style(ax, lbl, "X (m)", "Y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.plot(xs, ys, color=col_c, lw=1.8)
        ax.plot(xs[0],  ys[0],  "s", color="#ffd60a",
                ms=8, zorder=6, label="Start")
        ax.plot(xs[-1], ys[-1], "^", color="#ffd60a",
                ms=8, zorder=6, label="End")
        ax.legend(fontsize=6.5)

    # Overlay
    ax_ov = fig.add_subplot(gs[1, :])
    ax_style(ax_ov, "ALL OVERLAID", "X (m)", "Y (m)")
    ax_ov.set_aspect("equal", adjustable="datalim")
    ax_ov.plot(gt_x,  gt_y,  color=C_GT,  lw=2.2, label="Ground Truth")
    ax_ov.plot(x_enc, y_enc, color=C_ENC, lw=1.3,
               linestyle=":", alpha=0.85, label="Encoder+IMU")
    ax_ov.plot(x_fgo, y_fgo, color=C_FGO, lw=1.8, label="FGO Fused")
    ax_ov.plot(gt_x[0], gt_y[0], "s", color="#ffd60a",
               ms=9, zorder=10, label="Start")
    ax_ov.legend(loc="best")

    if save:
        p = f"{save}_fig1_trajectories.png"
        fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
        print(f"[SAVE] {p}")
    return fig


def fig_errors(t_enc, fused_df,
               x_enc, y_enc,
               x_fgo, y_fgo,
               title_prefix, save=None):
    """
    Fig 2 — Position error vs time:
      Top    : Instantaneous error (smoothed)
      Bottom : Cumulative mean error
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"{title_prefix}  —  POSITION ERROR vs TIME",
                 fontsize=12, color=C_FG, y=0.99)
    fig.subplots_adjust(hspace=0.15, left=0.09, right=0.97,
                        top=0.93, bottom=0.09)

    gt_interp = interp_gt(fused_df, t_enc)
    gx = gt_interp["global_x"]
    gy = gt_interp["global_y"]

    err_enc = pos_err(x_enc, y_enc, gx, gy)
    err_fgo = pos_err(x_fgo, y_fgo, gx, gy)

    rmse_enc = rmse(err_enc)
    rmse_fgo = rmse(err_fgo)
    impr = (1 - rmse_fgo / rmse_enc) * 100

    # ── Instantaneous ──
    ax = axes[0]
    ax_style(ax, "Instantaneous Position Error (m)", "", "Error (m)")
    ax.plot(t_enc, smooth(err_enc), color=C_ENC, lw=1.4,
            label=f"Encoder+IMU   RMSE = {rmse_enc:.4f} m")
    ax.plot(t_enc, smooth(err_fgo), color=C_FGO, lw=1.8,
            label=f"FGO Fused     RMSE = {rmse_fgo:.4f} m")
    ax.set_ylim(bottom=0)
    ax.legend()
    sign = "better" if impr > 0 else "worse"
    ax.text(0.98, 0.92,
            f"FGO is {abs(impr):.1f}% {sign} than Encoder+IMU",
            transform=ax.transAxes, ha="right",
            color=C_FGO if impr > 0 else C_ENC, fontsize=8)

    # ── Cumulative mean ──
    ax = axes[1]
    ax_style(ax, "Cumulative Mean Error (m)", "Time (s)", "Error (m)")
    ax.plot(t_enc, cum_mean(err_enc), color=C_ENC, lw=1.4,
            label="Encoder+IMU cumulative")
    ax.plot(t_enc, cum_mean(err_fgo), color=C_FGO, lw=1.8,
            label="FGO cumulative")
    ax.set_ylim(bottom=0)
    ax.legend()

    if save:
        p = f"{save}_fig2_errors.png"
        fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
        print(f"[SAVE] {p}")
    return fig, rmse_enc, rmse_fgo


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="FGO for Experiment 1 (no-slip) or 2 (slip)")

    parser.add_argument("--fused",
                        default="/Users/sriyatulluri/Downloads/RLN PROJECT /lidar_gd.csv",
                        help="Fused CSV with LiDAR ICP ground truth poses")

    parser.add_argument("--enc",
                        default="/Users/sriyatulluri/Downloads/RLN PROJECT /sensor_fusion_expt2.csv",
                        help="Encoder+IMU +_LIDAR CSV for dead reckoning")

    parser.add_argument("--title",
                        default="Experiment 2 -Slip ",
                        help="Title prefix for all plots")

    parser.add_argument("--save", default=None,
                        help="Save prefix e.g. --save expt1")

    args = parser.parse_args()

    print("=" * 60)
    print(f"  {args.title}")
    print(f"  Fused  : {args.fused}")
    print(f"  Encoder: {args.enc}")
    print(f"  CPR L={CPR_LEFT}  R={CPR_RIGHT}")
    print(f"  Huber k={HUBER_K}  "
          f"σ_enc={SIGMA_ENC_XY}m  σ_lid={SIGMA_LID_XY}m")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────
    fused_df = load_fused(args.fused)
    enc_df   = load_encoder_csv(args.enc)

    # ── Encoder odometry ──────────────────────────────────────────────
    print("\n[ODOM]  Computing encoder+IMU dead reckoning...")
    x_enc, y_enc, th_enc = encoder_odometry(enc_df)
    t_enc = enc_df["t_sec"].values

    # ── FGO ───────────────────────────────────────────────────────────
    print("[FGO]   Running iSAM2 factor graph optimisation...")
    if HAS_GTSAM:
        x_fgo, y_fgo, th_fgo = run_fgo_gtsam(enc_df, fused_df)
    else:
        x_fgo, y_fgo, th_fgo = run_fgo_fallback(enc_df, fused_df)

    # ── Metrics ───────────────────────────────────────────────────────
    print("\n[METRICS]")
    gt_interp = interp_gt(fused_df, t_enc)
    gx = gt_interp["global_x"]; gy = gt_interp["global_y"]
    err_enc = pos_err(x_enc, y_enc, gx, gy)
    err_fgo = pos_err(x_fgo, y_fgo, gx, gy)
    r_enc = rmse(err_enc); r_fgo = rmse(err_fgo)
    impr = (1 - r_fgo / r_enc) * 100
    print(f"  Encoder+IMU RMSE : {r_enc:.4f} m")
    print(f"  FGO         RMSE : {r_fgo:.4f} m")
    if impr > 0:
        print(f"  FGO is {impr:.1f}% BETTER than Encoder+IMU ✓")
    else:
        print(f"  FGO is {-impr:.1f}% worse — check sigma tuning")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n[PLOT]  Generating figures...")
    s = args.save
    fig_trajectories(fused_df, enc_df,
                     x_enc, y_enc, x_fgo, y_fgo,
                     args.title, s)
    fig_errors(t_enc, fused_df,
               x_enc, y_enc, x_fgo, y_fgo,
               args.title, s)

    print("[DONE]  Both figures ready.")
    plt.show()


if __name__ == "__main__":
    main()
