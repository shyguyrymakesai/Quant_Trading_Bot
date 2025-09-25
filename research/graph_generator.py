import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_COLS = [
    "symbol",
    "tf",
    "WFV_Avg_OOS_Gross_Sharpe",
    "WFV_Avg_OOS_Net_Sharpe",
    "WFV_OOS_Gross_Sharpe_Std",
    "WFV_OOS_Net_Sharpe_Std",
    "WFV_OOS_MaxDD",
    "WFV_OOS_Trades",
    "WFV_OOS_TradesPerMonth",
    "WFV_OOS_Turnover",
    "WFV_OOS_FeesPctGrossPnL",
    "WFV_OOS_Exposure",
    "WFV_OOS_HitRate",
    "WFV_OOS_AvgHold",
]


def _maybe_percent(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.max(skipna=True) is not None and s.max(skipna=True) <= 1.0:
        return s * 100.0
    return s


def _label(df: pd.DataFrame) -> pd.Series:
    # Combine symbol/tf for x-axis labels
    sym = df["symbol"].astype(str).str.replace("USDT", "", regex=False)
    return (sym + " " + df["tf"].astype(str)).values


def plot_from_summary(csv_path: str, outdir: str = "research/reports/plots"):
    if not os.path.isabs(csv_path):
        # Resolve relative to CWD; if missing, also try relative to this script
        if not os.path.exists(csv_path):
            alt = Path(__file__).resolve().parent / "reports" / Path(csv_path).name
            if alt.exists():
                csv_path = str(alt)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in SUMMARY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    os.makedirs(outdir, exist_ok=True)
    labels = _label(df)

    # 1) Sharpe (Gross vs Net) with std error bars
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        df["WFV_Avg_OOS_Gross_Sharpe"],
        width,
        label="Gross",
        yerr=pd.to_numeric(df["WFV_OOS_Gross_Sharpe_Std"], errors="coerce"),
        capsize=4,
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        df["WFV_Avg_OOS_Net_Sharpe"],
        width,
        label="Net",
        yerr=pd.to_numeric(df["WFV_OOS_Net_Sharpe_Std"], errors="coerce"),
        capsize=4,
        color="#F58518",
    )
    ax.set_title("WFV OOS Sharpe (Gross vs Net)")
    ax.set_xticks(x, labels, rotation=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(outdir) / "wfv_sharpe.png", dpi=150)

    # 2) Max Drawdown (%)
    dd = _maybe_percent(df["WFV_OOS_MaxDD"])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(labels, dd, color="#E45756")
    ax.set_title("WFV OOS Max Drawdown (%)")
    ax.set_ylabel("%")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "wfv_maxdd.png", dpi=150)

    # 3) Trades per month and Turnover
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(
        labels,
        pd.to_numeric(df["WFV_OOS_TradesPerMonth"], errors="coerce"),
        color="#72B7B2",
    )
    ax1.set_ylabel("Trades / Month", color="#225B54")
    ax1.set_title("WFV OOS Trades/Month and Turnover")
    ax1.tick_params(axis="y", labelcolor="#225B54")

    ax2 = ax1.twinx()
    ax2.plot(
        labels,
        pd.to_numeric(df["WFV_OOS_Turnover"], errors="coerce"),
        color="#FF9DA6",
        marker="o",
    )
    ax2.set_ylabel("Turnover (× equity)", color="#8B1E3F")
    ax2.tick_params(axis="y", labelcolor="#8B1E3F")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "wfv_trades_turnover.png", dpi=150)

    # 4) Fees as % of Gross PnL
    fees_pct = _maybe_percent(df["WFV_OOS_FeesPctGrossPnL"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, fees_pct, color="#54A24B")
    ax.set_title("WFV OOS Fees as % of Gross PnL")
    ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "wfv_fees_pct_gross.png", dpi=150)

    # 5) Exposure % and Hit Rate %
    exposure = _maybe_percent(df["WFV_OOS_Exposure"])
    hit = _maybe_percent(df["WFV_OOS_HitRate"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, exposure, alpha=0.7, label="Exposure %", color="#4C78A8")
    ax.bar(labels, hit, alpha=0.7, label="Hit Rate %", color="#F58518")
    ax.set_title("WFV OOS Exposure and Hit Rate (%)")
    ax.set_ylabel("%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(outdir) / "wfv_exposure_hit.png", dpi=150)

    # 6) Avg holding time (hours)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(
        labels, pd.to_numeric(df["WFV_OOS_AvgHold"], errors="coerce"), color="#B279A2"
    )
    ax.set_title("WFV OOS Average Holding Time (hours)")
    ax.set_ylabel("Hours")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "wfv_avg_hold.png", dpi=150)

    print(f"Saved plots to: {outdir}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_csv = script_dir / "reports" / "wfv_eth_1h_fee_reduction.csv"
    default_outdir = script_dir / "reports" / "plots"

    parser = argparse.ArgumentParser(description="Plot WFV summary charts from CSV.")
    parser.add_argument(
        "--csv",
        default=str(default_csv),
        help="Path to WFV summary CSV",
    )
    parser.add_argument(
        "--outdir",
        default=str(default_outdir),
        help="Directory to save plots",
    )
    args = parser.parse_args()
    plot_from_summary(args.csv, args.outdir)


if __name__ == "__main__":
    main()
