import argparse, os
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
    "WFV_OOS_FeesPctGrossPnL",
]


def _maybe_percent(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if pd.notna(s).any() and s.abs().max(skipna=True) <= 1.0:
        return s * 100.0
    return s


def _labels(df: pd.DataFrame):
    return list(
        df["symbol"].astype(str).str.replace("USDT", "", regex=False)
        + " "
        + df["tf"].astype(str)
    )


def plot_summary(summary_csv: str, outdir: str, equity_csv: str | None):
    # resolve path
    if not os.path.isabs(summary_csv) and not os.path.exists(summary_csv):
        alt = Path(__file__).resolve().parent / "reports" / Path(summary_csv).name
        if alt.exists():
            summary_csv = str(alt)
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    missing = [c for c in SUMMARY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    labels = _labels(df)
    x = np.arange(len(df))

    # 1) Sharpe (Gross vs Net) with error bars
    width = 0.35
    gross = pd.to_numeric(df["WFV_Avg_OOS_Gross_Sharpe"], errors="coerce")
    net = pd.to_numeric(df["WFV_Avg_OOS_Net_Sharpe"], errors="coerce")
    gstd = (
        pd.to_numeric(df["WFV_OOS_Gross_Sharpe_Std"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    nstd = (
        pd.to_numeric(df["WFV_OOS_Net_Sharpe_Std"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, gross, width, label="Gross", yerr=gstd, capsize=4)
    ax.bar(x + width / 2, net, width, label="Net", yerr=nstd, capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("WFV OOS Sharpe (Gross vs Net)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "wfv_sharpe.png", dpi=150)

    # 2) Max Drawdown (%)  — use a dot + annotation (cleaner for single point)
    dd = _maybe_percent(df["WFV_OOS_MaxDD"])
    fig, ax = plt.subplots(figsize=(6, 4))
    y = float(dd.iloc[0])
    ax.plot([0], [y], marker="o", markersize=10)
    ax.hlines(0, -0.5, 0.5, linewidth=0.8, color="black")
    ax.set_xlim(-0.5, 0.5)
    pad = max(0.5, abs(y) * 0.4)  # headroom
    ax.set_ylim(y - pad, max(0.5, 0) + pad * 0.2)
    ax.set_xticks([0])
    ax.set_xticklabels([labels[0]])
    ax.set_ylabel("%")
    ax.set_title("WFV OOS Max Drawdown (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        f"{y:.2f}%",
        (0, y),
        textcoords="offset points",
        xytext=(0, -12),
        ha="center",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(Path(outdir) / "wfv_maxdd.png", dpi=150)

    # 3) Fees as % of Gross PnL — also dot + annotation
    fees_pct = _maybe_percent(df["WFV_OOS_FeesPctGrossPnL"])
    fig, ax = plt.subplots(figsize=(6, 4))
    v = float(fees_pct.iloc[0])
    ax.plot([0], [v], marker="o", markersize=10)
    ax.set_xlim(-0.5, 0.5)
    top = max(5.0, v * 1.4)
    ax.set_ylim(0, top)
    ax.set_xticks([0])
    ax.set_xticklabels([labels[0]])
    ax.set_ylabel("%")
    ax.set_title("WFV OOS Fees as % of Gross PnL")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        f"{v:.2f}%",
        (0, v),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(Path(outdir) / "wfv_fees_pct_gross.png", dpi=150)

    # 4) Optional equity + underwater (if provided)
    if equity_csv:
        ecsv = Path(equity_csv)
        if not ecsv.exists():
            # also try sibling to summary CSV
            ecsv = Path(summary_csv).with_name(ecsv.name)
        if ecsv.exists():
            edf = pd.read_csv(ecsv, parse_dates=["timestamp"])
            if {"cum_gross", "cum_net", "drawdown"} <= set(edf.columns):
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(edf["timestamp"], edf["cum_gross"], label="Gross")
                ax.plot(edf["timestamp"], edf["cum_net"], label="Net (after fees)")
                ax.axhline(0, color="black", ls="--", lw=0.8)
                ax.set_title("Equity Curve (Gross vs Net)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out / "equity_curve.png", dpi=150)

                fig, ax = plt.subplots(figsize=(9, 3))
                ax.fill_between(edf["timestamp"], edf["drawdown"], 0, alpha=0.3)
                ax.set_title("Underwater Plot (Drawdown %)")
                fig.tight_layout()
                fig.savefig(out / "underwater.png", dpi=150)

    print(f"Saved plots to: {out}")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default=str(here / "reports" / "wfv_eth_1h_fee_reduction.csv")
    )
    parser.add_argument("--outdir", default=str(here / "reports" / "plots"))
    parser.add_argument(
        "--equity",
        default="",
        help="Optional equity CSV with columns: timestamp,cum_gross,cum_net,drawdown",
    )
    args = parser.parse_args()
    plot_summary(args.csv, args.outdir, args.equity or None)


if __name__ == "__main__":
    main()
