import matplotlib.pyplot as plt
import pandas as pd

from config_5 import PROCESSED_DIR, FIG_DIR


def load_summary():
    csv_path = PROCESSED_DIR / "summary_all_alloys.csv"
    df = pd.read_csv(csv_path)
    return df


def clean_success_rows(df):
    """
    Keep rows that have valid computed results.
    """
    if "status" in df.columns:
        df = df[df["status"] != "failed"].copy()

    needed = ["solute_element", "actual_solute_fraction", "B_over_G", "Tm_linear_K"]
    for c in needed:
        df = df[df[c].notna()]
    return df


def plot_bg_vs_composition(df):
    """
    Figure 1:
    Single figure with B/G vs composition for Ta, Nb, Mo alloys.
    """
    plt.figure(figsize=(8, 6))

    for solute in ["Ta", "Nb", "Mo"]:
        sub = df[df["solute_element"] == solute].copy()
        sub = sub.sort_values("actual_solute_fraction")
        plt.plot(
            sub["actual_solute_fraction"],
            sub["B_over_G"],
            marker="o",
            label=f"W-{solute}"
        )

    plt.xlabel("Solute fraction x")
    plt.ylabel("B/G")
    plt.title("B/G vs composition for W-alloys")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "BG_vs_composition.png"
    plt.savefig(out, dpi=300)
    plt.show()

    return out


def plot_bg_vs_tm(df):
    """
    Figure 2:
    Single figure with B/G vs Tm for all alloy systems.
    """
    plt.figure(figsize=(8, 6))

    for solute in ["Ta", "Nb", "Mo"]:
        sub = df[df["solute_element"] == solute].copy()
        sub = sub.sort_values("Tm_linear_K")
        plt.plot(
            sub["Tm_linear_K"],
            sub["B_over_G"],
            marker="o",
            label=f"W-{solute}"
        )

    plt.xlabel("Linear melting point Tm (K)")
    plt.ylabel("B/G")
    plt.title("B/G vs linear-rule melting point for W-alloys")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "BG_vs_Tm.png"
    plt.savefig(out, dpi=300)
    plt.show()

    return out


def main():
    df = load_summary()
    df = clean_success_rows(df)

    out1 = plot_bg_vs_composition(df)
    out2 = plot_bg_vs_tm(df)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()