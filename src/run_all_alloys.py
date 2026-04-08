import pandas as pd

from config_5 import (
    SOLUTES,
    COMPOSITIONS,
    RANDOM_SEED,
    USE_MULTIPLE_SEEDS,
    SEED_LIST,
    PROCESSED_DIR,
)
from run_single_case import run_single_case
from io_utils import save_records_csv


def run_all_alloys():
    """
    Batch run all W-Ta / W-Nb / W-Mo compositions.
    """
    all_rows = []

    if USE_MULTIPLE_SEEDS:
        seeds = SEED_LIST
    else:
        seeds = [RANDOM_SEED]

    for solute in SOLUTES:
        for x in COMPOSITIONS:
            for seed in seeds:
                try:
                    result = run_single_case(
                        solute_element=solute,
                        nominal_solute_fraction=x,
                        seed=seed,
                    )
                    summary = result["summary"]
                    summary["status"] = "success"
                    print(summary.keys())
                    all_rows.append(summary)
                except Exception as e:
                    row = {
                        "base_element": "W",
                        "solute_element": solute,
                        "nominal_solute_fraction": x,
                        "seed": seed,
                        "status": "failed",
                        "error_message": str(e),
                    }
                    all_rows.append(row)
                    print(f"[FAILED] {solute}, x={x:.2f}, seed={seed}")
                    print(str(e))

    df = pd.DataFrame(all_rows)
    summary_csv = PROCESSED_DIR / "summary_all_alloys.csv"
    df.to_csv(summary_csv, index=False)

    # 1. 过滤失败的（如果有）
    if "status" in df.columns:
        df_ok = df[df["status"] == "success"].copy()
    else:
        df_ok = df.copy()

    agg = df_ok.groupby(
        ["solute_element", "nominal_solute_fraction"],
        as_index=False
    ).agg({
        "B_GPa": ["mean", "std", "count"],
        "G_GPa": ["mean", "std", "count"],
        "B_over_G": ["mean", "std", "count"],
        "Tm_linear_K": ["mean"],
    })

    agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    agg_csv = PROCESSED_DIR / "summary_all_alloys_aggregated.csv"
    agg.to_csv(agg_csv, index=False)

    print(f"Aggregated results saved to: {agg_csv}")

    print("=" * 80)
    print(f"Batch completed. Summary saved to: {summary_csv}")
    return df


if __name__ == "__main__":
    run_all_alloys()
    