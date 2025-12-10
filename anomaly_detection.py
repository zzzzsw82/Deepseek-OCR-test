# """Detect anomalies in OCR-derived CSVs using an Isolation Forest (unsupervised)."""

# from __future__ import annotations

# import argparse
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest


# def load_numeric_df(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     numeric = df.select_dtypes(include=[np.number]).copy()
#     if numeric.empty:
#         raise ValueError("No numeric columns found for anomaly detection.")
#     numeric = numeric.fillna(numeric.median())
#     return df, numeric


# def detect_anomalies(
#     numeric_df: pd.DataFrame,
#     *,
#     contamination: float,
#     random_state: Optional[int],
# ) -> tuple[np.ndarray, np.ndarray]:
#     model = IsolationForest(
#         n_estimators=200,
#         contamination=contamination,
#         random_state=random_state,
#     )
#     model.fit(numeric_df)
#     scores = model.decision_function(numeric_df)
#     flags = model.predict(numeric_df) == -1  # True = anomaly
#     return scores, flags


# def run(
#     input_csv: Path,
#     output_csv: Path,
#     *,
#     contamination: float,
#     random_state: Optional[int],
# ) -> Path:
#     full_df, numeric_df = load_numeric_df(input_csv)
#     scores, flags = detect_anomalies(
#         numeric_df,
#         contamination=contamination,
#         random_state=random_state,
#     )

#     full_df["anomaly_score"] = scores
#     full_df["anomaly_flag"] = flags
#     anomalies = full_df[full_df["anomaly_flag"]]
#     anomalies.to_csv(output_csv, index=False, encoding="utf-8")
#     return output_csv


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Mark anomalous rows in an OCR CSV using Isolation Forest.")
#     parser.add_argument("--input-csv", required=True, type=Path, help="CSV file produced after OCR/格式化。")
#     parser.add_argument(
#         "--output-csv",
#         type=Path,
#         default=Path("anomalies.csv"),
#         help="Where to save flagged anomalies.",
#     )
#     parser.add_argument(
#         "--contamination",
#         type=float,
#         default=0.05,
#         help="Expected anomaly proportion (0-0.5) for Isolation Forest.",
#     )
#     parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     output_path = run(
#         input_csv=args.input_csv,
#         output_csv=args.output_csv,
#         contamination=args.contamination,
#         random_state=args.seed,
#     )
#     print(f"Anomalies saved to: {output_path}")


# if __name__ == "__main__":
#     main()
