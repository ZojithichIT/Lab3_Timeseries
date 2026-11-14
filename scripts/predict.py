"""
Standalone inference and artifact verification script for Lab 3 Time Series Forecasting.

Loads saved .joblib models from Lab3_TS/models/, verifies test-set & trailing-window
accuracy, and generates recursive multi-step future forecasts beyond the latest timestamp.

Outputs:
- logs/inference.log (append mode)
- logs/latest_predictions.csv
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure Lab3_TS package root is on sys.path before joblib.load unpickles scripts.models
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.config import Config
from scripts.data_loader import DataLoader
from scripts.evaluation import evaluate_forecast
from scripts.feature_engineering import DatasetSplits, TimeSeriesFeatureBuilder
import scripts.models  # noqa: F401 - required for joblib unpickling of forecaster classes


def setup_inference_logger(log_path: Path) -> logging.Logger:
    """Configure logger in append mode for inference runs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n")

    logger = logging.getLogger("Lab3InferencePipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_feature_splits_cache(series: pd.Series) -> Dict[str, DatasetSplits]:
    """Build and cache supervised Train/Val/Test splits for each Lab 3 feature design."""
    df_sel = TimeSeriesFeatureBuilder.create_selective_lags(series, lags=Config.SELECTIVE_LAGS)
    df_dense24 = TimeSeriesFeatureBuilder.create_lag_features(series, p=Config.DEFAULT_LAGS)
    df_roll = TimeSeriesFeatureBuilder.create_lags_and_rolling_features(
        series, lags=Config.SELECTIVE_LAGS, windows=Config.ROLLING_WINDOWS
    )
    return {
        "selective_29": TimeSeriesFeatureBuilder.time_based_split(df_sel, ratios=Config.SPLIT_RATIOS, export_dir=None),
        "dense_24": TimeSeriesFeatureBuilder.time_based_split(df_dense24, ratios=Config.SPLIT_RATIOS, export_dir=None),
        "rolling_35": TimeSeriesFeatureBuilder.time_based_split(df_roll, ratios=Config.SPLIT_RATIOS, export_dir=None),
    }


def match_splits_for_features(
    feature_names: List[str],
    splits_cache: Dict[str, DatasetSplits],
) -> DatasetSplits:
    """Select the DatasetSplits object matching a saved model's feature_names_."""
    if any(col.startswith("rolling_") for col in feature_names):
        return splits_cache["rolling_35"]
    dense_24_cols = [f"lag_{i}" for i in range(1, Config.DEFAULT_LAGS + 1)]
    if list(feature_names) == dense_24_cols:
        return splits_cache["dense_24"]
    return splits_cache["selective_29"]


def resolve_selected_models(model_arg: str) -> List[Tuple[str, Path]]:
    """Resolve CLI --model argument to one or more (display_name, artifact_path) pairs."""
    reverse_map = {fname: m_name for m_name, fname in Config.MODEL_FILENAMES.items()}

    if model_arg.lower() == "all":
        pairs: List[Tuple[str, Path]] = []
        for m_name, fname in Config.MODEL_FILENAMES.items():
            p = Config.MODELS_DIR / fname
            if p.exists():
                pairs.append((m_name, p))
        if not pairs:
            raise FileNotFoundError(f"No saved .joblib models found in {Config.MODELS_DIR}")
        return pairs

    # Match by display name
    if model_arg in Config.MODEL_FILENAMES:
        path = Config.MODELS_DIR / Config.MODEL_FILENAMES[model_arg]
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return [(model_arg, path)]

    # Match by filename (with or without .joblib) or direct path
    candidate = Path(model_arg)
    if not candidate.exists():
        fname = model_arg if model_arg.endswith(".joblib") else f"{model_arg}.joblib"
        candidate = Config.MODELS_DIR / fname
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not resolve model '{model_arg}'. Available files in {Config.MODELS_DIR}: "
            f"{[p.name for p in Config.MODELS_DIR.glob('*.joblib')]}"
        )
    display_name = reverse_map.get(candidate.name, candidate.stem)
    return [(display_name, candidate)]


def build_step_features(history: List[float], feature_names: List[str]) -> pd.DataFrame:
    """Compute one step of lag and rolling features dynamically from history."""
    row: Dict[str, float] = {}
    for col in feature_names:
        if col.startswith("lag_"):
            k = int(col.split("_")[1])
            row[col] = float(history[-k])
        elif col.startswith("rolling_mean_"):
            w = int(col.split("_")[-1])
            row[col] = float(np.mean(history[-w:]))
        elif col.startswith("rolling_std_"):
            w = int(col.split("_")[-1])
            row[col] = float(np.std(history[-w:], ddof=1))
        else:
            raise ValueError(f"Unsupported feature column during recursive inference: {col}")
    return pd.DataFrame([row], columns=feature_names)


def forecast_future_recursive(
    forecaster: Any,
    series: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    """Forecast future hours autoregressively beyond the end of the observed series."""
    feature_names = list(forecaster.feature_names_)
    history = [float(v) for v in series.values]
    last_ts = series.index.max()

    records: List[Dict[str, Any]] = []
    for step in range(1, horizon + 1):
        X_step = build_step_features(history, feature_names)
        y_hat = float(forecaster.predict(X_step)[0])
        next_ts = last_ts + pd.Timedelta(hours=step)
        records.append({
            "Datetime": str(next_ts),
            "horizon_step": step,
            "predicted_AEP_MW": round(y_hat, 4),
        })
        history.append(y_hat)
    return pd.DataFrame(records)


def run_inference(
    model_arg: str = "xgboost_tuned_29lags.joblib",
    eval_hours: int = 168,
    horizon: int = 24,
) -> None:
    """Load saved .joblib models, verify test-set accuracy, and export future forecasts."""
    logger = setup_inference_logger(Config.INFERENCE_LOG_PATH)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 90)
    logger.info(
        f"STARTING LAB 3 STANDALONE INFERENCE PIPELINE "
        f"(model={model_arg}, eval_hours={eval_hours}, horizon={horizon}h)"
    )
    logger.info("=" * 90)

    if not Config.MODELS_DIR.exists():
        raise FileNotFoundError(
            f"Models directory does not exist: {Config.MODELS_DIR}. Run train.py first."
        )

    loader = DataLoader(Config.RAW_DATA_PATH)
    series = loader.load_clean_series()
    logger.info(
        f"Loaded cleaned AEP series: {len(series):,} hours "
        f"({series.index.min()} -> {series.index.max()})"
    )

    splits_cache = build_feature_splits_cache(series)

    # Step 1: Verify all saved .joblib artifacts on Test Set and trailing eval_hours window
    logger.info("-" * 90)
    logger.info("[1/3] Verifying all saved .joblib model artifacts on Test Set & trailing window...")
    loaded_models: Dict[str, Any] = {}
    for m_name, fname in Config.MODEL_FILENAMES.items():
        art_path = Config.MODELS_DIR / fname
        if not art_path.exists():
            logger.warning(f"  -> Missing saved artifact: {art_path}")
            continue
        forecaster = joblib.load(art_path)
        loaded_models[fname] = forecaster

        feature_names = list(forecaster.feature_names_)
        splits = match_splits_for_features(feature_names, splits_cache)
        X_test = splits.X_test[feature_names]
        y_test = splits.y_test
        y_pred_test = forecaster.predict(X_test)

        full_m = evaluate_forecast(y_test, y_pred_test)
        tail_n = min(eval_hours, len(y_test))
        tail_m = evaluate_forecast(y_test.iloc[-tail_n:], y_pred_test[-tail_n:])
        logger.info(
            f"  -> {m_name:<32} [{fname}] | "
            f"Full Test RMSE: {full_m['RMSE']:6.2f} MW (MAE: {full_m['MAE']:6.2f}) | "
            f"Last {tail_n}h RMSE: {tail_m['RMSE']:6.2f} MW (MAE: {tail_m['MAE']:6.2f})"
        )

    # Check registry metadata if present
    if Config.MODEL_REGISTRY_PATH.exists():
        reg = json.loads(Config.MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
        logger.info(
            f"Model registry verified: {reg.get('models_count', 0)} models registered "
            f"(default: {reg.get('default_model', 'N/A')})"
        )

    # Step 2: Run trailing evaluation comparison + recursive future forecast for selected model(s)
    selected_pairs = resolve_selected_models(model_arg)
    logger.info("-" * 90)
    logger.info(
        f"[2/3] Generating {horizon}h recursive forecasts beyond {series.index.max()} "
        f"and {eval_hours}h trailing comparison..."
    )

    output_frames: List[pd.DataFrame] = []
    for display_name, art_path in selected_pairs:
        forecaster = loaded_models.get(art_path.name) or joblib.load(art_path)
        feature_names = list(forecaster.feature_names_)
        splits = match_splits_for_features(feature_names, splits_cache)

        X_test = splits.X_test[feature_names]
        y_test = splits.y_test
        y_pred_test = forecaster.predict(X_test)

        tail_n = min(eval_hours, len(y_test))
        y_tail_true = y_test.iloc[-tail_n:]
        y_tail_pred = y_pred_test[-tail_n:]

        trailing_df = pd.DataFrame({
            "Datetime": [str(ts) for ts in y_tail_true.index],
            "horizon_step": np.nan,
            "actual_AEP_MW": np.round(y_tail_true.values.astype(float), 4),
            "predicted_AEP_MW": np.round(np.asarray(y_tail_pred, dtype=float), 4),
            "abs_error_MW": np.round(np.abs(y_tail_true.values.astype(float) - np.asarray(y_tail_pred, dtype=float)), 4),
            "record_type": "trailing_eval",
            "model_name": display_name,
            "generated_at": generated_at,
        })

        future_df = forecast_future_recursive(forecaster, series, horizon=horizon)
        future_df["actual_AEP_MW"] = np.nan
        future_df["abs_error_MW"] = np.nan
        future_df["record_type"] = "future_forecast"
        future_df["model_name"] = display_name
        future_df["generated_at"] = generated_at
        future_df = future_df[
            [
                "Datetime",
                "horizon_step",
                "actual_AEP_MW",
                "predicted_AEP_MW",
                "abs_error_MW",
                "record_type",
                "model_name",
                "generated_at",
            ]
        ]

        output_frames.extend([trailing_df, future_df])
        logger.info(
            f"  -> {display_name}: Min Forecast = {future_df['predicted_AEP_MW'].min():,.2f} MW, "
            f"Max Forecast = {future_df['predicted_AEP_MW'].max():,.2f} MW, "
            f"Mean = {future_df['predicted_AEP_MW'].mean():,.2f} MW"
        )
        for _, f_row in future_df.iloc[[0, 5, 11, 17, -1] if len(future_df) >= 18 else range(len(future_df))].iterrows():
            logger.info(
                f"     Step h={int(f_row['horizon_step']):02d} ({f_row['Datetime']}) -> "
                f"Predicted AEP_MW: {f_row['predicted_AEP_MW']:,.2f} MW"
            )

    # Step 3: Export predictions CSV
    logger.info("-" * 90)
    logger.info("[3/3] Exporting predictions & trailing evaluation CSV...")
    combined_df = pd.concat(output_frames, ignore_index=True)
    Config.PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(Config.PREDICTIONS_CSV_PATH, index=False, encoding="utf-8")
    logger.info(
        f"-> Saved {len(combined_df):,} rows ({sum(combined_df['record_type'] == 'future_forecast')} future forecasts, "
        f"{sum(combined_df['record_type'] == 'trailing_eval')} trailing evals) to: {Config.PREDICTIONS_CSV_PATH}"
    )
    logger.info("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lab 3 Standalone Inference Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost_tuned_29lags.joblib",
        help="Saved model filename, display name, or 'all' (default: xgboost_tuned_29lags.joblib)",
    )
    parser.add_argument(
        "--eval-hours",
        type=int,
        default=168,
        help="Number of trailing Test Set hours to evaluate (default: 168)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Number of future hours to forecast recursively (default: 24)",
    )
    args = parser.parse_args()

    run_inference(model_arg=args.model, eval_hours=args.eval_hours, horizon=args.horizon)
