"""Core machine learning utilities for training and loading the Random Forest model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from django.conf import settings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data_from_directory(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all CSV files from a directory.

    Args:
        data_dir: Local directory containing CSV files.

    Returns:
        Concatenated DataFrame with all rows from the CSV files.

    Raises:
        FileNotFoundError: If the directory does not exist or no CSV files are found.
        KeyError: If the required target column is missing.
        ValueError: If the target column cannot be converted to numeric.
    """

    directory = Path(data_dir).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"La ruta especificada no es un directorio válido: {directory}")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en el directorio: {directory}")

    data_frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        data_frames.append(pd.read_csv(csv_file))

    data = pd.concat(data_frames, ignore_index=True)

    if "PUNT_GLOBAL" not in data.columns:
        raise KeyError("La columna objetivo 'PUNT_GLOBAL' no está presente en los datos.")

    # Aseguramos que el objetivo sea numérico. Si falla, se lanza un error claro.
    data["PUNT_GLOBAL"] = pd.to_numeric(data["PUNT_GLOBAL"], errors="raise")

    if data.empty:
        raise ValueError("Los archivos CSV están vacíos; no hay datos para entrenar.")

    return data


def _build_preprocessor(feature_frame: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Crea el preprocesador separando columnas numéricas y categóricas."""
    numeric_features = [
        col for col in feature_frame.columns if pd.api.types.is_numeric_dtype(feature_frame[col])
    ]
    categorical_features = [
        col
        for col in feature_frame.columns
        if pd.api.types.is_object_dtype(feature_frame[col])
        or pd.api.types.is_categorical_dtype(feature_frame[col])
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No se encontraron columnas de características para entrenar el modelo.")

    transformers = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_features, categorical_features


def train_random_forest(data_dir: str) -> Dict[str, Any]:
    """Train a Random Forest regressor using CSV files in a directory."""
    data = load_data_from_directory(data_dir)

    y = data["PUNT_GLOBAL"]
    X = data.drop(columns=["PUNT_GLOBAL"])

    preprocessor, numeric_features, categorical_features = _build_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(random_state=42, n_estimators=200),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions, squared=False),
        "mape": mean_absolute_percentage_error(y_test, predictions),
    }

    storage_dir = Path(getattr(settings, "MODEL_STORAGE_DIR", Path("models")))
    storage_dir.mkdir(parents=True, exist_ok=True)

    model_path = storage_dir / "random_forest_punt_global.pkl"
    joblib.dump(pipeline, model_path)

    metrics_path = storage_dir / "metrics.json"
    serializable_metrics = {
        "metrics": metrics,
        "n_registros": int(len(data)),
        "n_variables": int(X.shape[1]),
        "columnas_numericas": numeric_features,
        "columnas_categoricas": categorical_features,
    }
    with open(metrics_path, "w", encoding="utf-8") as metric_file:
        json.dump(serializable_metrics, metric_file, ensure_ascii=False, indent=4)

    return {
        "metrics": metrics,
        "n_registros": int(len(data)),
        "n_variables": int(X.shape[1]),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "columnas_numericas": numeric_features,
        "columnas_categoricas": categorical_features,
    }
