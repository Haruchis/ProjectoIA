"""Core machine learning utilities for training and prediction."""
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

MODEL_FILENAME = "random_forest_punt_global.pkl"
METRICS_FILENAME = "metrics.json"


def load_data_from_directory(data_dir: str, return_progress: bool = False) -> pd.DataFrame:
    """Load and concatenate tabular files from a directory.

    Supports CSV and Excel files (``.csv``, ``.xlsx``, ``.xls``). Minimal validation
    is performed: the directory must exist, at least one supported file must be
    present, the target column ``PUNT_GLOBAL`` must exist, and rows with missing
    values are removed. When ``return_progress`` is True, progress messages are
    returned alongside the dataframe.
    """

    directory = Path(data_dir).expanduser().resolve()
    progress: List[str] = []

    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"La ruta especificada no es un directorio válido: {directory}")

    supported_patterns = ["*.csv", "*.xlsx", "*.xls"]
    files: List[Path] = []
    for pattern in supported_patterns:
        files.extend(sorted(directory.glob(pattern)))

    if not files:
        raise FileNotFoundError(
            "No se encontraron archivos CSV o Excel en el directorio proporcionado."
        )

    progress.append(f"Archivos encontrados: {len(files)}")
    data_frames: List[pd.DataFrame] = []

    for index, file_path in enumerate(files, start=1):
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        progress.append(f"({index}/{len(files)}) Cargado {file_path.name} con {len(df)} filas")
        data_frames.append(df)

    data = pd.concat(data_frames, ignore_index=True)
    progress.append(f"Total de filas combinadas: {len(data)}")

    if data.empty:
        raise ValueError("Los archivos CSV están vacíos; no hay datos para entrenar.")

    if "PUNT_GLOBAL" not in data.columns:
        raise KeyError("La columna objetivo 'PUNT_GLOBAL' no está presente en los datos.")

    # Conversión básica a numérico; si falla se informa al usuario.
    data["PUNT_GLOBAL"] = pd.to_numeric(data["PUNT_GLOBAL"], errors="raise")

    # Eliminamos filas con valores faltantes para evitar fallos en el pipeline.
    before_drop = len(data)
    data = data.dropna()
    removed = before_drop - len(data)
    if removed:
        progress.append(f"Filas eliminadas por valores faltantes: {removed}")
    progress.append(f"Filas finales para entrenamiento: {len(data)}")

    if data.empty:
        raise ValueError("No hay filas válidas después de eliminar valores faltantes.")

    if return_progress:
        return data, progress

    return data


def _split_feature_types(feature_frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical columns based on dtypes."""

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

    return numeric_features, categorical_features


def _harmonize_feature_dtypes(features: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
    """Ensure categorical columns are strings to avoid mixed-type issues in encoders."""

    harmonized = features.copy()
    if categorical_features:
        harmonized[categorical_features] = harmonized[categorical_features].astype(str)
    return harmonized


def _build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Create the ColumnTransformer for preprocessing features."""

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

    return ColumnTransformer(transformers=transformers)


def _metrics_dict(y_true, y_pred) -> Dict[str, float]:
    """Compute evaluation metrics for regression."""

    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def train_random_forest(data_dir: str) -> Dict[str, Any]:
    """Train a Random Forest model and persist artifacts to disk."""

    data, progress = load_data_from_directory(data_dir, return_progress=True)

    y = data["PUNT_GLOBAL"]
    X = data.drop(columns=["PUNT_GLOBAL"]).copy()

    # Intentamos convertir columnas de objetos numéricos a valores numéricos reales
    # para aprovechar el escalado. Si toda la columna puede convertirse, se mantiene
    # como numérica; de lo contrario, quedará como categórica.
    for column in X.columns:
        coerced = pd.to_numeric(X[column], errors="coerce")
        if coerced.notna().sum() == len(X[column]):
            X[column] = coerced

    numeric_features, categorical_features = _split_feature_types(X)
    X = _harmonize_feature_dtypes(X, categorical_features)
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    random_state=42,
                    n_estimators=200,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = _metrics_dict(y_test, predictions)

    progress.extend(
        [
            "Entrenamiento finalizado.",
            "Generando predicciones de validación.",
            "Calculando métricas y guardando artefactos.",
        ]
    )

    storage_dir = Path(getattr(settings, "MODEL_DIR", Path("models")))
    storage_dir.mkdir(parents=True, exist_ok=True)

    model_path = storage_dir / MODEL_FILENAME
    metrics_path = storage_dir / METRICS_FILENAME

    joblib.dump(pipeline, model_path)

    serializable_metrics = {
        "metrics": metrics,
        "n_registros": int(len(data)),
        "n_variables": int(X.shape[1]),
        "columnas_numericas": numeric_features,
        "columnas_categoricas": categorical_features,
        "progreso": progress,
    }
    with open(metrics_path, "w", encoding="utf-8") as metric_file:
        json.dump(serializable_metrics, metric_file, ensure_ascii=False, indent=4)

    return {
        **serializable_metrics,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }


def load_trained_model() -> Pipeline:
    """Load the persisted model pipeline."""

    model_path = Path(getattr(settings, "MODEL_DIR", Path("models"))) / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            "No se encontró un modelo entrenado. Entrene el modelo antes de predecir."
        )

    return joblib.load(model_path)


def predict_from_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for a given dataframe using the saved model."""

    if dataframe.empty:
        raise ValueError("El archivo CSV no contiene datos para predecir.")

    # En caso de que el archivo traiga la columna objetivo por error, la removemos.
    dataframe = dataframe.copy()
    if "PUNT_GLOBAL" in dataframe.columns:
        dataframe = dataframe.drop(columns=["PUNT_GLOBAL"])

    # Repetimos la lógica de armonización de tipos para evitar errores de codificación.
    for column in dataframe.columns:
        coerced = pd.to_numeric(dataframe[column], errors="coerce")
        if coerced.notna().sum() == len(dataframe[column]):
            dataframe[column] = coerced
    _, categorical_features = _split_feature_types(dataframe)
    dataframe = _harmonize_feature_dtypes(dataframe, categorical_features)

    pipeline = load_trained_model()
    predictions = pipeline.predict(dataframe)

    result_df = dataframe.copy()
    result_df["PREDICCION_PUNT_GLOBAL"] = predictions
    return result_df
