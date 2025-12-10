codex/create-django-project-for-random-forest-model-qz73b2
"""Core machine learning utilities for training and prediction."""

"""Core machine learning utilities for training and loading the Random Forest model."""
 main
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

codex/create-django-project-for-random-forest-model-qz73b2
MODEL_FILENAME = "random_forest_punt_global.pkl"
METRICS_FILENAME = "metrics.json"

main

def load_data_from_directory(data_dir: str) -> pd.DataFrame:
 

    directory = Path(data_dir).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"La ruta especificada no es un directorio válido: {directory}")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en el directorio: {directory}")

 codex/create-django-project-for-random-forest-model-qz73b2
    data_frames: List[pd.DataFrame] = [pd.read_csv(csv_file) for csv_file in csv_files]
    data = pd.concat(data_frames, ignore_index=True)

    if data.empty:
        raise ValueError("Los archivos CSV están vacíos; no hay datos para entrenar.")

    data_frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        data_frames.append(pd.read_csv(csv_file))

    data = pd.concat(data_frames, ignore_index=True)
main

    if "PUNT_GLOBAL" not in data.columns:
        raise KeyError("La columna objetivo 'PUNT_GLOBAL' no está presente en los datos.")

codex/create-django-project-for-random-forest-model-qz73b2
    # Conversión básica a numérico; si falla se informa al usuario.
    data["PUNT_GLOBAL"] = pd.to_numeric(data["PUNT_GLOBAL"], errors="raise")

    # Eliminamos filas con valores faltantes para evitar fallos en el pipeline.
    data = data.dropna()
    if data.empty:
        raise ValueError("No hay filas válidas después de eliminar valores faltantes.")

    # Aseguramos que el objetivo sea numérico. Si falla, se lanza un error claro.
    data["PUNT_GLOBAL"] = pd.to_numeric(data["PUNT_GLOBAL"], errors="raise")

    if data.empty:
        raise ValueError("Los archivos CSV están vacíos; no hay datos para entrenar.")
 main

    return data


 codex/create-django-project-for-random-forest-model-qz73b2
def _split_feature_types(feature_frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical columns based on dtypes."""

def _build_preprocessor(feature_frame: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Crea el preprocesador separando columnas numéricas y categóricas."""
 main
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

 codex/create-django-project-for-random-forest-model-qz73b2
    return numeric_features, categorical_features


def _build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Create the ColumnTransformer for preprocessing features."""

 main
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

 codex/create-django-project-for-random-forest-model-qz73b2
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

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_features, categorical_features


def train_random_forest(data_dir: str) -> Dict[str, Any]:
    """Train a Random Forest regressor using CSV files in a directory."""
 main
    data = load_data_from_directory(data_dir)

    y = data["PUNT_GLOBAL"]
    X = data.drop(columns=["PUNT_GLOBAL"])

codex/create-django-project-for-random-forest-model-qz73b2
    numeric_features, categorical_features = _split_feature_types(X)
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    preprocessor, numeric_features, categorical_features = _build_preprocessor(X)
 main

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
 codex/create-django-project-for-random-forest-model-qz73b2
                RandomForestRegressor(
                    random_state=42,
                    n_estimators=200,
                ),

                RandomForestRegressor(random_state=42, n_estimators=200),
 main
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 codex/create-django-project-for-random-forest-model-qz73b2
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = _metrics_dict(y_test, predictions)

    storage_dir = Path(getattr(settings, "MODEL_DIR", Path("models")))
    storage_dir.mkdir(parents=True, exist_ok=True)

    model_path = storage_dir / MODEL_FILENAME
    metrics_path = storage_dir / METRICS_FILENAME

    joblib.dump(pipeline, model_path)


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
 main
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
 codex/create-django-project-for-random-forest-model-qz73b2
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

    # En caso de que el CSV traiga la columna objetivo por error, la removemos.
    dataframe = dataframe.copy()
    if "PUNT_GLOBAL" in dataframe.columns:
        dataframe = dataframe.drop(columns=["PUNT_GLOBAL"])

    pipeline = load_trained_model()
    predictions = pipeline.predict(dataframe)

    result_df = dataframe.copy()
    result_df["PREDICCION_PUNT_GLOBAL"] = predictions
    return result_df

        "metrics": metrics,
        "n_registros": int(len(data)),
        "n_variables": int(X.shape[1]),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "columnas_numericas": numeric_features,
        "columnas_categoricas": categorical_features,
    }
 main
