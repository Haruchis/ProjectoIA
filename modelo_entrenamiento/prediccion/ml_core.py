
"""Core machine learning utilities for training and prediction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib

"""Core de ML para entrenar y predecir PUNT_GLOBAL con Random Forest."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
>>>>>>> main
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


def _coerce_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric when fully convertible, otherwise to strings."""

    cleaned = features.copy()

    for column in cleaned.columns:
        coerced = pd.to_numeric(cleaned[column], errors="coerce")
        if coerced.notna().all():
            # Columna completamente numérica, preservamos dtype numérico real
            cleaned[column] = coerced
        else:
            # Mezcla de tipos: forzamos a string para evitar errores en OneHotEncoder
            cleaned[column] = cleaned[column].astype(str)

    return cleaned


def _drop_missing_after_typing(
    features: pd.DataFrame, target: pd.Series, progress: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove rows with missing values after type coercion.

    Drop rows where any feature or the target contains ``NaN``. This is a minimal,
    transparent cleaning step to avoid scikit-learn errors when datasets include
    celdas vacías o valores faltantes mezclados con texto.
    """

    mask = features.notna().all(axis=1) & target.notna()
    removed = int(len(features) - mask.sum())
    if removed:
        progress.append(
            f"Filas eliminadas tras coerción de tipos por valores faltantes: {removed}"
        )

    cleaned_features = features.loc[mask].reset_index(drop=True)
    cleaned_target = target.loc[mask].reset_index(drop=True)

    if cleaned_features.empty:
        raise ValueError(
            "No quedan filas válidas después de eliminar valores faltantes. Revise los datos."
        )

    return cleaned_features, cleaned_target


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
    X = _coerce_feature_columns(data.drop(columns=["PUNT_GLOBAL"]))

    # Nueva limpieza mínima tras coerción para eliminar cualquier NaN residual.
    X, y = _drop_missing_after_typing(X, y, progress)

    numeric_features, categorical_features = _split_feature_types(X)
    X = _harmonize_feature_dtypes(X, categorical_features)
    preprocessor = _build_preprocessor(numeric_features, categorical_features)


TARGET_COL = "PUNT_GLOBAL"


def _load_csvs_from_dir(data_dir: str | Path) -> pd.DataFrame:
    """
    Carga y concatena todos los CSV dentro de un directorio.

    :param data_dir: Ruta al directorio con archivos .csv
    :return: DataFrame concatenado
    """
    data_path = Path(data_dir)

    if not data_path.exists() or not data_path.is_dir():
        raise ValueError(f"La ruta especificada no es un directorio válido: {data_path}")

    csv_files: List[Path] = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No se encontraron archivos CSV en el directorio: {data_path}")

    frames: List[pd.DataFrame] = []

def _load_csvs_from_dir(data_dir: str | Path) -> pd.DataFrame:
    """
    Carga y concatena todos los CSV dentro de un directorio.

    :param data_dir: Ruta al directorio con archivos .csv
    :return: DataFrame concatenado
    """
    data_path = Path(data_dir)

    if not data_path.exists() or not data_path.is_dir():
        raise ValueError(f"La ruta especificada no es un directorio válido: {data_path}")

    # Lista de archivos .csv en el directorio
    csv_files: List[Path] = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No se encontraron archivos CSV en el directorio: {data_path}")

    # Aquí vamos a ir acumulando los DataFrames
    frames: List[pd.DataFrame] = []

    for f in csv_files:
        try:
            # Intento con UTF-8 primero
            frames.append(pd.read_csv(f))
        except UnicodeDecodeError:
            # Si falla, intento con latin1 (típico de archivos guardados desde Excel en Windows)
            try:
                frames.append(pd.read_csv(f, encoding="latin1"))
            except Exception as exc:
                raise ValueError(
                    f"Error leyendo el archivo {f} con utf-8 ni latin1: {exc}"
                ) from exc
        except Exception as exc:
            raise ValueError(f"Error leyendo el archivo {f}: {exc}") from exc

    if not frames:
        raise ValueError("No se pudo cargar ningún CSV válido.")

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("El DataFrame combinado está vacío. Revise los archivos CSV.")

    return df



    if not frames:
        raise ValueError("No se pudo cargar ningún CSV válido.")

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("El DataFrame combinado está vacío. Revise los archivos CSV.")

    return df


def _build_preprocessor_and_features(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    A partir de un DataFrame, identifica columnas numéricas y categóricas
    y construye un ColumnTransformer con escalado y OneHotEncoder.
    """
    # Quitamos la columna objetivo si está presente
    features_df = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df

    numeric_cols = features_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = features_df.select_dtypes(exclude=["number"]).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError("No se encontraron columnas numéricas ni categóricas para entrenar el modelo.")

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    preprocessor = ColumnTransformer(transformers)

    return preprocessor, numeric_cols, categorical_cols


def train_random_forest(data_dir: str | Path) -> Dict[str, Any]:
    """
    Entrena un RandomForestRegressor para predecir PUNT_GLOBAL a partir de CSVs
    ubicados en un directorio.

    Devuelve un diccionario con métricas, metadatos y registro de progreso.
    """
    progress: list[str] = []

    progress.append("Iniciando carga de archivos CSV...")
    df = _load_csvs_from_dir(data_dir)
    progress.append(f"Se cargaron {len(df)} filas en total desde los CSV.")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"La columna objetivo '{TARGET_COL}' no está presente en los datos. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    progress.append("Construyendo preprocesador numérico/categórico...")
    preprocessor, num_cols, cat_cols = _build_preprocessor_and_features(df)

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )


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
    dataframe = _coerce_feature_columns(dataframe)
    _, categorical_features = _split_feature_types(dataframe)
    dataframe = _harmonize_feature_dtypes(dataframe, categorical_features)

    pipeline = load_trained_model()
    predictions = pipeline.predict(dataframe)

    result_df = dataframe.copy()
    result_df["PREDICCION_PUNT_GLOBAL"] = predictions

            ("model", rf),
        ]
    )

    progress.append("Dividiendo en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    progress.append("Entrenando el modelo Random Forest...")
    pipeline.fit(X_train, y_train)

    progress.append("Calculando métricas de evaluación...")
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics: Dict[str, Any] = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    model_dir: Path = Path(getattr(settings, "MODEL_DIR", Path("models")))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "random_forest_punt_global.pkl"
    metrics_path = model_dir / "metrics_random_forest_punt_global.csv"

    progress.append(f"Guardando modelo entrenado en: {model_path}")
    joblib.dump(pipeline, model_path)

    progress.append(f"Guardando métricas en: {metrics_path}")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    progress.append("Entrenamiento finalizado correctamente.")

    return {
        "metrics": metrics,
        "n_registros": int(len(df)),
        "n_variables": int(df.shape[1]),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "columnas_numericas": num_cols,
        "columnas_categoricas": cat_cols,
        "progress": progress,
    }


def predict_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Carga el modelo entrenado desde MODEL_DIR y genera la columna
    PREDICCION_PUNT_GLOBAL para el DataFrame recibido.
    """
    model_dir: Path = Path(getattr(settings, "MODEL_DIR", Path("models")))
    model_path = model_dir / "random_forest_punt_global.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en: {model_path}. "
            "Entrene el modelo antes de hacer predicciones."
        )

    pipeline: Pipeline = joblib.load(model_path)

    # Por si el CSV que suben trae también PUNT_GLOBAL, lo quitamos de las features
    if TARGET_COL in df.columns:
        X_new = df.drop(columns=[TARGET_COL])
    else:
        X_new = df.copy()

    preds = pipeline.predict(X_new)

    result_df = df.copy()
    result_df["PREDICCION_PUNT_GLOBAL"] = preds


    return result_df
