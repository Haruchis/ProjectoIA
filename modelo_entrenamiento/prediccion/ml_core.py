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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "PUNT_GLOBAL"


def _load_csvs_from_dir(data_dir: str | Path) -> pd.DataFrame:
    """
    Carga y concatena todos los CSV dentro de un directorio.

    :param data_dir: Ruta al directororio con archivos .csv
    :return: DataFrame concatenado
    """
    data_path = Path(data_dir)

    if not data_path.exists() or not data_path.is_dir():
        raise ValueError(f"La ruta especificada no es un directorio válido: {data_path}")

    # Lista de archivos .csv en el directorio
    csv_files: List[Path] = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No se encontraron archivos CSV en el directorio: {data_path}")

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


def _build_preprocessor_and_features(
    df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    A partir de un DataFrame, identifica columnas numéricas y categóricas
    y construye un ColumnTransformer con escalado y OneHotEncoder.
    """
    # Quitamos la columna objetivo si está presente
    features_df = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df

    numeric_cols = features_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = features_df.select_dtypes(exclude=["number"]).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError(
            "No se encontraron columnas numéricas ni categóricas para entrenar el modelo."
        )

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

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

    # Carpeta donde se guardan el modelo y métricas
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
