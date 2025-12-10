"""Views para entrenar y predecir PUNT_GLOBAL usando Random Forest."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from .ml_core import predict_from_dataframe, train_random_forest


def train_model(request: HttpRequest) -> HttpResponse:
    """
    Vista para lanzar el entrenamiento del modelo a partir de varios CSV
    ubicados en un directorio local.
    """
    context: Dict[str, Any] = {}

    if request.method == "POST":
        # Ruta que el usuario escribe en el formulario
        data_dir = request.POST.get("data_dir", "").strip()
        context["data_dir"] = data_dir

        if not data_dir:
            context["error"] = "Debe proporcionar la ruta al directorio que contiene los archivos CSV."
        else:
            try:
                # Esta función debe:
                # - Leer todos los CSV de la carpeta
                # - Entrenar el modelo con la columna objetivo PUNT_GLOBAL
                # - Guardar el modelo y métricas en MODEL_DIR
                result = train_random_forest(data_dir)

                context.update(
                    {
                        "metrics": result.get("metrics", {}),
                        "n_registros": result.get("n_registros"),
                        "n_variables": result.get("n_variables"),
                        "model_path": result.get("model_path"),
                        "metrics_path": result.get("metrics_path"),
                        "columnas_numericas": result.get("columnas_numericas", []),
                        "columnas_categoricas": result.get("columnas_categoricas", []),
                        "success": "Entrenamiento completado correctamente.",
                    }
                )
                # Si todo salió bien, nos aseguramos de que no quede error viejo
                context.pop("error", None)
            except Exception as exc:  # noqa: BLE001
                context["error"] = f"Error durante el entrenamiento: {exc}"

    return render(request, "train.html", context)


def predict(request: HttpRequest) -> HttpResponse:
    """
    Vista para cargar un CSV nuevo, aplicar el modelo entrenado y generar
    la columna PREDICCION_PUNT_GLOBAL. Permite vista previa o descarga.
    """
    context: Dict[str, Any] = {}

    # Verificamos que exista carpeta de modelos
    model_dir: Path = Path(getattr(settings, "MODEL_DIR", Path("models")))
    if not model_dir.exists():
        context["error"] = (
            "No se encontró el directorio de modelos entrenados. "
            "Verifique la configuración de MODEL_DIR y entrene el modelo primero."
        )

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        action = request.POST.get("action", "preview")

        if not uploaded_file:
            context["error"] = "Debe subir un archivo CSV para generar predicciones."
        else:
            try:
                # Leemos el archivo en memoria
                data_bytes = uploaded_file.read()
                df = pd.read_csv(BytesIO(data_bytes))

                # Esta función debe:
                # - Cargar el modelo desde MODEL_DIR
                # - Aplicar el preprocesamiento
                # - Devolver un DataFrame con PREDICCION_PUNT_GLOBAL
                result_df = predict_from_dataframe(df)

                # Si el usuario pidió descarga, devolvemos el CSV
                if action == "download":
                    response = HttpResponse(content_type="text/csv")
                    response["Content-Disposition"] = 'attachment; filename="predicciones.csv"'
                    result_df.to_csv(response, index=False)
                    return response

                # Si es vista previa, mostramos las primeras filas
                preview_rows = min(len(result_df), 10)
                context["preview_table"] = result_df.head(preview_rows).to_html(
                    index=False,
                    classes="table table-striped",
                )
                context["has_predictions"] = True
                context.pop("error", None)
            except Exception as exc:  # noqa: BLE001
                context["error"] = f"Error al generar predicciones: {exc}"

    return render(request, "predict.html", context)
