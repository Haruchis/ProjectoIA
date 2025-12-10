"""Views for training and predicting Saber Pro PUNT_GLOBAL using Random Forest."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render

from .ml_core import train_random_forest


def train_view(request):
    """Simple view to trigger model training from CSV files in a directory."""
    context: Dict[str, Any] = {}

    if request.method == "POST":
        data_dir = request.POST.get("data_dir", "").strip()
        context["data_dir"] = data_dir
        if not data_dir:
            context["error"] = "Debe proporcionar la ruta al directorio que contiene los CSV."
        else:
            try:
                result = train_random_forest(data_dir)
                context.update(
                    {
                        "metrics": result.get("metrics", {}),
                        "n_registros": result.get("n_registros"),
                        "n_variables": result.get("n_variables"),
                        "model_path": result.get("model_path"),
                        "columnas_numericas": result.get("columnas_numericas", []),
                        "columnas_categoricas": result.get("columnas_categoricas", []),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - mostrar mensaje claro al usuario
                context["error"] = str(exc)

    return render(request, "prediccion/train.html", context)


def predict_view(request):
    """View to load a trained model and predict PUNT_GLOBAL for a new CSV."""
    context: Dict[str, Any] = {}
    model_path = Path(getattr(settings, "MODEL_STORAGE_DIR", Path("models"))) / "random_forest_punt_global.pkl"

    if not model_path.exists():
        context["error"] = (
            "No se encontró un modelo entrenado. Entrene el modelo antes de realizar predicciones."
        )
        return render(request, "prediccion/predict.html", context)

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        action = request.POST.get("action", "preview")

        if not uploaded_file:
            context["error"] = "Debe subir un archivo CSV para generar predicciones."
        else:
            try:
                pipeline = joblib.load(model_path)
                data_bytes = uploaded_file.read()
                dataframe = pd.read_csv(BytesIO(data_bytes))

                predictions = pipeline.predict(dataframe)
                result_df = dataframe.copy()
                result_df["PREDICCION_PUNT_GLOBAL"] = predictions

                if action == "download":
                    response = HttpResponse(content_type="text/csv")
                    response["Content-Disposition"] = "attachment; filename=predicciones.csv"
                    result_df.to_csv(response, index=False)
                    return response

                context["preview_table"] = result_df.head().to_html(index=False, classes="table table-striped")
                context["has_predictions"] = True
            except Exception as exc:  # noqa: BLE001 - mostramos el error en la plantilla
                context["error"] = str(exc)

    return render(request, "prediccion/predict.html", context)
