"""Views for training and predicting Saber Pro PUNT_GLOBAL using Random Forest."""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render

from .ml_core import predict_from_dataframe, train_random_forest


def train_model(request):
    """View to trigger model training from CSV files in a directory."""

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
                        "metrics_path": result.get("metrics_path"),
                        "columnas_numericas": result.get("columnas_numericas", []),
                        "columnas_categoricas": result.get("columnas_categoricas", []),
                        "progreso": result.get("progreso", []),
                        "success": "Entrenamiento completado correctamente.",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - mostramos el error en la plantilla
                context["error"] = str(exc)

    return render(request, "train.html", context)


def predict(request):
    """View to load a trained model and predict PUNT_GLOBAL for a new CSV."""

    context: Dict[str, Any] = {}

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        action = request.POST.get("action", "preview")

        if not uploaded_file:
            context["error"] = "Debe subir un archivo CSV o Excel para generar predicciones."
        else:
            try:
                file_bytes = uploaded_file.read()
                file_suffix = uploaded_file.name.lower()

                if file_suffix.endswith((".xlsx", ".xls")):
                    dataframe = pd.read_excel(BytesIO(file_bytes))
                else:
                    dataframe = pd.read_csv(BytesIO(file_bytes))

                result_df = predict_from_dataframe(dataframe)

                if action == "download":
                    response = HttpResponse(content_type="text/csv")
                    response["Content-Disposition"] = "attachment; filename=predicciones.csv"
                    result_df.to_csv(response, index=False)
                    return response

                preview_rows = min(len(result_df), 10)
                context["preview_table"] = result_df.head(preview_rows).to_html(
                    index=False, classes="table table-striped"
                )
                context["has_predictions"] = True
            except Exception as exc:  # noqa: BLE001 - mostramos el error en la plantilla
                context["error"] = str(exc)

    return render(request, "predict.html", context)
