import numpy as np
import pickle
from django.shortcuts import render
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .forms import ModelComparisonForm

def compare_models(request):
    if request.method == "POST":
        # Get selected models and metric
        model_1_name = request.POST.get("model_1")
        model_2_name = request.POST.get("model_2")
        metric = request.POST.get("metric")

        # Load models
        model_1 = pickle.load(open(f"ckd_app/models/{model_1_name}.sav", "rb"))
        model_2 = pickle.load(open(f"ckd_app/models/{model_2_name}.sav", "rb"))

        # Load x_test_data.npy and y_test_data.npy
        x_test_path = "ckd_app/data/x_test_data.npy"
        y_test_path = "ckd_app/data/y_test_data.npy"
        x_test_data = np.load(x_test_path)
        y_test_data = np.load(y_test_path)

        # Perform predictions
        predictions_1 = model_1.predict(x_test_data)
        predictions_2 = model_2.predict(x_test_data)

        # Calculate the selected metric
        if metric == "accuracy":
            metric_1 = accuracy_score(y_test_data, predictions_1)
            metric_2 = accuracy_score(y_test_data, predictions_2)
        elif metric == "precision":
            metric_1 = precision_score(y_test_data, predictions_1)
            metric_2 = precision_score(y_test_data, predictions_2)
        elif metric == "recall":
            metric_1 = recall_score(y_test_data, predictions_1)
            metric_2 = recall_score(y_test_data, predictions_2)
        elif metric == "f1":
            metric_1 = f1_score(y_test_data, predictions_1)
            metric_2 = f1_score(y_test_data, predictions_2)

        # Prepare context for the results page
        context = {
            "model_1_name": model_1_name,
            "model_2_name": model_2_name,
            "predictions_1": predictions_1.tolist(),
            "predictions_2": predictions_2.tolist(),
            "metric": metric,
            "metric_1": metric_1,
            "metric_2": metric_2,
        }
        return render(request, "compare_results.html", context)

    form = ModelComparisonForm()
    return render(request, "select_models.html", {"form": form})
