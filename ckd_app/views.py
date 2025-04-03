import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import urllib, base64
from django.shortcuts import render, redirect
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from .forms import ModelComparisonForm, CKDPredictionForm
import json
from sklearn.preprocessing import StandardScaler

def compare_models(request):
    if request.method == "POST":
        form = ModelComparisonForm(request.POST)
        if form.is_valid():
            # Get selected models and metric
            model_1_name = form.cleaned_data["model_1"]
            model_2_name = form.cleaned_data["model_2"]
            metric = form.cleaned_data["metric"]

            # Store selected models and metric in session
            request.session['model_1_name'] = model_1_name
            request.session['model_2_name'] = model_2_name
            request.session['metric'] = metric

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
    else:
        form = ModelComparisonForm()
    return render(request, "compare.html", {"form": form})

def metrics(request):
    model_1_name = request.session.get('model_1_name')
    model_2_name = request.session.get('model_2_name')

    if model_1_name and model_2_name:
        # Load models
        model_1 = pickle.load(open(f"ckd_app/models/{model_1_name}.sav", "rb"))
        model_2 = pickle.load(open(f"ckd_app/models/{model_2_name}.sav", "rb"))

        # Load test data
        x_test_path = "ckd_app/data/x_test_data.npy"
        y_test_path = "ckd_app/data/y_test_data.npy"
        x_test_data = np.load(x_test_path)
        y_test_data = np.load(y_test_path)

        # Perform predictions
        predictions_1 = model_1.predict(x_test_data)
        predictions_2 = model_2.predict(x_test_data)

        # Calculate all metrics
        metrics_1 = {
            "accuracy": accuracy_score(y_test_data, predictions_1),
            "precision": precision_score(y_test_data, predictions_1),
            "recall": recall_score(y_test_data, predictions_1),
            "f1": f1_score(y_test_data, predictions_1),
        }
        metrics_2 = {
            "accuracy": accuracy_score(y_test_data, predictions_2),
            "precision": precision_score(y_test_data, predictions_2),
            "recall": recall_score(y_test_data, predictions_2),
            "f1": f1_score(y_test_data, predictions_2),
        }

        context = {
            "model_1_name": model_1_name,
            "model_2_name": model_2_name,
            "metrics_1": metrics_1,
            "metrics_2": metrics_2,
        }
        return render(request, "metrics_results.html", context)
    else:
        return redirect('compare')

def metric_bar(request):
    model_1_name = request.session.get('model_1_name')
    model_2_name = request.session.get('model_2_name')

    if model_1_name and model_2_name:
        # Load models
        model_1 = pickle.load(open(f"ckd_app/models/{model_1_name}.sav", "rb"))
        model_2 = pickle.load(open(f"ckd_app/models/{model_2_name}.sav", "rb"))

        # Load test data
        x_test_path = "ckd_app/data/x_test_data.npy"
        y_test_path = "ckd_app/data/y_test_data.npy"
        x_test_data = np.load(x_test_path)
        y_test_data = np.load(y_test_path)

        # Perform predictions
        predictions_1 = model_1.predict(x_test_data)
        predictions_2 = model_2.predict(x_test_data)

        # Calculate all metrics
        metrics_1 = {
            "accuracy": accuracy_score(y_test_data, predictions_1),
            "precision": precision_score(y_test_data, predictions_1),
            "recall": recall_score(y_test_data, predictions_1),
            "f1": f1_score(y_test_data, predictions_1),
        }
        metrics_2 = {
            "accuracy": accuracy_score(y_test_data, predictions_2),
            "precision": precision_score(y_test_data, predictions_2),
            "recall": recall_score(y_test_data, predictions_2),
            "f1": f1_score(y_test_data, predictions_2),
        }

        # Generate bar chart
        labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        model_1_metrics = [metrics_1['accuracy'], metrics_1['precision'], metrics_1['recall'], metrics_1['f1']]
        model_2_metrics = [metrics_2['accuracy'], metrics_2['precision'], metrics_2['recall'], metrics_2['f1']]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, model_1_metrics, width, label=model_1_name)
        rects2 = ax.bar(x + width/2, model_2_metrics, width, label=model_2_name)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by model and metric')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        # Save the plot to a PNG image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

        context = {
            "model_1_name": model_1_name,
            "model_2_name": model_2_name,
            "metrics_1": metrics_1,
            "metrics_2": metrics_2,
            "bar_chart": uri,
        }
        return render(request, "metric_bar.html", context)
    else:
        return redirect('compare')

def metric_pie(request):
    model_1_name = request.session.get('model_1_name')
    model_2_name = request.session.get('model_2_name')

    if model_1_name and model_2_name:
        # Load models
        model_1 = pickle.load(open(f"ckd_app/models/{model_1_name}.sav", "rb"))
        model_2 = pickle.load(open(f"ckd_app/models/{model_2_name}.sav", "rb"))

        # Load test data
        x_test_path = "ckd_app/data/x_test_data.npy"
        y_test_path = "ckd_app/data/y_test_data.npy"
        x_test_data = np.load(x_test_path)
        y_test_data = np.load(y_test_path)

        # Perform predictions
        predictions_1 = model_1.predict(x_test_data)
        predictions_2 = model_2.predict(x_test_data)

        # Calculate all metrics
        metrics_1 = {
            "accuracy": accuracy_score(y_test_data, predictions_1),
            "precision": precision_score(y_test_data, predictions_1),
            "recall": recall_score(y_test_data, predictions_1),
            "f1": f1_score(y_test_data, predictions_1),
        }
        metrics_2 = {
            "accuracy": accuracy_score(y_test_data, predictions_2),
            "precision": precision_score(y_test_data, predictions_2),
            "recall": recall_score(y_test_data, predictions_2),
            "f1": f1_score(y_test_data, predictions_2),
        }

        # Generate pie chart
        labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        sizes_1 = [metrics_1['accuracy'], metrics_1['precision'], metrics_1['recall'], metrics_1['f1']]
        sizes_2 = [metrics_2['accuracy'], metrics_2['precision'], metrics_2['recall'], metrics_2['f1']]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.pie(sizes_1, labels=labels, autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title(model_1_name)

        ax2.pie(sizes_2, labels=labels, autopct='%1.1f%%', startangle=140)
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax2.set_title(model_2_name)

        fig.tight_layout()

        # Save the plot to a PNG image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

        context = {
            "model_1_name": model_1_name,
            "model_2_name": model_2_name,
            "metrics_1": metrics_1,
            "metrics_2": metrics_2,
            "pie_chart": uri,
        }
        return render(request, "metric_pie.html", context)
    else:
        return redirect('compare')

def confusion_matrix_view(request):
    model_1_name = request.session.get('model_1_name')
    model_2_name = request.session.get('model_2_name')

    if model_1_name and model_2_name:
        # Load models
        model_1 = pickle.load(open(f"ckd_app/models/{model_1_name}.sav", "rb"))
        model_2 = pickle.load(open(f"ckd_app/models/{model_2_name}.sav", "rb"))

        # Load test data
        x_test_path = "ckd_app/data/x_test_data.npy"
        y_test_path = "ckd_app/data/y_test_data.npy"
        x_test_data = np.load(x_test_path)
        y_test_data = np.load(y_test_path)

        # Perform predictions
        predictions_1 = model_1.predict(x_test_data)
        predictions_2 = model_2.predict(x_test_data)

        # Calculate confusion matrices
        cm_1 = confusion_matrix(y_test_data, predictions_1)
        cm_2 = confusion_matrix(y_test_data, predictions_2)

        # Generate confusion matrix plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.matshow(cm_1, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm_1.shape[0]):
            for j in range(cm_1.shape[1]):
                ax1.text(x=j, y=i, s=cm_1[i, j], va='center', ha='center')
        ax1.set_title(model_1_name)
        ax1.set_xlabel('Predicted labels')
        ax1.set_ylabel('True labels')

        ax2.matshow(cm_2, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm_2.shape[0]):
            for j in range(cm_2.shape[1]):
                ax2.text(x=j, y=i, s=cm_2[i, j], va='center', ha='center')
        ax2.set_title(model_2_name)
        ax2.set_xlabel('Predicted labels')
        ax2.set_ylabel('True labels')

        fig.tight_layout()

        # Save the plot to a PNG image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

        context = {
            "model_1_name": model_1_name,
            "model_2_name": model_2_name,
            "confusion_matrix_1": cm_1,
            "confusion_matrix_2": cm_2,
            "confusion_matrix_plot": uri,
        }
        return render(request, "confusion_matrix.html", context)
    else:
        return redirect('compare')

def classification_report_view(request):
    model_1_name = request.session.get('model_1_name')
    model_2_name = request.session.get('model_2_name')

    if model_1_name and model_2_name:
        # Load models
        model_1 = pickle.load(open(f"ckd_app/models/{model_1_name}.sav", "rb"))
        model_2 = pickle.load(open(f"ckd_app/models/{model_2_name}.sav", "rb"))

        # Load test data
        x_test_path = "ckd_app/data/x_test_data.npy"
        y_test_path = "ckd_app/data/y_test_data.npy"
        x_test_data = np.load(x_test_path)
        y_test_data = np.load(y_test_path)

        # Perform predictions
        predictions_1 = model_1.predict(x_test_data)
        predictions_2 = model_2.predict(x_test_data)

        # Generate classification reports
        report_1 = classification_report(y_test_data, predictions_1, output_dict=True)
        report_2 = classification_report(y_test_data, predictions_2, output_dict=True)

        context = {
            "model_1_name": model_1_name,
            "model_2_name": model_2_name,
            "report_1": json.dumps(report_1),
            "report_2": json.dumps(report_2),
        }
        return render(request, "classification_report.html", context)
    else:
        return redirect('compare')

def predict_ckd(request):
    prediction = None
    probabilities = None
    
    if request.method == 'POST':
        form = CKDPredictionForm(request.POST)
        if form.is_valid():
            # Get form data
            features = [
                form.cleaned_data[field] for field in [
                    'age', 'bp', 'sg', 'al', 'su', 'pcc', 'ba', 'bgr',
                    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                    'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
                ]
            ]
            
            # Load the scaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(np.array(features).reshape(1, -1))
            
            # Load the model (using best performing model - Decision Tree)
            model = pickle.load(open('ckd_app/models/DT_model.sav', 'rb'))
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probabilities_raw = model.predict_proba(features_scaled)[0]
            no_ckd_prob, ckd_prob = probabilities_raw[0], probabilities_raw[1]
            probabilities = {
                '0': float(no_ckd_prob * 100),  # Convert to percentage
                '1': float(ckd_prob * 100)      # Convert to percentage
            }
            
    else:
        form = CKDPredictionForm()
    
    return render(request, 'predict.html', {
        'form': form,
        'prediction': prediction,
        'probabilities': probabilities
    })
