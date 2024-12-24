from django import forms

class ModelComparisonForm(forms.Form):
    MODEL_CHOICES = [
        ('KN_model', 'K-Nearest Neighbors'),
        ('LR_model', 'Logistic Regression'),
        ('DT_model', 'Decision Tree'),  # Added Decision Tree model
    ]
    METRIC_CHOICES = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 Score'),
    ]
    model_1 = forms.ChoiceField(choices=MODEL_CHOICES, label="Select First Model")
    model_2 = forms.ChoiceField(choices=MODEL_CHOICES, label="Select Second Model")
    metric = forms.ChoiceField(choices=METRIC_CHOICES, label="Select Metric")
