from django import forms

class ModelComparisonForm(forms.Form):
    MODEL_CHOICES = [
        ('KN_model', 'K-Nearest Neighbors'),
        ('LR_model', 'Logistic Regression'),
        ('DT_model', 'Decision Tree'),
        ('RF_model', 'Random Forest'),  # Added Random Forest model
        ('SVM_model', 'Support Vector Machine'),  # Added Support Vector Machine model
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

class CKDPredictionForm(forms.Form):
    # Numerical inputs
    age = forms.FloatField(min_value=0, max_value=100)
    bp = forms.FloatField(label='Blood Pressure', min_value=0, max_value=200)
    sg = forms.FloatField(label='Specific Gravity', min_value=1.0, max_value=1.025)
    al = forms.FloatField(label='Albumin', min_value=0, max_value=5)
    su = forms.FloatField(label='Sugar', min_value=0, max_value=5)
    bgr = forms.FloatField(label='Blood Glucose Random', min_value=0)
    bu = forms.FloatField(label='Blood Urea', min_value=0)
    sc = forms.FloatField(label='Serum Creatinine', min_value=0)
    sod = forms.FloatField(label='Sodium', min_value=0)
    pot = forms.FloatField(label='Potassium', min_value=0)
    hemo = forms.FloatField(label='Hemoglobin', min_value=0)
    pcv = forms.FloatField(label='Packed Cell Volume', min_value=0)
    wc = forms.FloatField(label='White Blood Cell Count', min_value=0)
    rc = forms.FloatField(label='Red Blood Cell Count', min_value=0)

    # Categorical inputs
    CHOICES = [(0, 'No'), (1, 'Yes')]
    
    pcc = forms.ChoiceField(label='Pus Cell Clumps', choices=CHOICES)
    ba = forms.ChoiceField(label='Bacteria', choices=CHOICES)
    htn = forms.ChoiceField(label='Hypertension', choices=CHOICES)
    dm = forms.ChoiceField(label='Diabetes Mellitus', choices=CHOICES)
    cad = forms.ChoiceField(label='Coronary Artery Disease', choices=CHOICES)
    appet = forms.ChoiceField(label='Appetite', choices=CHOICES)
    pe = forms.ChoiceField(label='Pedal Edema', choices=CHOICES)
    ane = forms.ChoiceField(label='Anemia', choices=CHOICES)
