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
    age = forms.FloatField(
        min_value=0, 
        max_value=100, 
        label='Age',
        error_messages={
            'min_value': 'Age must be at least 0 years.',
            'max_value': 'Age cannot exceed 100 years.',
            'required': 'Please enter your age.',
            'invalid': 'Please enter a valid number for age.'
        }
    )
    bp = forms.FloatField(
        label='Blood Pressure', 
        min_value=0, 
        max_value=200,
        error_messages={
            'min_value': 'Blood pressure must be at least 0 mmHg.',
            'max_value': 'Blood pressure cannot exceed 200 mmHg.',
            'required': 'Please enter blood pressure.',
            'invalid': 'Please enter a valid number for blood pressure.'
        }
    )
    sg = forms.FloatField(
        label='Specific Gravity', 
        min_value=1.0, 
        max_value=1.025,
        error_messages={
            'min_value': 'Specific gravity must be at least 1.0.',
            'max_value': 'Specific gravity cannot exceed 1.025.',
            'required': 'Please enter specific gravity.',
            'invalid': 'Please enter a valid number for specific gravity.'
        }
    )
    al = forms.FloatField(
        label='Albumin', 
        min_value=0, 
        max_value=5,
        error_messages={
            'min_value': 'Albumin level must be at least 0.',
            'max_value': 'Albumin level cannot exceed 5.',
            'required': 'Please enter albumin level.',
            'invalid': 'Please enter a valid number for albumin level.'
        }
    )
    su = forms.FloatField(
        label='Sugar', 
        min_value=0, 
        max_value=5,
        error_messages={
            'min_value': 'Sugar level must be at least 0.',
            'max_value': 'Sugar level cannot exceed 5.',
            'required': 'Please enter sugar level.',
            'invalid': 'Please enter a valid number for sugar level.'
        }
    )
    bgr = forms.FloatField(
        label='Blood Glucose Random', 
        min_value=0,
        error_messages={
            'min_value': 'Blood glucose level must be at least 0.',
            'required': 'Please enter blood glucose level.',
            'invalid': 'Please enter a valid number for blood glucose level.'
        }
    )
    bu = forms.FloatField(
        label='Blood Urea', 
        min_value=0,
        error_messages={
            'min_value': 'Blood urea level must be at least 0.',
            'required': 'Please enter blood urea level.',
            'invalid': 'Please enter a valid number for blood urea level.'
        }
    )
    sc = forms.FloatField(
        label='Serum Creatinine', 
        min_value=0,
        error_messages={
            'min_value': 'Serum creatinine level must be at least 0.',
            'required': 'Please enter serum creatinine level.',
            'invalid': 'Please enter a valid number for serum creatinine level.'
        }
    )
    sod = forms.FloatField(
        label='Sodium', 
        min_value=0,
        error_messages={
            'min_value': 'Sodium level must be at least 0.',
            'required': 'Please enter sodium level.',
            'invalid': 'Please enter a valid number for sodium level.'
        }
    )
    pot = forms.FloatField(
        label='Potassium', 
        min_value=0,
        error_messages={
            'min_value': 'Potassium level must be at least 0.',
            'required': 'Please enter potassium level.',
            'invalid': 'Please enter a valid number for potassium level.'
        }
    )
    hemo = forms.FloatField(
        label='Hemoglobin', 
        min_value=0,
        error_messages={
            'min_value': 'Hemoglobin level must be at least 0.',
            'required': 'Please enter hemoglobin level.',
            'invalid': 'Please enter a valid number for hemoglobin level.'
        }
    )
    pcv = forms.FloatField(
        label='Packed Cell Volume', 
        min_value=0,
        error_messages={
            'min_value': 'Packed cell volume must be at least 0.',
            'required': 'Please enter packed cell volume.',
            'invalid': 'Please enter a valid number for packed cell volume.'
        }
    )
    wc = forms.FloatField(
        label='White Blood Cell Count', 
        min_value=0,
        error_messages={
            'min_value': 'White blood cell count must be at least 0.',
            'required': 'Please enter white blood cell count.',
            'invalid': 'Please enter a valid number for white blood cell count.'
        }
    )
    rc = forms.FloatField(
        label='Red Blood Cell Count', 
        min_value=0,
        error_messages={
            'min_value': 'Red blood cell count must be at least 0.',
            'required': 'Please enter red blood cell count.',
            'invalid': 'Please enter a valid number for red blood cell count.'
        }
    )

    # Categorical inputs
    CHOICES = [('0', 'No'), ('1', 'Yes')]
    RBC_CHOICES = [('0', 'Normal'), ('1', 'Abnormal')]
    PC_CHOICES = [('0', 'Normal'), ('1', 'Abnormal')]
    
    rbc = forms.ChoiceField(
        label='Red Blood Cells', 
        choices=RBC_CHOICES, 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select red blood cell status.',
            'invalid_choice': 'Please select a valid option for red blood cells.'
        }
    )
    pc = forms.ChoiceField(
        label='Pus Cell', 
        choices=PC_CHOICES, 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select pus cell status.',
            'invalid_choice': 'Please select a valid option for pus cells.'
        }
    )
    pcc = forms.ChoiceField(
        label='Pus Cell Clumps', 
        choices=CHOICES, 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select pus cell clumps status.',
            'invalid_choice': 'Please select a valid option for pus cell clumps.'
        }
    )
    ba = forms.ChoiceField(
        label='Bacteria', 
        choices=CHOICES, 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select bacteria status.',
            'invalid_choice': 'Please select a valid option for bacteria.'
        }
    )
    htn = forms.ChoiceField(
        label='Hypertension', 
        choices=CHOICES, 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select hypertension status.',
            'invalid_choice': 'Please select a valid option for hypertension.'
        }
    )
    dm = forms.ChoiceField(
        label='Diabetes Mellitus', 
        choices=CHOICES, 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select diabetes mellitus status.',
            'invalid_choice': 'Please select a valid option for diabetes mellitus.'
        }
    )
    cad = forms.ChoiceField(
        label='Coronary Artery Disease', 
        choices=CHOICES, 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select coronary artery disease status.',
            'invalid_choice': 'Please select a valid option for coronary artery disease.'
        }
    )
    appet = forms.ChoiceField(
        label='Appetite', 
        choices=[('0', 'Poor'), ('1', 'Good')], 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select appetite status.',
            'invalid_choice': 'Please select a valid option for appetite.'
        }
    )
    pe = forms.ChoiceField(
        label='Pedal Edema', 
        choices=CHOICES, 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select pedal edema status.',
            'invalid_choice': 'Please select a valid option for pedal edema.'
        }
    )
    ane = forms.ChoiceField(
        label='Anemia', 
        choices=CHOICES, 
        widget=forms.RadioSelect,
        error_messages={
            'required': 'Please select anemia status.',
            'invalid_choice': 'Please select a valid option for anemia.'
        }
    )
