Markdown

# CKD Project

This project is a Django web application for comparing different machine learning models on a Chronic Kidney Disease (CKD) dataset. The application allows users to select two models, compare their performance using various metrics, and visualize the results using bar charts, pie charts, confusion matrices, and classification reports.

## Features

- Compare different machine learning models
- Visualize model performance using bar charts and pie charts
- Display confusion matrices and classification reports

## Models Supported

- K-Nearest Neighbors (KN_model)
- Logistic Regression (LR_model)
- Decision Tree (DT_model)
- Random Forest (RF_model)
- Support Vector Machine (SVM_model)

## Setup

### Prerequisites

- Python 3.6+
- pip (Python package installer)
- Virtual environment tool (e.g., `venv`)

### Installation

1. Clone the repository:

   ```bash
   git clone [https://github.com/ShamimAhmed82245/ckd_project](https://github.com/ShamimAhmed82245/ckd_project)
   cd ckd_project
   Create a virtual environment:
   ```

Bash

python -m venv myenv
Activate the virtual environment:

On Windows:

Bash

myenv\Scripts\activate
On macOS and Linux:

Bash

source myenv/bin/activate
Install 1 the required packages:

1.  github.com
    github.com

Bash

pip install -r requirements.txt
Apply the migrations:

Bash

python manage.py makemigrations
python manage.py migrate
Run the development server:

Bash

python manage.py runserver
Open your web browser and go to http://127.0.0.1:8000/.

Usage
Select two models from the dropdown menus on the "Compare Models" page.
Select a metric to compare the models.
Click the "Compare" button to see the results.
Navigate to different pages using the menu bar to view bar charts, pie charts, confusion matrices, and classification reports.
