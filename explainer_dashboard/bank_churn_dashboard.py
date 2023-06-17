import pandas as pd
from os import getcwd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from pycaret.classification import load_model

# Retrieve the Test Data
ml_path = "/mlruns/1/bd6045e85a4b485cafdb639247bf439e/artifacts"
test_data_path = getcwd() + ml_path

# Test Data
test_df = pd.read_csv(f"{test_data_path}/Test.csv")
test_df = test_df.iloc[:, 1:]

# Features, Target
X_test = test_df.drop("Exited", axis=1)
y_test = test_df["Exited"]

# Load the model from PyCaret
p = "/pycaret_assets/models/bank_churn_experiment_ensembles_v2_best_model"
model_path = (
    getcwd() + p
)
model = load_model(model_path)

# Setting up the Classifier Experiment
classif_experiment = ClassifierExplainer(
    model, X_test, y_test, shap='tree', n_jobs=-1)

# Dashboard
ExplainerDashboard(classif_experiment,
                   title="Bank Churn Analytics").run()
