# Used cars kicks classification - Modeling, Support Vector Machine with SGD
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import optuna


# XGBOOST IN SKLEARN PIPELINE, TUNED WITH OPTUNA & CROSSVALIDATION


# Retrieve train-test indices for each fold from RepeatedStratifiedKFold

# For each fold, fit model / pipeline with eval_set as validation fold, retrieve
# score with evals_result, then average them and give it to Optuna

# If the pipeline doesn't allow eval_set, first transform train & val data with 
# pipeline, then fit model

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold

https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
