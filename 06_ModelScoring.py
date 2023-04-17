# Used cars kicks classification - Model scoring
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


from sklearn.metrics import PrecisionRecallDisplay, precision_recall_fscore_support, brier_score_loss
import matplotlib.pyplot as plt


# Create dummy classifier pipeline


# Create logistic regression pipeline with optimal hyperparameters


# Create SVM pipeline with optimal hyperparameters


# Create XGBoost pipeline with optimal hyperparameters


# Define function that fits every model pipeline & predicts test data, retrieves
# performance metrics, plots the PRC curve 


# Fit on training data, predict testing data, retrieve positive label probs
pipe_logistic.fit(x_train, y_train)
y_pred = pipe_logistic.predict(x_test)
y_prob = pipe_logistic.predict_proba(x_test)
y_prob_pos = np.array([x[1] for x in y_prob])


# Plot PRC curve
PrecisionRecallDisplay.from_predictions(y_test, y_prob_pos)
plt.show()
plt.close("all")


# Compute precision - recall - F score
precision_recall_fscore_support(
  y_test, y_pred, average = "binary", pos_label = 1)
  

# Compute Brier score
brier_score_loss(y_test, y_pred, pos_label = 1)
