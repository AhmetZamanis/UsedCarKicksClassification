


from sklearn.metrics import PrecisionRecallDisplay, precision_recall_fscore_support, brier_score_loss
import matplotlib.pyplot as plt


# Set parameters to optimal
pipe_logistic.set_params(
    model_logistic__C = 18.1818,
    model_logistic__l1_ratio = 0.7
  )


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
