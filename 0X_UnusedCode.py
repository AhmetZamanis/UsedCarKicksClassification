

from category_encoders.datasets import load_compass, load_postcodes

X, y = load_compass()
X, y = load_postcodes('binary')

# How to map zipcodes-states hierarchy

# With dict
dict_hierarchy = {
  "VNZIP1": {
    "FL": ("33619", "33..."),
    "GA": ("30212", "30..."),
    ...
    }
}

# All unique zipcodes for a state
df.loc[df["VNST"] == "FL", "VNZIP1"].unique().tolist()





test = pipe_preproc.fit_transform(x_train, y_train)
test[["Make", "BYRNO", "VNST", "VNZIP1"]].describe()


encode_target_zip.fit(x_train, y_train)
encode_target_zip.transform(x_train, y_train)

encode_target.fit(x_train, y_train)
encode_target.transform(x_train, y_train)


test = pipe_preproc.fit_transform(x_train, y_train)
test[0]
x_train.iloc[0,]



# Perform train-validation split
x_train_xgb, x_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
  x_train, y_train, test_size = 0.2, random_state = 1923, stratify = y_train
)



# Test loop for XGB crossvalidation: Works.
best_scores_test = []

for split in cv_indices:

  # Get train-test indices and sets
  train_index = split[0]
  val_index = split[1]

  x_tr = x_train.iloc[train_index, ]
  y_tr = y_train.iloc[train_index, ]
  x_val = x_train.iloc[val_index, ]
  y_val = y_train.iloc[val_index, ]

  # Perform preprocessing
  x_tr = pipe_process.fit_transform(x_tr, y_tr)
  x_val = pipe_process.transform(x_val)

  # Fit model and print eval set score
  model_xgb.fit(X = x_tr, y = y_tr, eval_set = [(x_val, y_val)], verbose = True)
  best_scores_test.append(model_xgb.best_score)

del split



# Compute class weight (for labels 0, 1)
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)
sample_weight = np.where(y_train == 1, class_weight[1], class_weight[0])


# Compute class weight (for labels -1, 1)
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)
sample_weight = np.where(y_train == 1, class_weight[0], class_weight[1])


# # Create logistic regression pipeline with optimal hyperparameters
# best_trial_logistic = pd.read_csv("./ModifiedData/trials_logistic.csv").iloc[0,]
# pipe_logistic = Pipeline(steps = [
#   ("preprocessing", pipe_process),
#   ("model_logistic", LogisticRegression(
#       penalty = "elasticnet",
#       C = (1 / best_trial_logistic["params_reg_strength"]),
#       l1_ratio = best_trial_logistic["params_l1_ratio"],
#       solver = "saga",
#       random_state = 1923,
#       max_iter = 1000,
#       class_weight = "balanced",
#     )
#   )
# ])


# Fit on training data, predict testing data, retrieve positive label probs
pipe_logistic.fit(x_train, y_train)
y_pred = pipe_logistic.predict(x_test)
y_prob = pipe_logistic.predict_proba(x_test)
y_prob_pos = np.array([x[1] for x in y_prob])

# Compute PRC values
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_pos, pos_label = 1)

pd.DataFrame({
  "Model": "Logistic",
  "Precision": precision,
  "Recall": recall})
  
for key in models_dict.keys():
  print(models_dict[key])
  

# Compute PRAUC
# 0.4307 for LogisticRegression
# 0.4223 for SGD with native early stopping
auc(recall, precision)

# Compute Brier score
# 0.194 for LogisticRegression
# 0.1958 for SGD with native early stopping
brier_score_loss(y_test, y_prob_pos, pos_label = 1, sample_weight = sample_weight_test)


# Compute Brier skill score for each model, with dummy classifier as reference:
# 1 - (brier_score_model / brier_score_dummy) (1 is best, 0 is worst)





# Make named dictionary of models
models_dict = {
  "Dummy": pipe_dummy,
  "Logistic": pipe_logistic,
  "SVM": pipe_svm,
  "XGBoost": pipe_xgb
}


# Fit models




# Define function that fits every model pipeline & scores test data, plots PRC
# curve
def score_models(models_dict):
  
  # # Make dataframe to store precision-recall values for each threshold
  # df_prc = pd.DataFrame(
  #   columns = ["Model", "Precision", "Recall"])
  
  # Make dataframe to store PRAUC & Brier scores
  df_scores = pd.DataFrame(
    columns = ["Model", "Avg. precision score", "Brier score"]
  )
  
  for key in models_dict.keys():
    
    # Fit model, predict classes & probs for test data
    model = models_dict[key]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    y_prob_pos = np.array([x[1] for x in y_prob])
    
    # Compute PRC values
    precision, recall, threshold = precision_recall_curve(
      y_test, y_prob_pos, pos_label = 1)
      
    # Compute Average precision score
    avg_precision = average_precision_score(y_test, y_prob_pos, pos_label = 1)
    
    # Compute Brier score
    brier_score = brier_score_loss(
      y_test, y_prob_pos, pos_label = 1, sample_weight = sample_weight_test)
    
    # Make dataframe of precision-recall values for each threshold
    prc = pd.DataFrame({
      "Model": key,
      "Precision": precision,
      "Recall": recall
    })
    
    # Concatenate PRC values to full dataframe
    df_prc = pd.concat([df_prc, prc])
    
    # Make dataframe of PRAUC & Brier scores, concatenate to full dataframe
    scores = pd.DataFrame({
      "Model": [key],
      "Avg. precision score": [avg_precision],
      "Brier score": [brier_score]
    })
    df_scores = pd.concat([df_scores, scores])
  
  # Drop dummy classifier's PRC values
  df_prc = df_prc.loc[df_prc["Model"] != "Dummy"]
  
  return df_prc, df_scores


# Retrieve PRC values & performance scores
df_prc, df_scores = score_models(models_dict)


# Plot PRC curves
_ = sns.lineplot(x = "Recall", y = "Precision", data = df_prc, hue = "Model")
plt.show()
plt.close("all")




# Fit on training data, predict testing data, retrieve positive label probs
pipe_dummy.fit(x_train, y_train)
y_pred = pipe_dummy.predict(x_test)
y_prob = pipe_dummy.predict_proba(x_test)
y_prob_pos = np.array([x[1] for x in y_prob])

# Compute PRC values
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_pos, pos_label = 1)


# Compute Average Precision
# 0.4307 for LogisticRegression
# 0.4223 for SGD with native early stopping
average_precision_score(y_test, y_prob_pos, pos_label = 1)


# Compute Brier score
# 0.194 for LogisticRegression
# 0.1958 for SGD with native early stopping
brier_score_loss(y_test, y_prob_pos, pos_label = 1, sample_weight = sample_weight_test)


# Compute Brier skill score for each model, with dummy classifier as reference:
# 1 - (brier_score_model / brier_score_dummy) (1 is best, 0 is worst)




# Create dummy classifier pipeline
pipe_dummy = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("model_dummy", DummyClassifier(strategy = "prior"))
])
