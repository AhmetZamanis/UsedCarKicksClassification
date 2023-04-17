

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
