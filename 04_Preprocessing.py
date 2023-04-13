# Used cars kicks classification - Preprocessing
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("03_FeatureEngineering.py").read())


from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder


# Target and features
y = df["IsBadBuy"].astype(int)
x = df.drop("IsBadBuy", axis = 1)


# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size = 0.2, random_state = 1923, stratify = y
)


# Preprocessing pipeline: Target encoders - scaler

# Target encoder: VNZIP1 with hierarchy (apply first)
hier_states = pd.DataFrame(x["VNST"]).rename({"VNST": "HIER_VNZIP1_1"}, axis = 1)

encode_target_zip = TargetEncoder(
  cols = ["VNZIP1"],
  hierarchy = hier_states
)

# Target encoder: Make, BYRNO, VNST (apply second):
encode_target = TargetEncoder(
  cols = ["Make", "BYRNO", "VNST"]
)

# Scale & center
scaler_minmax = MinMaxScaler()

# Pipeline
pipe_process = Pipeline(steps = [
  ("target_encoder_zipcode", encode_target_zip),
  ("target_encoder", encode_target),
  ("minmax_scaler", scaler_minmax)
  ])


# Inner validation method for hyperparameter tuning
cv_kfold = RepeatedStratifiedKFold(
  n_splits = 5, n_repeats = 2, random_state = 1923)
