# Used cars kicks classification - Preprocessing
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("./Scripts/03_FeatureEngineering.py").read())


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

# Target encoder: VNZIP1 with hierarchy
hier_states = pd.DataFrame(x["VNST"]).rename({"VNST": "HIER_VNZIP1_1"}, axis = 1)
encode_target_zip = TargetEncoder(cols = ["VNZIP1"], hierarchy = hier_states)


# Target encoder: ModelSubModel with hierarchy
hier_submodels = pd.DataFrame(x[["Make", "Model"]]).rename({
  "Make": "HIER_ModelSubModel_1", "Model": "HIER_ModelSubModel_2"}, axis = 1)
encode_target_submodel = TargetEncoder(
  cols = ["ModelSubModel"], hierarchy = hier_submodels)


# Target encoder: Model with hierarchy 
hier_models = pd.DataFrame(x["Make"]).rename({"Make": "HIER_Model_1"}, axis = 1)
encode_target_model = TargetEncoder(cols = ["Model"], hierarchy = hier_models)


# Target encoder: Make, BYRNO, VNST (apply last):
encode_target = TargetEncoder(cols = ["Make", "BYRNO", "VNST"])


# Scale & center
scaler_minmax = MinMaxScaler()


# Pipeline
pipe_process = Pipeline(steps = [
  ("target_encoder_zipcode", encode_target_zip),
  ("target_encoder_submodel", encode_target_submodel),
  ("target_encoder_model", encode_target_model),
  ("target_encoder", encode_target),
  ("minmax_scaler", scaler_minmax)
  ])


# Inner validation method for hyperparameter tuning
cv_kfold = StratifiedKFold(n_splits = 3)
