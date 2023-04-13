

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
