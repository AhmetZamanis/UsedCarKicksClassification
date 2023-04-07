# Used cars kicks classification - Data cleaning
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("01_DataInspect.py").read())


# Convert purchase date from UNIX to datetime
df["PurchDate"] = pd.to_datetime(df["PurchDate"], unit = "s")


# String purchase year doesn't match vehicle age most of the time. Base = 2010.
# Keep both in I guess.
age_calculated = df["VehYear"].apply(lambda x: 2010 - x)
(age_calculated == df["VehicleAge"]).sum()


# TOYOTA SCION into SCION
df.loc[df["Make"] == "TOYOTA SCION", "Make"] = "SCION"


# Recode NAs in color as NOT AVAIL
df.loc[pd.isna(df["Color"]), "Color"] = "NOT AVAIL"


# Replace "Manual" with MANUAL in transmission
df.loc[df["Transmission"] == "Manual", "Transmission"] = "MANUAL"


# Work out transmission NAs from car
df.loc[pd.isna(df["Transmission"]), ["Make", "Model", "Trim", "SubModel"]]
