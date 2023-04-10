# Used cars kicks classification - Feature engineering
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("02_DataClean.py").read())

# Time features from date: Purchase year, month, day of week?
df["PurchaseYear"] = df["PurchDate"].dt.year
df["PurchaseMonth"] = df["PurchDate"].dt.month
df["PurchaseDay"] = df["PurchDate"].dt.weekday
df = df.drop("PurchDate", axis = 1)


# Features from model, trim, submodel: Drivetrain, engine type, chassis type, 
# doors

# Engine types from Model: V6, V8, I4, I-4, 4C, 6C

# Drivetrain types from Model: 2WD, 4WD, AWD, FWD, RWD

# Chassis types from SubModel: WAGON, SEDAN, COUPE, HATCHBACK, CONVERTIBLE
df.loc[pd.isnull(df["SubModel"]), "Model"]

df["ChassisWagon"] = df["SubModel"].str.contains("WAGON")
df.loc[pd.isnull(df["ChassisWagon"]), "ChassisWagon"] = [
  False, False, False, False, False, False, False, True]
df["ChassisWagon"] = df["ChassisWagon"].astype(int)

df["ChassisSedan"] = df["SubModel"].str.contains("SEDAN")
df.loc[pd.isnull(df["ChassisSedan"]), "ChassisSedan"] = [
  True, True, False, True, True, True, False, False]
df["ChassisSedan"] = df["ChassisSedan"].astype(int)

df["ChassisCoupe"] = df["SubModel"].str.contains("COUPE")
df.loc[pd.isnull(df["ChassisCoupe"]), "ChassisCoupe"] = False
df["ChassisCoupe"] = df["ChassisCoupe"].astype(int)

df["ChassisHatch"] = df["SubModel"].str.contains("HATCHBACK")
df.loc[pd.isnull(df["ChassisHatch"]), "ChassisHatch"] = False
df["ChassisHatch"] = df["ChassisHatch"].astype(int)

df["ChassisConvertible"] = df["SubModel"].str.contains("CONVERTIBLE")
df.loc[pd.isnull(df["ChassisConvertible"]), "ChassisConvertible"] = False
df["ChassisConvertible"] = df["ChassisConvertible"].astype(int)


# Door types from SubModel: 4D
df["FourDoors"] = df["SubModel"].str.contains("4D")
df.loc[pd.isnull(df["FourDoors"]), "FourDoors"] = [
  True, True, False, True, True, True, False, False]
df["FourDoors"] = df["FourDoors"].astype(int)


# Displacement from SubModel: 1.0L - 8.0L


# Drop model, trim, submodel



# Miles per year: VehOdo / VehicleAge
df["MilesPerYear"] = df["VehOdo"] / df["VehicleAge"]
df.loc[df["MilesPerYear"] == np.inf, "MilesPerYear"] = df["VehOdo"] # Replace inf values


# Premiums / discounts paid: VehBCost / MMR prices


# Warranty ratio:  WarrantyCost / VehBCost
df["WarrantyRatio"] = df["WarrantyCost"] / df["VehBCost"]
df["WarrantyRatio"].sort_values(ascending = False)
df.loc[df["WarrantyRatio"] > 2] # Leave the one obs with purchase price = 1

