# Used cars kicks classification - Feature engineering
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("02_DataClean.py").read())


from feature_engine.encoding import OneHotEncoder
from feature_engine.creation import CyclicalFeatures


# Time features from date: Purchase year, month, day of week?
df["PurchaseYear"] = df["PurchDate"].dt.year
df["PurchaseMonth"] = df["PurchDate"].dt.month
df["PurchaseDay"] = df["PurchDate"].dt.weekday
df = df.drop("PurchDate", axis = 1)


# Features from model, trim, submodel: Drivetrain, engine type, chassis type, 
# doors
# Engine types from Model: V6, V8, I4, I-4, 4C, 6C
df["EngineV6"] = df["Model"].str.contains("V6").astype(int)
df["EngineV8"] = df["Model"].str.contains("V8").astype(int)
df["EngineI4"] = df["Model"].str.contains("I4|I-4", regex = True).astype(int)
df["Engine4C"] = df["Model"].str.contains("4C").astype(int)
df["Engine6C"] = df["Model"].str.contains("6C").astype(int)

# Drivetrain types from Model: 2WD, 4WD, AWD, FWD, RWD
df["2WD"] = df["Model"].str.contains("2WD").astype(int)
df["4WD"] = df["Model"].str.contains("4WD").astype(int)
df["AWD"] = df["Model"].str.contains("AWD").astype(int)
df["FWD"] = df["Model"].str.contains("FWD").astype(int)
df["RWD"] = df["Model"].str.contains("RWD").astype(int)

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

# Displacement from SubModel: 1.0L - 8.0L. If single numeric column, would raise
# too many NAs. If dummies, would add too many columns.


# Drop model, trim, submodel
df = df.drop(["Model", "Trim", "SubModel"], axis = 1)


# Miles per year: VehOdo / VehicleAge
df["MilesPerYear"] = df["VehOdo"] / df["VehicleAge"]
df.loc[df["MilesPerYear"] == np.inf, "MilesPerYear"] = df["VehOdo"] # Replace inf values


# Premiums / discounts paid on MMR prices: VehBCost - MMR price / MMR price
df["PremiumAuctionAvg"] = (df["VehBCost"] - df["MMRAcquisitionAuctionAveragePrice"]) / df[
  "MMRAcquisitionAuctionAveragePrice"]

df["PremiumAuctionClean"] = (df["VehBCost"] - df["MMRAcquisitionAuctionCleanPrice"]) / df[
  "MMRAcquisitionAuctionCleanPrice"]

df["PremiumRetailAvg"] = (df["VehBCost"] - df["MMRAcquisitionRetailAveragePrice"]) / df[
  "MMRAcquisitionRetailAveragePrice"]
  
df["PremiumRetailClean"] = (df["VehBCost"] - df["MMRAcquisitonRetailCleanPrice"]) / df[
  "MMRAcquisitonRetailCleanPrice"]

df[["PremiumAuctionAvg", "PremiumAuctionClean", "PremiumRetailClean", "PremiumRetailClean"]].describe()

df.loc[df["PremiumAuctionAvg"] < -0.9]


# Warranty ratio:  WarrantyCost / VehBCost
df["WarrantyRatio"] = df["WarrantyCost"] / df["VehBCost"]
df["WarrantyRatio"].sort_values(ascending = False)
df.loc[df["WarrantyRatio"] > 2] # Drop the one obs with purchase price = 1
df = df.loc[df["VehBCost"] != 1].copy()


# One hot encode:
# Auction, VehYear, Color, Transmission, WheelType, Nationality, Size, PRIMEUNIT,
# AUCGUART, PurchaseYear
encoder_onehot = OneHotEncoder(
  drop_last = True,
  drop_last_binary = True,
  variables = ['Auction', 'VehYear', 'Color', 'Transmission', 'WheelType', 
  'Nationality', 'Size', 'PurchaseYear'],
  ignore_format = True)
df = encoder_onehot.fit_transform(df)

# Only use known values for PRIMEUNIT and AUCGUART, ignore unknowns
df["PRIMEUNIT_YES"] = (df["PRIMEUNIT"] == "YES").astype(int)
df["PRIMEUNIT_NO"] = (df["PRIMEUNIT"] == "NO").astype(int)
df["AUCGUART_GREEN"] = (df["AUCGUART"] == "GREEN").astype(int)
df["AUCGUART_RED"] = (df["AUCGUART"] == "RED").astype(int)
df = df.drop(["PRIMEUNIT", "AUCGUART"], axis = 1)


# Cyclical encode:, PurchaseMonth, PurchaseDay
encoder_cyclical = CyclicalFeatures(
  variables = ["PurchaseMonth", "PurchaseDay"],
  drop_original = True)
df = encoder_cyclical.fit_transform(df)
