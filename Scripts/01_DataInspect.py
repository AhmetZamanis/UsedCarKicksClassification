# Used cars kicks classification - Data inspection
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


import pandas as pd
import numpy as np
from scipy.io.arff import loadarff


# Set print options
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)


# Load raw data
raw_data = loadarff("./RawData/kick.arff")


# Convert to pandas dataframe
df = pd.DataFrame(raw_data[0])


# Print first & last 5 rows, all columns
df


# Convert object columns from bytes to string
object_cols = df.select_dtypes(["object"]).columns

for column in object_cols:
  df[column] = df[column].apply(lambda x: x.decode("utf-8"))
del column


# Replace "?" values with NAs
for column in object_cols:
  df.loc[df[column] == "?", column] = np.nan
del column


# Missing values
pd.isnull(df).sum()


# Target is imbalanced, 12.3% positive.
df["IsBadBuy"].value_counts(normalize = True)


# Purchase date is in UNIX timestamp. Convert to datetime
df["PurchDate"]


# 3 unique auctioneers. ADESA, MANHEIM and other. Mostly MANHEIM.
df["Auction"].value_counts(normalize = True)


# Vehicle years from 2001 to 2010. Base year is 2010. Crosscheck both columns.
df[["VehYear", "VehicleAge"]].describe()


# There are brands with few observations. Recode some into main brands, like
# Toyota SCION to Toyota? Or target encode?
df["Make"].value_counts()


# There are 1063 unique models, many with only 1 observation.
df["Model"].value_counts() 


# 135 trims, many with only 1 observation. 2360 missing values.
df["Trim"].value_counts() 


# 864 submodels, many with only 1 observation. 8 missing values.
df["SubModel"].value_counts() 

# 2718 unique model & submodel combinations, many with only 1 observation.
(df["Model"] + " " + df["SubModel"]).value_counts()


# 3000+ unique cars in dataset (some could be different spellings of same car)
(df["Model"] + df["Trim"] + df["SubModel"]).value_counts()


# Features to extract from model, trim, submodel strings: Drivetrain, engine type,
# chassis type, doors, dummies for top N most frequent cars?
df[["Make", "Model", "Trim", "SubModel"]]


# 8 missing values for color, recode them as NOT AVAIL
df["Color"].value_counts()


# 9 missing values from transmission. Try to work them out from car model. 1
# manual spelled differently.
df["Transmission"].value_counts()


# 3169 missing values in WheelTypeID, 3174 in WheelType. Crosscheck these columns.
df["WheelTypeID"].value_counts()
df["WheelType"].value_counts()


# 5 missing values in Nationality. Work these out from car model. Also look at OTHER
# types, work them out if easy.
df["Nationality"].value_counts()


# 5 missing values in size. Work these out from the model.
df["Size"].value_counts()


# Unnecessary column?  5 missing values that can be worked out from model.
df["TopThreeAmericanName"].value_counts()


# MMR stands for Manheim Market Report. MMR prices are calculated from Manheim
# car sales, taking into account car condition. Outliers are excluded. Could
# crosscheck the missing values with cars of same model.
pd.isnull(df[['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
       'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
       'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
       'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']]).sum()

# Some MMR values are zero or one, 828 of them (not including current MMRs)
df.loc[
  (df["MMRAcquisitionAuctionAveragePrice"] == 0) |
  (df["MMRAcquisitionAuctionCleanPrice"] == 0) |
  (df["MMRAcquisitionRetailAveragePrice"] == 0) |
  (df["MMRAcquisitonRetailCleanPrice"] == 0)] # 828 zeroes
  
df.loc[
  (df["MMRAcquisitionAuctionAveragePrice"] < 100) |
  (df["MMRAcquisitionAuctionCleanPrice"] < 100) |
  (df["MMRAcquisitionRetailAveragePrice"] < 100) |
  (df["MMRAcquisitonRetailCleanPrice"] < 100)]

  
# 95% missing column. Missing values are possibly NO. YES means there was unusual
# demand for the car.
df["PRIMEUNIT"].value_counts(normalize = True)


# 95% missing column. AUCGUART is the vehicle inspection level at auction. Green
# is inspected, yellow is partial information available, red is you buy what you
# see. Could assume yellow for missing values.
df["AUCGUART"].value_counts(normalize = True)


# BYRNO is buyer no. 74 unique buyers, some with only 1 observation.
df["BYRNO"].value_counts()


# VNZIP1 is zipcode of purchase location, 153 locations, some with only 1 obs.
df["VNZIP1"].value_counts()


# VNST is purchase state. Keep all states & dummy encode
df["VNST"].value_counts()


# VehBCost is purchase price. 68 missing values, could be estimated from other cost
# columns?
df["VehBCost"].describe()


# Warranty cost is for 36 months, until 36k miles
df["WarrantyCost"].describe()


# Convert purchase date from UNIX to datetime
df["PurchDate"] = pd.to_datetime(df["PurchDate"], unit = "s")


# String purchase year almost always matches PurchYear - VehYear.
purch_year = df["PurchDate"].dt.year
veh_year = df["VehYear"]
((purch_year - veh_year) == df["VehicleAge"]).sum()
