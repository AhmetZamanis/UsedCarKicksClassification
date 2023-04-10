# Used cars kicks classification - Data cleaning
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("01_DataInspect.py").read())


# TOYOTA SCION into SCION
df.loc[df["Make"] == "TOYOTA SCION", "Make"] = "SCION"


# Recode NAs in color as NOT AVAIL
df.loc[pd.isna(df["Color"]), "Color"] = "NOT AVAIL"


# Replace "Manual" with MANUAL in transmission
df.loc[df["Transmission"] == "Manual", "Transmission"] = "MANUAL"


# Work out & impute transmission NAs from car
df.loc[pd.isna(df["Transmission"]), ["VehYear", "Make", "Model", "Trim", "SubModel"]]

transmission_nas = [
  "AUTO", "MANUAL", "MANUAL", "MANUAL", "AUTO", "MANUAL", "MANUAL", "AUTO", "AUTO"]
  
df.loc[pd.isna(df["Transmission"]), "Transmission"] = transmission_nas


# Crosscheck the WheelTypeID and WheelType columns
df[["WheelTypeID", "WheelType"]]
pd.isnull(df.loc[pd.isnull(df["WheelTypeID"]), "WheelType"]).sum() # All WheelTypeID
# NANs are also WheelType NANs
pd.isnull(df.loc[pd.isnull(df["WheelType"]), "WheelTypeID"]).sum() # All except
# 5 WheelType NANs are also WheelTypeID NANs. Those 5 are ID 0

(df.loc[df["WheelTypeID"] == "1", "WheelType"] == "Alloy").sum() # "1" = "Alloy"
(df.loc[df["WheelTypeID"] == "2", "WheelType"] == "Covers").sum() # "2" = "Covers"
(df.loc[df["WheelTypeID"] == "3", "WheelType"] == "Special").sum() # "3" = "Special"
df.loc[df["WheelTypeID"] == "0", "WheelType"] # "0" = "Nan"

df.loc[pd.isnull(df["WheelType"]), "WheelType"] = "Other"
df = df.drop("WheelTypeID", axis = 1)


# Work out the 5 missing Nationality values from make
df.loc[pd.isnull(df["Nationality"]), "Make"]

df.loc[df["Nationality"] == "TOP LINE ASIAN", "Make"].value_counts()
df.loc[df["Nationality"] == "OTHER ASIAN", "Make"].value_counts()
df.loc[df["Nationality"] == "OTHER", "Make"].value_counts()

nationality_nas = ["AMERICAN", "AMERICAN", "OTHER ASIAN", "AMERICAN", "AMERICAN"]
df.loc[pd.isnull(df["Nationality"]), "Nationality"] = nationality_nas


# Work out the 5 missing Size values from make & model
df.loc[pd.isnull(df["Size"]), ["VehYear", "Make", "Model"]]

df.loc[df["Model"].str.contains("SIERRA")]
df.loc[df["Model"].str.contains("NITRO 4WD")]
df.loc[df["Model"].str.contains("ELANTRA")]
df.loc[df["Model"].str.contains("PATRIOT 2WD")]

size_nas = ["LARGE TRUCK", "MEDIUM SUV", "MEDIUM", "SMALL SUV", "SMALL SUV"]
df.loc[pd.isnull(df["Size"]), "Size"] = size_nas


# Drop top 3 American column
df = df.drop("TopThreeAmericanName", axis = 1)


# Crosscheck MMR NAs
df.loc[pd.isnull(df['MMRAcquisitionAuctionAveragePrice'])] # All other MMRs missing
df.loc[pd.isnull(df['MMRAcquisitionAuctionCleanPrice'])] # All other MMRs missing
df.loc[pd.isnull(df['MMRAcquisitionRetailAveragePrice'])] # All other MMRs missing
df.loc[pd.isnull(df['MMRAcquisitonRetailCleanPrice'])] # All other MMRs missing
df.loc[pd.isnull(df['MMRCurrentAuctionAveragePrice'])] # Some Acquisition MMRs present
df.loc[pd.isnull(df['MMRCurrentAuctionCleanPrice'])] # Some Acquisition MMRs present
df.loc[pd.isnull(df['MMRCurrentRetailAveragePrice'])] # Some Acquisition MMRs present
df.loc[pd.isnull(df['MMRCurrentRetailCleanPrice'])] # Some Acquisition MMRs present


# Drop rows with NAs in MRMRAcquisition
df = df.dropna(subset = [
  'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
  'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice'])
  
  
# Drop current MMR features to make the exercise more realistic
df = df.drop([
  'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
  'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice'], axis = 1)


# Fill NAs in PRIMEUNIT and AUCGUART with UNKNOWN.
df.loc[pd.isnull(df["PRIMEUNIT"]), "PRIMEUNIT"] = "UNKNOWN"
df.loc[pd.isnull(df["AUCGUART"]), "AUCGUART"] = "UNKNOWN"


# Drop NAs in purchase price
df = df.dropna(subset = "VehBCost")
