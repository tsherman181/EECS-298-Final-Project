"""
Link to download LA Crime Data
https://catalog.data.gov/dataset/crime-data-from-2020-to-present

Below I have done some basic data cleaning which includes:
    * Simplifying the racial classifications and dropping rows missing this data
    * Categorizing the crimes as either violent or non-violent
        (according to the UCR-COMPSTAT classifcations which can be found here: https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8/about_data)
    * Totalling the number of crimes the suspect was charged with
"""

import pandas as pd

df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
df["Vict Descent"] = df["Vict Descent"].replace(
    {
        "A": "Asian",
        "B": "Black",
        "C": "Asian",
        "D": "Asian",
        "F": "Asian",
        "G": "Hispanic",
        "H": "Hispanic",
        "I": "Native American/Pacific Islander",
        "J": "Asian",
        "K": "Asian",
        "L": "Asian",
        "O": "Other",
        "P": "Native American/Pacific Islander",
        "S": "Native American/Pacific Islander",
        "U": "Native American/Pacific Islander",
        "V": "Asian",
        "W": "White",
        "X": "Unknown",
        "Z": "Asian",
    }
)
df = df.dropna(subset=["Vict Descent"])
df["Violent/Non-Violent Crime"] = (
    df["Crm Cd 1"]
    .isin(
        {
            110,
            113,
            121,
            122,
            815,
            820,
            821,
            210,
            220,
            230,
            231,
            235,
            236,
            250,
            251,
            761,
            926,
        }
    )
    .astype(int)
    .replace({1: "Violent", 0: "Non-Violent"})
)

df["Number of Crimes"] = (
    df[["Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4"]].notna().sum(axis=1)
)
