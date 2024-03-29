import pandas as pd

# Assuming your CSV file is named 'data.csv' and is located in the current directory
file_path = "Census_Microdata.csv"

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Now you can work with the DataFrame 'df'
print(df.head())
