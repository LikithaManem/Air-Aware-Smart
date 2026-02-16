

import pandas as pd
import matplotlib.pyplot as plt



# Step 1: Create raw data

data = {
    "Name": ["Ramesh", "Suresh", "Lakshmi", "Fatima", "John"],
    "Years": [20, 35, None, 40, 28],
    "Dust": ["High", "Medium", "High", None, "Low"],
    "Traffic": ["Increased", "Increased", "Same", "Increased", None],
    "Health": ["Yes", "Yes", "No", "Yes", None]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)



# Step 2: Preprocessing


# Fill numerical column with mean
df["Years"] = df["Years"].fillna(df["Years"].mean())

# Fill categorical columns with mode
df["Dust"] = df["Dust"].fillna(df["Dust"].mode()[0])
df["Traffic"] = df["Traffic"].fillna(df["Traffic"].mode()[0])
df["Health"] = df["Health"].fillna(df["Health"].mode()[0])

print("\nAfter Preprocessing:")
print(df)



# Step 3: Feature Engineering
# Create resident_type from number of years

df["Resident_Type"] = df["Years"].apply(lambda x: "Very Old" if x >= 30 else "Old")

print("\nAfter Feature Engineering:")
print(df)



# Step 4: Basic EDA
print("\nStatistical Summary:")
print(df.describe())



# Step 5: Visualization

dust_count = df["Dust"].value_counts()

plt.bar(dust_count.index, dust_count.values)
plt.title("Dust Level from Survey")
plt.xlabel("Dust Category")
plt.ylabel("Number of People")
plt.show()