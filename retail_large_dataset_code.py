import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# Load dataset
df = pd.read_csv(
    r"C:\Users\vijay\OneDrive\Desktop\project 3\retail_large_dataset.csv",
    encoding='unicode_escape'
)

# Shape of dataset
print("Shape of data:", df.shape)

# Preview data
print(df.head())

# Check duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# Data info
print("\nData Info:")
df.info()

# Convert column datatype (handle errors safely)
df['final_price'] = pd.to_numeric(df['final_price'], errors='coerce')

# Check datatype
print("\nData type of final_price:", df['final_price'].dtype)

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:\n", df.describe())

# Additional stats
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nStd:\n", df.std(numeric_only=True))
print("\nVariance:\n", df.var(numeric_only=True))
print("\nSkewness:\n", df.skew(numeric_only=True))
print("\nKurtosis:\n", df.kurtosis(numeric_only=True))

# Histogram
df["age"].plot(kind='hist', bins=20)
plt.title("Age Distribution")
plt.show()

sns.histplot(data=df, x='age', bins=20)
plt.title("Age Distribution (Seaborn)")
plt.show()

# Boxplot
df.boxplot(column='age')
plt.title("Age Boxplot")
plt.show()

# Correct IQR calculation for age
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

max_limit = Q3 + 1.5 * IQR
print("Age Max Limit:", max_limit)

# Outliers
print(df[df['age'] > max_limit])

# Product price boxplot
df.boxplot(column="product_price")
plt.title("Product Price Boxplot")
plt.show()

# Correct IQR for product_price
Q1 = df['product_price'].quantile(0.25)
Q3 = df['product_price'].quantile(0.75)
IQR = Q3 - Q1

max_limit = Q3 + 1.5 * IQR
print("Product Price Max Limit:", max_limit)

print(df[df["product_price"] > max_limit])

# Other boxplots
cols = ['quantity', 'discount_percentage', 'final_price', 'delivery_days']

for col in cols:
    df.boxplot(column=col)
    plt.title(f"{col} Boxplot")
    plt.show()

# Categorical plots
cat_cols = ['product_category', 'customer_segment', 'payment_method', 'return_status']

for col in cat_cols:
    df[col].value_counts().plot(kind='bar')
    plt.title(f"{col} Count")
    plt.xticks(rotation=45)
    plt.show()

# Describe categorical properly
print(df.describe(include='object'))