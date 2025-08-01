import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("HR-Employee-Attrition.csv")

print(df.head())
print(df.isnull())
print(df.isnull().sum()) 
print(df.info())

sns.set_theme(style="whitegrid")
sns.set_context("notebook") 

# Attrition count
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Distribution")
plt.show()

# Percentage distribution
attrition_percent = df['Attrition'].value_counts(normalize=True) * 100
print(attrition_percent)

# Insight:
# Most employees are not leaving (~84%), indicating class imbalance.

# Gender
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()

# Job Role
plt.figure(figsize=(12, 6))
sns.countplot(x='JobRole', data=df)
plt.title("Job Role Distribution")
plt.xticks(rotation=45)
plt.show()

# Age distribution
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

# Monthly Income
sns.histplot(df['MonthlyIncome'], kde=True)
plt.title("Monthly Income Distribution")
plt.show()

# Distance from Home
sns.histplot(df['DistanceFromHome'], kde=True)
plt.title("Distance from Home")
plt.show()

# Total Working Years
sns.histplot(df['TotalWorkingYears'], kde=True)
plt.title("Total Working Years")
plt.show()

# Attrition count by Gender
sns.countplot(data=df, x='Gender', hue='Attrition')
plt.title("Attrition by Gender")
plt.show()

# Attrition count by Department
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title("Attrition by Department")
plt.show()

# Boxplot: Monthly Income distribution by Attrition
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome')
plt.title("Income vs Attrition")
plt.show()

# Swarmplot: Age vs Attrition
sns.swarmplot(data=df, x='Attrition', y='Age')
plt.title("Age vs Attrition")
plt.show()

# Scatterplot: Total Working Years vs Age
sns.scatterplot(data=df, x='Age', y='TotalWorkingYears', hue='Attrition')
plt.title("Age vs Total Working Years")
plt.show()

# Correlation heatmap (for overall numerical relationships)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# These are common drops in the HR Attrition dataset
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df.drop(columns=drop_cols, inplace=True)

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Check categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(categorical_cols)

df['LoyaltyScore'] = df['YearsInCurrentRole'] / (df['TotalWorkingYears'] + 1)
df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='JobRole', y='MonthlyIncome')
plt.title("Monthly Income by Job Role")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_income_by_jobrole.png", dpi=300)
plt.show()

# Key Insights:
# Male employees are more dominant in R&D.
# Employees with low income are more likely to leave.
# Age has a weak negative correlation with attrition.
# LoyaltyScore can help identify at-risk employees.