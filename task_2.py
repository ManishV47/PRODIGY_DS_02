import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
titanic_data = pd.read_csv('titanicTrain.csv')

# Display the first few rows of the dataset to understand its structure
print(titanic_data.head())


# Check for missing values
print(titanic_data.isnull().sum())

# Dealing with missing values in 'Age' column by imputing median age
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Handling missing values in 'Embarked' column by filling with the most common value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Dropping 'Cabin' column due to high missing


# Survival rate by sex
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival Count by Sex')
plt.show()

# Survival rate by passenger class
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title('Survival Count by Passenger Class')
plt.show()

# Survival rate by age
sns.histplot(data=titanic_data, x='Age', hue='Survived', kde=True)
plt.title('Survival Count by Age')
plt.show()

# Survival rate by embarked port
sns.countplot(x='Embarked', hue='Survived', data=titanic_data)
plt.title('Survival Count by Embarked Port')
plt.show()

# Correlation matrix for numerical variables
corr_matrix = titanic_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()