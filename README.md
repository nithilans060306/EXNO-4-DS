# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv("bmi.csv")

df_null_sum=df.isnull().sum()
df_null_sum
```
### Output:
<img width="201" height="167" alt="image" src="https://github.com/user-attachments/assets/eed55a93-28a5-488b-ba3f-a079123cb3cb" />

```python
df.dropna()
```
### Output:
<img width="520" height="480" alt="image" src="https://github.com/user-attachments/assets/89e64b91-22c7-4a66-9e67-b99cf8c1ff59" />


```python
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
### Output:
<img width="209" height="112" alt="image" src="https://github.com/user-attachments/assets/872b482a-1640-4e4e-908b-b45b9e43a429" />

```python
df1 = df.copy()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']])
df1
```

### Output:
<img width="537" height="494" alt="image" src="https://github.com/user-attachments/assets/cb7ad47a-caa8-4aa4-828d-d75220b7dace" />

```python
df3 = df.copy()

from sklearn.preprocessing import MaxAbsScaler
sc = MaxAbsScaler()
df3[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']])
df3
```

### Output:
<img width="494" height="487" alt="image" src="https://github.com/user-attachments/assets/d1d28e28-a4a9-4583-9a47-fa73f9783627" />

```python
df4 = df.copy()

from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
df4[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']    ])
df4
```

### Output:
<img width="495" height="481" alt="image" src="https://github.com/user-attachments/assets/98212d9e-ca2d-404a-ac70-58da42a9fda0" />

```python
df = pd.read_csv("income(1) (1).csv")
df_null_sum=df.isnull().sum()
df_null_sum
```
### Output:
<img width="383" height="384" alt="image" src="https://github.com/user-attachments/assets/3cdad9c4-0bf5-4cb4-8698-fabf2bb8c104" />

```python
# Chi Square Test
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns]
```

### Output:
<img width="1072" height="488" alt="image" src="https://github.com/user-attachments/assets/4f23e142-407c-4b2d-8e4c-db1c35549710" />

```python
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
### Output:
<img width="984" height="475" alt="image" src="https://github.com/user-attachments/assets/664bb0be-c599-457b-8102-c78a8cd1e13d" />

```python
from sklearn.feature_selection import SelectKBest, chi2, f_classif

X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_new_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi square test")
print(selected_features_chi2)
```

### Output:
<img width="840" height="151" alt="image" src="https://github.com/user-attachments/assets/8416412e-2be5-4887-8811-2c244bd4eb2c" />

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss', 'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
accuracy
```

# Output:
<img width="291" height="77" alt="image" src="https://github.com/user-attachments/assets/90fe38a6-dc90-48d0-a3e9-0ffb696fcd8d" />

```python
from skfeature.function.similarity_based import fisher_score

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)

X = df.drop(columns=['SalStat'])
y = df['SalStat']

k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)

selected_features_anova = X.columns[selector_anova.get_support()]
print("Selected features using ANOVA")
print(selected_features_anova)

```
### Output:

<img width="905" height="95" alt="image" src="https://github.com/user-attachments/assets/11ec561b-933c-4fe2-9da5-ff92a93f8497" />

# RESULT:
Successfully completed the Feature Scaling and Selection
