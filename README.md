# titanic-demo
Demo for a simple app to host a fitted model trained on a subset of the titanic disaster dataset, 
which will predict if a person with a given Age, Gender, and wheter or not they travelled alone,
would have survived the distaster.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SoftStackFactory/titanic-demo/master)



# Titanic Model
<hr>

## Imports

#### Import `pandas` and `numpy`  for Data Manipulation


```python
import pandas as pd
import numpy as np
```

<br>

#### import `matplotlib` and `seaborn` for data visualization


```python
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.style.use('seaborn-muted')
```

<br>

#### Import `SkLearn` functions to model Data


```python
# To Split Dataset
from sklearn.model_selection import train_test_split


# Import Model
from sklearn.ensemble import GradientBoostingClassifier


# Model Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
```

<br>

#### Import `pickle` to save finished model


```python
import pickle
```

<br>
<br>
<br>
<br>
<hr>

## Load Data


```python
train = pd.read_csv('./data/titanic.csv')
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>ParCh</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>



<br>
<br>
<br>
<br>

# 1. Explatory Data Analysis
<hr>

## Women and Children?

### `Gender` vs. `Survived`


```python
fig = plt.figure(figsize=(8,6))
ax = fig.gca()
sns.countplot(train['Gender'], hue=train['Survived'], ax=ax)
plt.show()
```


![png](output_21_0.png)


### `Age` and `Gender` VS. `Survived`


```python
fig = plt.figure(figsize=(8,6))
ax = fig.gca()
sns.violinplot("Gender", "Age", hue = "Survived", data = train, split = True)

plt.show()
```


![png](output_23_0.png)


<br>
<br>
<br>
<br>

# 2. Feature Engineering and Preprocessing
<hr>

## Where You travelling Alone ?


```python
train['FamilySize'] = train['SibSp'] + train['ParCh'] + 1
```


```python
train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, "IsAlone"] = 1
```

<br>


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>ParCh</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>IsAlone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>Mrs</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<br>

## Drop All Columns except for `Age`, `Gender`, and `IsAlone`


```python
train = train.drop(['PassengerId', 'Pclass', 'Name', 'SibSp',
       'ParCh', 'Ticket', 'Fare', 'Cabin', 'Title','FamilySize', 'Embarked'], axis=1)
```

<br>

## Label Encoding
Most machine learning models cannot interpret string values directly, we must encode them into numerical values!

### Convert `Gender` into a binary column: `IsFemale`


```python
train['IsFemale'] = train['Gender'].replace(['male','female'],[0,1])
```


```python
train = train.drop(['Gender'],axis=1)
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>IsAlone</th>
      <th>IsFemale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>26.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<br>
<br>
<br>
<br>

# 3. Create Model
<hr>

## A. Split Dataset into `train` and `test`
**We split the dataset into two sets:**
* `X_train` and `y_train`: Will be passed into the model to learn the patterns in the data
* `X_test` and `y_test`: Will be used to test the validity of the model's predictions.


```python
features = train.drop(['Survived'], axis=1)
target = train['Survived']
```


```python
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.90, random_state=100)
```

    C:\Users\David\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)
    

### Check Dimensions


```python
print("X_train:", X_train.shape, "y_train:", y_train.shape)
```

    X_train: (801, 3) y_train: (801,)
    


```python
print("X_test:", X_test.shape, "y_test:", y_test.shape)
```

    X_test: (90, 3) y_test: (90,)
    

<br>

## B. Train Model

The parameters that I passed in are called `hyperparameters`, these were "discovered" through a process called cross-validation, which can be applied with the `SciKit_Learn` function `GridSearchCV`


```python
model = GradientBoostingClassifier(learning_rate=0.02, n_estimators=200, max_features=None)
```


```python
model.fit(X_train.values, y_train.values)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.02, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=200,
                  n_iter_no_change=None, presort='auto', random_state=None,
                  subsample=1.0, tol=0.0001, validation_fraction=0.1,
                  verbose=0, warm_start=False)



## C. Make Predictions on X_test


```python
predictions = model.predict(X_test)
```

<br>
<br>
<br>
<br>

# 4. Evaluate Model Performance 

<hr>

### A. Check `accuracy` of `predictions` by comparing to  `y_test`


```python
accuracy_score(y_test, predictions)
```




    0.7333333333333333



### B. Check `confusion_matrix` of `predictions`


```python
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
#     prep work
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    
#     make Heatmap and set custom tick marks
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    
#     Set plot labels
    ax.set_ylabel('True label', fontsize=fontsize*1.5)
    ax.set_xlabel('Predicted label',fontsize=fontsize*1.5)
    
    return fig
```


```python
cm = confusion_matrix(y_test,predictions)
labels = ["Perished","Survived"]
```


```python
_ = print_confusion_matrix(confusion_matrix = cm, class_names=labels)
```


![png](output_62_0.png)


<br>
<br>
<br>
<br>

# 5. Export Model
<hr>


```python
from sklearn.externals import joblib
joblib.dump(model, '../models/titanic_grad_boost.joblib') 
```




    ['../models/titanic_grad_boost.joblib']
