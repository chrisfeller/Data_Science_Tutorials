### Data School: Building an Effective Machine Learning Workflow with scikit-learn
#### June 2020

---
#### Live Course Session 1
**Link**
* [Video Link](https://www.crowdcast.io/e/ml-course/1)

**Part 1: Review of the basic machine learning workflow**
```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
df = pd.read_csv('http://bit.ly/kaggletrain', nrows=10)

X = df[['Parch', 'Fare']]
y = df['Survived']

logreg = LogisticRegression(solver='liblinear', random_state=1)

# 3-fold cross validation
cross_val_score(logreg, X, y, cv=3, scoring='accuracy').mean()

logreg.fit(X, y)

df_new = pd.read_csv('http://bit.ly/kaggletest', nrows=10)
X_new = df_new[['Parch', 'Fare']]

logreg.predict(X_new)
```

**Part 2: Encode Categorical Data**
* One-Hot Encoding
    ```
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()
    ohe.fit_transform(df[['Embarked']])
    ```
    * One-Hot Encoder returns a sparse matrix.
    * To return a dense matrix run `OneHotEncoder(sparse=False)` but don't use this in production code.
    * To get the categorical order use `ohe.categories_`

**Part 3: Using ColumnTransformer and Pipeline**
```
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

cols = ['Parch', 'Fare', 'Embarked', 'Sex']

df = pd.read_csv('http://bit.ly/kaggletrain', nrows=10)
X = df[cols]
y = df['Survived']

df_new = pd.read_csv('http://bit.ly/kaggletest', nrows=10)
X_new = df_new[cols]

ohe = OneHotEncoder()

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    remainder='passthrough') # passthrough argument assures columns not used in column transformer are passed through to feature matrix

logreg = LogisticRegression(solver='liblinear', random_state=1)

pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.predict(X_new)
```

**Part 4: Encoding Text Data**
```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name']

df = pd.read_csv('http://bit.ly/kaggletrain', nrows=10)
X = df[cols]
y = df['Survived']

df_new = pd.read_csv('http://bit.ly/kaggletest', nrows=10)
X_new = df_new[cols]

ohe = OneHotEncoder()

vect = CountVectorizer()

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    remainder='passthrough') # passthrough argument assures columns not used in column transformer are passed through to feature matrix

logreg = LogisticRegression(solver='liblinear', random_state=1)

pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.predict(X_new)
```

---
#### Office Hours 1
**Link**
* [Video Link](https://www.crowdcast.io/e/ml-course/2)

**How to handle convergence warnings?**
* Simplest solution is to change the solver selection in your model.

**How do you drop redundant column in OneHotEncoder?**
* `ohe = OneHotEncoder(drop=False)`
* Useful when you have perfectly correlated features with un-regularized regression or neural networks. Other than that, won't gain much of a benefit and the downside is the model will struggle to handle new categories not existing in the training data.

**What encoding should you use with ordinal features?**
* Use `OrdinalEncoder`:
    ```
    from sklearn.preprocessing import OrdinalEncoder
    # you must specify the order of the categories
    ore = OrdinalEncoder(categproes=['first', 'second', 'third']])
    ore.fit_transform(df)
    ```

**When should you use LabelEncoder?**
* Only with targets never with features.
* New sklearn versions can now handle string targets so there's really no need to use `LabelEncder()` anymore.

**In ColumnTransformer, what are the other options for the 'remainder' parameter?**
* `passthrough`: Assures columns not used in column transformer are passed through to feature matrix
* `drop`: Removes columns not used in column transformer. *This is the default*

**How do you get the column names for the output of ColumnTransformer?**
* `ct.get_feature_names()`

**How can you specify columns for a ColumnTransformer without typing them out one-by-one?**
* By position:
    ```
    ct = make_column_transformer(
        (ohe, [2, 3]),
        (vect, 4),
        remainder='passthrough'
        )
    ```
* By slice:
    ```
    ct = make_column_transformer(
        (ohe, [slice(2, 4)),
        (vect, 4),
        remainder='passthrough'
        )
    ```
* Use `ColumnSelector` for regex matching column names:
    ```
    from sklearn.compose import make_column_selector
    cs = make_column_selector(pattern='nba_')

    ct = make_column_transformer(
        (ohe, cs),
        (vect, 4),
        remainder='passthrough'
        )
    ```
    * `ColumnSelector` also can select columns by dtypes

**How do you access each step in a pipeline?**
* `pipe.named_steps.keys()`
* You can then select individual steps via:
    `pipe.named_steps.logisticregression`

**What is the difference between `make_pipeline()` and `Pipeline()`?**
* `Pipeline` allows you to manually name steps, while `make_pipeline()` automatically names each step.
* `make_pipeline()` code is more readable.
* The only time to use `Pipeline()` is when you have customer transformers that you want to gridsearch over and need to name them manually.

**How do you pass multiple columns to `CountVectorizer()`?**
* Via two separate tuples:
    ```
    ct = make_column_transformer(
        (vect, 'Name'),
        (vect), 'Sex'
        )
    ```

**How can you save a fit pipeline?**
* Option #1: Pickle File:
    ```
    import pickle
    with open(pipe.pickle', 'wb') as f:
        pickle.dump(pipe, f)

    with open('pipe.pickle', 'rb') as f:
        pipe_from_pickle = pickle.load(f)
    pipe_from_pickle.predict(X_new)
    ```
* Option #2: Joblib:
    ```
    import joblib
    joblib.dump(pipe, 'pipe.joblib')

    pipe_from_joblib = joblib.load('pipe.joblib')
    pipe_from_joblib.predict(X_new)
    ```
* Joblib is more efficient and popular with sklearn objects
* Both pickle and joblib are version specific so you need to read in the file in whatever version you wrote it out.

---
#### Live Course Session 2
**Link**
* [Video Link](https://www.crowdcast.io/e/ml-course/3)

**Part 5: Handling Missing Values**
```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']

df = pd.read_csv('http://bit.ly/kaggletrain', nrows=10)
X = df[cols]
y = df['Survived']

df_new = pd.read_csv('http://bit.ly/kaggletest', nrows=10)
X_new = df_new[cols]

imp = SimpleImputer()

ohe = OneHotEncoder()

vect = CountVectorizer()

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age']),
    remainder='passthrough') # passthrough argument assures columns not used in column transformer are passed through to feature matrix

logreg = LogisticRegression(solver='liblinear', random_state=1)

pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.predict(X_new)
```
* To add an indicator column, which is a binary indicator of if the row had a missing value for a given column:
    * `imp = SimpleImputer(add_indicator=True)`
    * This imputes the missing value in the original column AND creates a new binary column saying if there was a missing value.
    * Useful for data missing not at random
* Other imputation strategies:
    - `IterativeImputer()`
    - `KNNImputer()`

**Part 6:Switching to the Full Dataset**
* Combine OHE and simple imputer in to a separate column_transformer.
```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']

df = pd.read_csv('http://bit.ly/kaggletrain')

X = df[cols]
y = df['Survived']

df_new = pd.read_csv('http://bit.ly/kaggletest')
X_new = df_new[cols]

# for categorical variables with null values, impute 'missing'
# potentially a better approach than mode as the variable might not be missing at random
imp_constant = SimpleImputer(strategy = 'constant', fill_value='missing')
ohe = OneHotEncoder()

imp_ohe = make_pipeline(imp_constant, ohe)
vect = CountVectorizer()
imp = SimpleImputer()

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    remainder='passthrough') # passthrough argument assures columns not used in column transformer are passed through to feature matrix

logreg = LogisticRegression(solver='liblinear', random_state=1)

pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.predict(X_new)
```
* To get the learned attributes for imputation:
    * `ct.named_transformers_.simpleimputer.statistics_`

**Part 7: Evaluating and Tuning a Pipeline**
* Cross-validate pipeline:
```
from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
```
* Gridsearch parameters in pipeline:
```
from sklearn.model_selection import GridSearchCV

params = {}
params['logisticregression__penalty'] = ['l1', 'l2']
params['logisticregression__C'] = [0.1, 1, 10]

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)

results = pd.DataFrame(grid.cv_results_)\
            .sort_values('rank_test_score')
```
* Gridsearch transformers in pipeline:
```
# tune drop parameter of OHE
params['columntransformer__pipeline__onehotencoder__drop'] = [None, 'first']

# tune ngram range in countvectorizer
params['columntransformer__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]

# tune add indicator in simple imputer
params['columntransformer__simpleimputer_add_indicator'] = ['False, True']

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)

results = pd.DataFrame(grid.cv_results_)\
            .sort_values('rank_test_score')
```       
* To get best score:
`grid.best_score_`
* To get parameters:
`grid.best_params_`
* To make predictions on best model:
`grid.predict(X_new)`


---
#### Office Hours 2
**Link**
* [Video Link](https://www.crowdcast.io/e/ml-course/4)

**Imputation and OHE**
* You must always combine imputation and OHE in its own separate pipeline within the all-encompassing pipeline. If you instead have imputation and ohe in separate steps you will simply hstack the imputed column and the ohe columns, which will lead to an error.
* Notice how it's split out below:
```
imp_constant = SimpleImputer(strategy = 'constant', fill_value='missing')
ohe = OneHotEncoder()
imp_ohe = make_pipeline(imp_constant, ohe)

vect = CountVectorizer()
imp = SimpleImputer()

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    remainder='passthrough')
```

**Stratified K-Fold**
* Rows are not randomized in each fold! So if your rows are in order prior to the split, shuffle them first.

**FeatureUnion vs. ColumnTransformer**
* `FeatureUnion()` was a precursor to `ColumnTransformer()` and thus you should just use `ColumnTransformer()`

**Feature Selection**
- `SelectPercentile()`
- `SelectFromModel()`
    - Will select features via a model (logistic regression, random forest, etc.) coefficients or feature importances.
- For both of these, you must tune the threshold in gridsearch.

**Outlier Detection**
- `RobustScaler()`: Scaler that is robust to outliers
- Sklearn does not currently support transformers that remove rows so most outlier detection/removal won't work in pipelines.
- One approach could be to impute outliers with 'missing' in training data prior to pipeline.

**CustomTransformers**
* Example: Put a floor on Age and Fare
```
from sklearn.preprocessing import FunctionTransformer

get_floor = FunctionTransformer(np.floor)

make_column_transformer(
    (get_floor, ['Age', 'Fare']),
    remainder='drop')
```
* Example: Create new column equal to the first character in `Cabin`
```
def first_letter(df):
    return pd.DataFrame(df).apply(lambda x: x.str.slice(0,1))

get_first_letter = FunctionTransformer(first_letter)

make_column_transformer(
    (get_first_letter, ['Cabin']), remainder='drop')
```
