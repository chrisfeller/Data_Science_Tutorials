### Hands-On Machine Learning with Scikit-Learn & TensorFlow
#### Aurelien Geron
#### March 2018

---
#### Preface
**Github Repo**
* textbook repo: https://github.com/ageron/handson-ml

---
#### Chapter 1 | The Machine Learning Landscape
**What is Machine Learning**
* Definition: Machine Learning is the science (and art) of programming computers so they can learn from data.
* Alternative Definition: Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed.
* A Third Definition: A computer program is said to learn from experience *E* with respect to some task *T* and some performance measure *P*, if its performance on *T*, as measured by *P*, improves with some experience *E*.

**Why use Machine Learning**
* Machine Learning is great for:
    * Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better.
    * Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution.
    * Fluctuating environments: a Machine Learning system can adapt to new data.
    * Getting insights about complex problems and large amounts of data.

**Types of Machine Learning Systems**
* Despite the wide range of Machine Learning types, it is most useful to classify them in broad categories based on:
    * Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning)
    * Whether or not they can learn incrementally on the fly (online versus batch learning)
    * Whether they work by simply comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model, much like scientists do (instance-based versus model-based learning).
    *Many Machine Learning techniques are a combination of these*

**Supervised vs. Unsupervised Learning**
* Machine Learning systems can be classified according to the amount and type of supervision they get during training.
* There are four main categories:
    1) Supervised Learning
        * In supervised learning, the training data you feed to the algorithm includes the desired solutions, called *labels.*
        * Typical supervised learning tasks are classification and regression.
        * Example: spam/ham classification or regression to predict the selling price of a car.
        * Most important supervised learning algorithms:
            1) k-Nearest Neighbors
            2) Linear Regression
            3) Logistic Regression
            4) Support Vector Machines (SVMs)
            5) Decision Trees and Random Forests
            6) Neural networks (although some neural network architectures can be unsupervised, such as autoencoders and restricted Boltzmann machines. They can also be semisupervised, such as in deep belief networks and unsupervised pre-training).

    2) Unsupervised Learning
        * In unsupervised learning, the training data is unlabeled.
        * Most important unsupervised learning algorithms:
            1) Clustering:
                * k-Means
                * Hierarchical Cluster Analysis (HCA)
                * Expectation Maximization
            2) Visualization and Dimensionality Reduction
                * Principal Component Analysis (PCA)
                * Kernel PCA
                * Locally-Linear Embedding (LLE)
                * t-distributed Stochastic Neighbor Embedding (t-SNE)
            3) Association Rule Learning
                * Apriori
                * Eclat

    3) Semisupervised Learning
        * Semisupervised learning involves algorithms that can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data.
        * Most semisupervised learning algorithms are combinations of unsupervised and supervised algorithms.

    4) Reinforcement Learning
        * The learning system, called an *agent* in this context, can observe the environment, select and perform actions, and get *rewards* in return (or *penalties* in the form of negative rewards). It must then learn by itself what is the best strategy, called a *policy*, to get the most reward over time. A policy defines what action the agent should choose when it is in the given situation.

**Batch and Online Learning**
* Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data.
* Two categories:
    1) Batch Learning
        * In batch learning, the system is incapable of learning incrementally: it must be trained using all of the available data.
            * Since this generally takes a great deal of time and computing resources, it is typically done offline.
        * First, the system is trained, and then it is launched into production and runs without learning anymore; it just applies that it has learned.
            *This is called offline learning*

    2) Online Learning
        * In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called *mini-batches.*
        * Online learning is great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously.

**Instance-Based Versus Model-Based Learning**
* The last way to categorize Machine Learning systems is by how they generalize to new data.
* There are two main approaches to generalization:
    1) Instance-based Learning
        * The system learns the examples by heart, then generalizes to new cases using a similarity measure.

    2) Model-Based Learning
        * Model-based learning involves building a model based on examples and then using that model to make predictions.

**Model Evaluation**
* There are two main ways to evaluate a model:
    1) You can define a *utility function (or fitness function)* that measures how good your model is.
    2) Define a *cost function* that measures how bad your model is.

**General Workflow of a Machine Learning Problem**
1) Study the data
2) Select a model
3) Train the model on the training data (i.e., the learning algorithm searches for the model parameter values that minimize a cost function).
4) Apply the model to make predictions on new cases, hoping that the model generalizes well to new data.

**Main Challenges of Machine Learning**
* The two main things that can go wrong in machine learning:
    1) Bad Algorithm
        * Overfitting the Training Data
            * Overfitting Definition: The model performs well on the training data, but it does not generalize well.
        * Underfitting the Training Data
            * Occurs when your model is too simple to learn the underlying structure of the data.
    2) Bad Data
        * Non-representative Training Data
            * In order to generalize well, it is crucial that your training data be representative of the new cases you want to generalize to.
            * If the sample is too small, you will have *sampling noise* (i.e., non-representative data as a result of chance).
            * Even if the sample is large, it can be non-representative if the sampling method is flawed (this is called *sampling bias*).
        * Poor Quality Data
            * If your training data is full of errors, outliers, and noise, it will make it harder for the system to detect the underlying patterns, so your system is less likely to perform well.
        * Irrelevant Features
             * Garbage In, Garbage Out
             * Your system will only be capable of learning if the training data contains enough relevant features and not too many irrelevant ones.

---
#### Chapter 2 | End-to-End Machine Learning Project
**Introduction**
* In this chapter, we will complete a machine learning project end-to-end using the California Housing Prices dataset.

**Step 1: Frame the Problem and Look at the Big Picture**
* The first question to ask your boss is what exactly is the business objective; building a model is probably not the end goal.
    * How does the company expect to use and benefit from this model?
    * This is important because it will determine how you frame the problem, what algorithms you will select, what performance measures you will use to evaluate your model, and how much effort you should spend tweaking it.
    * **Business Objective**: Build a model to predict the median housing price of a California district. This prediction will be fed to another machine learning system, which will determine whether it is worth investing in a given area or not.
* The next question to ask is what the current solution looks like (if any).
    * **Current Solution**: The district housing prices are currently estimated manually by experts using a set of complex rules. These experts usually have a 10% error.
* Next, decide how the problem should be framed (supervised/unsupervised, online/offline, etc.).
    * **Frame Problem**: This is a supervised multivariate regression problem that can probably be solved best with an offline (batch) algorithm since housing data is far from streaming.
* How will you measure performance?
    * A typical performance measure for regression problems is the Root Mean Square Error (RMSE).
        * RMSE gives an idea of how much error the system typically makes in its predictions, with a higher weight for large errors.
        * RMSE is sensitive to outliers, but when outliers are exceptionally rare and our data is normally distributed, the RMSE performs very well and is generally preferred.  
    * **Performance Metric**: Root Mean Square Error (RMSE)
* Check assumptions.
    * It is good practice to list and verify any assumptions you've made such as what is the output supposed to look like? What is the end user expecting from this model?

**Step 2: Get the Data**
* Data for this exercise will come from a comma-separated value (CSV) file called *housing.csv*.
~~~
import pandas as pd
housing = pd.read_csv('housing.csv')
~~~
* The `info()` method is useful to get a quick description of the data, in particular the total number of rows, each attribute's data type, and number of non-null values:
~~~
housing.info()
~~~
* Our dataset has 10 features (`longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value`, and `ocean_proximity`).
* Notice that `total_bedrooms` is the only feature with missing data (207 null values).
* Also notice that all features are numerical (`float64`) except for the `ocean_proximity` feature, which is categorical. To see how many categories exist in `ocean_proximity` use `.value_counts()`:
~~~
housing['ocean_proximity'].value_counts()
~~~
* To get a summary table of the numerical features, use `.describe()`:
~~~
housing.describe()
~~~
* Another quick way to get a feel for your numerical features is to plot a histogram of each:
~~~
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(12,9))
plt.show()
~~~
* Now that we have a general idea of what the data looks like, we should set aside a test set.
    * The test set is typically 20% of the dataset.
~~~
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
~~~
* Another way to train/test split would be to stratify based on the `median_income` feature so that both the train and test sets would be made up of equal distributions of `median_income`:
```
import numpy as np

# Create income categories
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

# Plot histogram of income categories
housing['income_cat'].hist()

# Stratify the train/test split based on new income categories
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Examine split of income categories between train/test sets
strat_test_set['income_cat'].value_counts()/len(strat_test_set)

# Drop income category feature as we only care about income as a continuous variable
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
```

**Step 3: Explore the Data**
* We will only examine the training set in this section:
~~~
housing = strat_test_set.copy()
~~~
* Since we have latitude and longitude data, let's plot the geographical data:
    * Add the argument `alpha` to better visualize where there is a high density of data points.
~~~
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
~~~
* A more advance step would involve adding housing price to the above plot:
~~~
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population']/100, label='population', figsize=(10,7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
~~~
* Now let's look at correlations within the features, specifically how each feature correlates with the median house value.
    * The correlation coefficient only measures linear correlations and thus may completely miss out on nonlinear relationships. It also has nothing to do with the slope of the relationship.
~~~
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
~~~
* There are some important features but most are raw counts (i.e., `total_rooms`). It would be more helpful if we transformed these features into `rooms_per_household`.
~~~
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']
~~~
* Examine the correlation matrix again w/ new features:
~~~
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
~~~

**Step 4: Prepare the Data**
* First, split the predictors and the target vector:
~~~
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_test_set['median_house_value'].copy()
~~~
* Most Machine Learning algorithms cannot work with missing features. As noted earlier, the `total_bedrooms` features has some missing values. There are three options to deal with this:
    1) Get rid of all observations with missing values in the `total_bedrooms` features
    ```
    housing.dropna(subset=['total_bedrooms'])
    ```
    2) Get rid of the entire feature `total_bedrooms`
    ```
    housing.drop('total_bedrooms', axis=1)
    ```
    3) Set the missing values to some value (zero, the mean, the median, etc. )
    ```
    housing['total_bedrooms'].fillna(median, inplace=True)
    ```
* sklearn provides a handy class to take care of missing values:
```
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')

# Median can only be computed for numerical features so let's filter out all categorical features
housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)

# To put that back into a pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
```

**scikit-learn Design**
* The main design principles of sciki-learn's API are:
1) Consistency
    * All objects share a consistent and simple interface
        * *Estimators*: Any object that can estimate some parameters based on a dataset is called an estimator (e.g., Imputer is an estimator). This is the `.fit()` step.
        * *Transformers*: Some estimators (such as Imputer) can also transform a dataset. This is the `.transform()`
        * *Predictors*: Some estimators are capable of making predictions given a dataset. Takes a dataset of new instances and returns a dataset of corresponding predictions. This is the `.predict()` step. Also has a `.score()` method to measure the quality of the predictions given a test set.
2) Inspection
    * All the estimator's hyperparameters are accessible directly via public instance variables (e.g., `imputer.strategy`), and all the estimator's learned parameters are also accessible via public instance variables with an underscore suffix (e.g., imputer.statistics_)
3) Nonproliferation of classes: Datasets are represented as NumPy arrays or SciPy sparse matrices, instead of homemade classes. Hyperparameters are just regular Python strings or numbers.
4) Composition: Existing building blocks are reused as much as possible. For example, it is easy to create a Pipeline estimator from an arbitrary sequence of transformers followed by a final estimator.
5) Sensible defaults: scikit-learn provides reasonable default values for most parameters, making it easy to create a baseline working system quickly.

**What is the difference between a NumPy matrix/array and a SciPy sparse matrix?**
* A SciPy sparse matrix only stores the location of the nonzero elements.
* Very useful when you have a sparse NumPy matrix, which is inefficient because you are wasting memory storing 0's.
* To convert a SciPy sparse matrix to a NumPy matrix use:
```
sparse_matrix.to_array()
```

**Custom Transformers**
* Although scikit-learn provides many useful transformers, you will need to write your own for tasks such as custom cleanup operations.
* To create a custom transformer that will fit into a Pipeline, all you need is to create a class and implement three methods:
    1) `.fit()` - returning `self`
    2) `.transform()`
    3) `.fit_transform()`
        * You can get this last one for free by simply adding `TransformerMixin` as a base class.
* If you add `BaseEstimator` as a base class (and avoid `*args` and `**kwargs` in your constructor) you will get two extra methods (`get_params()` and `set_params()`) that will be useful for automatic hyperparameter tuning.
* Here is an example of of a small transformer encompassing some of the other transformations we've discussed in this chapter:
```
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, housefuls_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # Cannot contain *args or **kwargs
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

**Feature Scaling**
* One of the most important transformations you need to apply to your data is feature scaling.
* With few exceptions, Machine Learning algorithms don't perform well when the input numerical attributes have very different scales.
* Note that scaling the target values is generally not required.
* There are two common ways to get all attributes to have the same scale:
    1) min-max scaling
        * AKA: Normalization
        * Values are shifted and rescaled so that they end up ranging from 0 to 1.
        * Done by subtracting the min value and dividing by the max minus the min.
        * sklearn provides a transformer called `MinMaxScaler` for this.
            * It has a `feature_range` hyperparameter that lets you change the range if you don't want it 0-1 for some reason.
    2) standardization
        * Done by subtracting the mean value, and then dividing by the variance so that the resulting distribution has unit variance.
        * Standardized values have a mean of zero.
        * Unlike normalization, standardization does not bound values by a specific range.
            * This may be a problem for some neural networks that expect an input value ranging form 0 to 1.
        * Standardization is much less affected by outliers as compared to normalization.
        * sklearn provides a transformer called `StandardScaler` for standardization.
* As with all transformations, it is important to fit the scalers to the training data only, not to the full dataset (including the test set).

**Transformation Pipelines (AKA Pipelines)**
* A sequence of data preprocessing components is called a data pipeline.
* Components typically run asynchronously and each component is fairly self-contained.
* scikit-learn provides a `Pipeline` class to help with such sequences of transformations.
* Simple Example:
```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```
* The Pipeline constructor takes a list of (names/estimators) pairs defining a sequence of steps.
* All but the last estimator must be a transformer (i.e., they must have a `fit_transform()` method).
* The names of each step can be anything you like as long as they don't include any double underscores `__`.
* When you call the pipeline's `fit()` method, it calls `fit_transform()` sequentially on all transformers, passing the output of each call as the parameter to the next call, until it reaches the final estimator, for which it just calls the `fit()` method.

**Pipeline Example**
* Let's create a full Pipeline to take in a pandas DataFrame, transform the continuous features (impute nulls, create the new features from the `CombineAttributeAdder()` transformer we built above, and scale the features), while also transforming the categorical features (one-hot-encoding).
* First, we have to build a custom transformer to handle pandas DataFrames:
```
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names:

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values
```
* Next, let's build two separate pipelines from the continuous features and the categorical features:
```
num_attribs = list(housing_nums)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombineAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding='onehot-dense')),
    ])
```
* Lastly, we must join the two pipelines to create a single pipeline.
    * We can do so by using scikit-learn's `FeatureUnion` class.
    * We give it a list of transformers (which can be entire transformer pipelines); when its `transform()` method is called, it runs each transformer's `transform()` method in parallel, waits for their output, and then concatenates them and returns the result.
```
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
    ])

# Run the whole pipelines
housing_prepared = full_pipeline.fit_transform(housing)
```

**Step 5: Select and Train a Model**
* Cross-Validation on a decision tree regressor model:
```
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# fit a decision tree model
tree_reg = DecisionTreeRegressor()

# 10-fold cross validation
scores = cross_val_score(treereg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)

# Look at scores of cross-val
def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard Deviation:', scores.std())

tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
```

**Saving Trained Models**
* You can easily save scikit-learn models by using Python's `pickle` module, or using `sklearn.externals.joblib`, which is more efficient at serializing large NumPY arrays:
```
from sklearn.externs import joblib

# to save a trained model
joblib.dum(my_model, 'my_model.pkl')

# to load a trained model
my_model_loaded = joblib.load('my_model.pkl')
```

**Step 6: Fine-Tune Your Model**
* One way to fine-tune the hyperparameters of your model is to do so manually, until you find a great combination. This would be incredibly tedious and time intensive.
* Instead use sklearn's `GridSearchCV` to search potential hyperparameters for you.
* All you have to do is tell it which hyperparameters you want it to experiment with, and what values to try out, and it will evaluate all the possible combinations of hyperparameter values, using cross-validation.
* Example using a random forest regression model:
```
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]},
    ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)

# To access the best parameters
grid_search.best_params_

# To access the best estimator directly
grid_search.best_estimator_
```
* When you have no idea what value a hyperparameter should have, a simple approach is to try out consecutive powers of 10.
* If `GridSearchCV` is initialized with `refit=True` (which is the default), then once it finds the best estimator using cross-validation, it retrains it on the whole training set.
    * This is usually a good idea since feeding it more data will likely improve its performance.
* An alternative to `GridSearchCV` is `RandomizedSearchCV`, which can be used when your hyperparameter search space is large.
    * Instead of trying out all possible combinations, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.

**Step 7: Launch, Monitor, and Maintain Your System**
* It is important to write monitoring code to check your system's live performance at regular intervals once it is in production.
* Models tend to 'rot' as data evolves over time, so you must have checks to see if the model needs to be trained on fresh data.
* Evaluating your system's performance will require sampling the system's predictions and evaluating them.
    * This is best done by a human analyst with contextual knowledge.
* You should also make sure you evaluate the system's input data quality.
* Lastly, you should generally train your model on a regular basis using fresh data. You should automate this process as much as possible.
