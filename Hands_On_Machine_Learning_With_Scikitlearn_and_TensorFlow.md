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

---
#### Chapter 3 | Classification
**Data**
* This chapter will utilize the MNIST handwritten dataset:
```
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

# Create feature matrix X and target vector y
X, y = mnist['data'], mnist['target']
```
* Perform a train/test split and then shuffle the training data:
    * Shuffling the training data guarantees that all cross-validation folds will be randomly similar.
    * This is important because some learning algorithms are sensitive to the order of the training instances.
```
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Shuffle training data
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
```
* As a quick example of classification, let's attempt to classify whether an image is a 5 or not using Stochastic Gradient Descent:
```
from sklearn.linear_model import SGDClassifier

# Create T/F for whether a label is 5 or not
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Build model
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Predict on a random observation in test data
test_digit = X_test[10]
sgd_clf.predict([test_digit])
```

**Classification Performance Measures**
* The best way to measure the performance of a classifier (and really most models) is cross validation:
```
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
```
* However, accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with imbalanced classes (i.e., skewed datasets in which some classes are much more frequent than others).

**Confusion Matrix**
* The best way to evaluate a classifier is via a confusion matrix:
```
# Make predictions in test data
y_pred = sgd_clf.predict(X_train)

# Build confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_pred)
```
* Each row in a confusion matrix represents an actual class, while each column represents a predicted class.
* sklearn's confusion matrix is displayed as followed:

| TN  | FP  |
|---|---|
| **FN**  | **TP**  |

* A perfect classifier would have only true positives and true negatives.
* From the confusion matrix you can calculate many more granular metrics:
    * Precision
        * Accuracy of the positive predictions
        * Equation: TP/(TP + FP)
        * In sklearn:
        ```
        from sklearn.metrics import precision_score
        precision_score(y_test, y_pred)
        ```
    * Recall (AKA: Sensitivity, AKA: True Positive Rate
        * Ratio of positive instances that are correctly detected by the classifier.
        * Equation: TP/(TP + FN)
        * In sklearn:
        ```
        from sklearn.metrics import recall_score
        recall_score(y_test, y_pred)
        ```
    * F-Score
        * Combination of precision and recall
        * The harmonic mean of precision and recall
        * Will only be high if both precision and recall are high
        * Will favor classifiers that have similar precision and recall.
        * In sklearn:
        ```
        from sklearn.metrics import f1_score
        f1_score(y_test, y_pred)
        ```
* Unfortunately, you can't have it both ways: increasing precision reduces recall, and vice versa. This is called the precision/recall tradeoff.

**The ROC Curve**
* The Receiver Operating Characteristic (ROC) curve is another tool use to evaluate classifiers.
* Plots the true positive rate (AKA recall) against the false positive rate.
    * The FPR is the ratio of negative instances that are incorrectly classified as positive.
    * In other words, the ROC curve plots sensitivity (recall) versus 1 - specificity.
* To plot the ROC curve:
```
from sklearn.metrics import roc_curve

y_proba = sgd_clf.predict_proba([test_digit])

fpr, tpr, thresholds = roc_curve(y_test_5, y_proba[:,1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

plot_roc_curve(fpr, tpr, label=None)
```
* One way to compare classifiers is to measure the area under the curve (AUC).
    * A perfect classifier has an AUC equal to one, whereas a purely random classifier will have an AUC equal to 0.5.
    * In sklearn:
    ```
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_test_5, y_proba[:,1])
    ```
**Multiclass Classification**
* Multiclass classifiers (also called multinomial classifiers) can distinguish between more than two classes.
* Some algorithsm (Random Forests, Naive Bayes) are capable of handling multiple classes directly, while others (Support Vector Machines and Logistic Regression) are strictly binary classifiers.
* However there are strategies to turn binary classifiers into multiclass classifiers:
    * One-versus-all (aka one versus-the-rest)
    * One-versus-one
* sklearn automatically uses one-versus-all for all binary classifiers when you try and predict multi-class problems (except for Support Vector Machines in which case it uses one-versus-one)

---
#### Chapter 4 | Training Model
**Linear Regression**
* There are two very different way to train a linear regression model:
    1) The Normal Equation
        * A direct 'closed-form' equation that directly computes the model parameters that best fit the model to the training set.
            * i.e., the model parameters that minimize the cost function over the training set.
        * The Normal Equation is a mathematical equation that gives the result directly.
        * The Normal Equation gets very slow when the number of features grows large (e.g., 100,000)
    2) Gradient Descent
        * Iterative optimization approach, that gradually tweaks the model parameters to minimize the cost function over the training set, eventually converging to the same set of parameters as the Normal Equation.
        * The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function.
        * An important hyperparameter in Gradient Descent is the size of the steps, determined by the learning rate hyperparameter.
        * In most algorithms, Gradient Descent won't always find a global minimum, however Linear Regression is a convex function so there are no local minimum and thus if you let Gradient Descent run long enough it will find the global minimum.
        * When using Gradient Descent, you should ensure that all features have similar scale (e.g., use sklearn's `StandardScaler()`) to decrease the training time of linear regression models.
        * Better than the normal equation when there are a large number of features (> 100,000) or too many training instances to fit in memory.
        * There exists multiple types of Gradient Descent:
            1) Batch Gradient Descent
                * At each step, compute the gradients based on the full training set
            2) Stochastic Gradient Descent
                * At each step, compute the gradients based on just one instance.
                * Has a better chance of finding the global minimum because it has the ability to jump out of local minimum.
            3) Mini-Batch Gradient Descent
                * At each step, compute the gradients based on a small random set of instances called mini-batches.
* While the Normal Equation can only perform Linear Regression, Gradient Descent algorithms can be used to train many other models.
* Linear Regression in sklearn:
```
# Imports
import numpy as np
from sklearn.linear_model import LinearRegression

# Create fake data
X = 2 * np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100,1)

# Build Model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Print intercept and coefficients
print(lin_reg.intercept_, lin_reg.coef_)

# Make a prediction
X_new = np.array([[0], [2]])
lin_reg.predict(X_new)
```

**Polynomial Regression**
* You can use linear regression to fit nonlinear data by adding powers of each feature as new features in the model.
    * This approach is called Polynomial Regression.
* When there are multiple features, Polynomial Regression is capable of finding relationships between features, which is something a plain linear regression cannot do.
* To transform our training data by adding the square (2^nd^ degree polynomial) of each feature in the training data:
```
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
```

**Under/Overfitting Models**
* If your model is underfitting the training data, adding more training examples will not help.You need to use a more complex model or come up with better features.
* If your model is overfitting the training data you could either decrease the complexity of your model or provide it with more training data.

**The Bias/Variance Tradeoff**
* A model's error can be expressed as the sum of three very different errors:
    1) Bias: This part of the error is due to wrong assumptions, such as assuming that the data is linear when it's actually quadratic.
        * A high bias model is likely to underfit the training data.
    2) Variance: This part of the error is due to the model's excessive sensitivity to small variations in the training data.
        * A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance, and thus overfit the training data.
    3) Irreducible Error: This part of the error is due to the noise of the data itself.
        * The only way to reduce irreducible error is to clean up the data( fix the data sources, or detect and remove outliers).
* Increasing a model's complexity will typically increase its variance and reduce its bias. Conversely, reducing a model's complexity increases its bias and reduces its variance.

**Regularized Linear Models**
* A good way to reduce overfitting is to regularize the model (i.e., to constrain it).
* For linear models, regularization is typically achieved by constraining the weights of the model.
* The three ways to do this are:
    1) Ridge Regression
    2) Lasso Regression
    3) Elastic Net

**Ridge Regression**
* Ridge regression is a regularized version of linear regression, in which a regularization term is added to the cost function.
* The regularization term forces the learning algorithm to not only fit the data but also to keep the model weights as small as possible.
* The hyperparameter \$alpha$ controls how much you want to regularize.
    * \$alpha$ = 0 is equivalent to linear regression.
    * IF \$alpha$ is large then all coefficients end up close to zero.
* It is important to scale your data before performing ridge regression as it is sensitive to the scale of the features.
* Uses L2 norm.

**Lasso Regression**
* Least Absolute Shrinkage and Selection Operator Regression (Lasso Regression) is another regularized version of linear regression.
* An important characteristic of Lasso Regression is that it tends to completely eliminate the weights of the least important features (i.e., sets them to zero).
    * In other words, Lasso automatically performs feature selection and outputs a sparse model.
* Unlike Ridge Regression, Lasso uses the L1 norm.

**Elastic Net**
* Elastic net is the middle ground between Ridge Regression and Lasso Regression.
* The regularization term is a simple mix of both Ridge and Lasso's regularization terms, and you can control the mix ratio `r`.
    * When `r`=0, Elastic Net is equivalent to Ridge Regression, and when `r`=1, it is equivalent to Lasso Regression.

**Logistic Regression**
* Logistic regression (also called logit regression) is commonly used to estimate the probability that an instance belongs to a particular class.
* The estimates are calculated via a sigmoid function and are bounded between 0 and 1.
* There is no Normal Equation to solve the cost function of logistic regression so we instead use gradient descent.

**Softmax Regression**
* The logistic regression model can be generalized to support multiple classes directly, without having to train and combine multiple binary classifiers.
    * This is called the softmax regression or multinomial logistic regression.
* Just like logistic regression, the softmax regression classifier predicts the class with the highest estimated probability.

---
#### Chapter 5 | Support Vector Machines
**Introduction**
* SVM's are particularly well suited for classification of complex but small- or medium-sized datasets.

**Linear SVM Classification**
* You can think of SVM classifier as fitting the widest possible street between two classes.
    * This street is called large margin classification.
    * The decision boundary, or 'street', is fully determined (or 'supported') by the instances located on the edge of the street.
        * These instances are called the support vectors.
* Unlike Logistic Regression, SVM does not output probabilities for each class.

**Soft Margin Classification**
* If we strictly impose that all instances be off the street and on the right side, this is called hard margin classification.
* There are two main issues with hard margin classification:
    1) It only works if the data is linearly separable
    2) It's quite sensitive to outliers.
* As a solution, we use soft margin classification where we allow for a few instances to be misclassified.
    * In sklearn you can control this budget via the hyperparameter `C`.
        * A smaller `C` value leads to a wider street but more margin violations.

**Nonlinear SVM Classification**
* When a linear boundary isn't present in the data, you can still use SVM's thanks to the kernel trick.
* The kernel trick serves a similar purpose as adding many polynomial features without having to actually add them.
* Example of kernel trick using 3 degrees:
```
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)
```
* As a rule of thumb, you should always try the linear kernel first (because it's fastest). Next, try the Gaussian RBF kernel.

---
#### Chapter 6 | Decision Trees
**Introduction**
* Decision Trees are versatile machine learning algorithms that can perform both classification and regression tasks, even with multioutput tasks.
* They are also the foundation of some of the most powerful algorithms such as Random Forests and Gradient Boosting.
* sklearn uses the CART algorithm, which produces only binary trees.
* sklearn defaults to using Gini impurity over entropy and others.
    * Gini impurity is slightly faster to compute but gives similar answers to entropy.
    * Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees.

**Parametric vs. Non-Parametric Models**
* Decision trees are a form of non-parametric models, not because they don't have any parameters (they often have lots), but because the number of parameters is not determined prior to training, so the model structure is free to stick closely to the data.
* Parametric models, in contrast, such as linear models have a predetermined number of parameters, so its degrees of freedom is limited thus reducing the risk of overfitting (but increasing the risk of underfitting).

**Decision Tree in sklearn**
```
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:,2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```

**Visualizing Decision Trees**
```
from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file = 'iris_tree.dot',
    feature_names = iris.feature_names[2:],
    class_names = iris.target_names,
    rounded=True,
    filled=True
    )
```
* Then from the command line:
```
$ dot -Tpng iris_tree.dot > iris_tree.png
```

---
#### Chapter 7 | Ensemble Learning and Random Forests
**Introduction**
* Ensemble learning is a technique in which you aggregate the predictions of a group of predictors (such as classifiers and regressors).
* A group of predictors is called an ensemble.
* An ensemble learning algorithm is called an ensemble method.
* An example of an ensemble method is Random Forests, which is an ensemble of Decision Trees.
* Some of the most common ensemble methods are bagging, Random Forests, boosting, an stacking.

**Voting Classifiers**
* A very simple way to create an even better classifier is to aggregate the predictions of each classifier and predict the class that gets the most votes.
    * This majority-vote classifier is called a hard voting classifier.
* The voting classifier often achieves a higher accuracy than the best classifier in the ensemble
* Ensemble methods work best when the predictors are independent from one another as possible.
    * One way to be sure of this is to use very different algorithms. This increases the chances that they will make very different types of errors, improving the ensemble's accuracy.
* Example of voting classifier:
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import Voting Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

log_clf = LogisticRegression()
rf_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimator=[('lr', log_clf, ('rf', rf_clf), ('svc', svm_clf))],
    voting='hard'
    )

voting_clf.fit(X_train, y_train)

for clf in (log_clf, rf_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```
* If all classifiers are able to estimate class probabilities (i.e., they have a `predict_proba()` method), then you can tell sklearn to predict the class with the highest class probability, averaged over all the individual classifiers.
     * This is called soft voting.
     * Soft voting often achieves higher performance than hard voting because it gives weight to highly confident votes.
     * To use soft voting, all you need to do is replace `voting='hard'` with `voting='soft'` and ensure that all classifiers can estimate class probabilities.
     * This is not the case for the SVC class by default, so you need to set it's `probability` hyperparameter to `True` (this will make the SVC class use cross-validation to estimate class probabilities, slowing down training, and it will add a `predict_proba()` method.)

**Bagging and Pasting**
* One way to get a diverse set of classifiers is to use very different training algorithms.
* Another approach is to use the same training algorithm for every predictor, but to train them on different random subsets of the training set.
    * When sampling is performed with replacement, this method is called bagging (short for bootstrap aggregating).
    * When sampling is performed without replacement, it is called pasting.
* The ensemble that results form bagging has a similar bias but a lower variance than a single predictor trained on the original training set.
* Bagging and pasting scale well because each individual tree can be fit in parallel.
* Bagging often outperforms pasting.

**Our-of-Bag Evaluation**
* With bagging, some instances may be sampled several times for any given predictor, while others may not be sampled at all.
* By default `BaggingClassifier` samples *m* training instances with replacement (`bootstrap=True`), where *m* is the size of the training set.
* This means that only about 63% of the training instances are sampled on average for each predictor.
* The remaining 37% of the training instances that are not sampled are called out-of-bag (oob) instances.
    * These are not the same 37% for all predictors!
* You can then in theory use these oob to test your predictor.

**Random Forests**
* The Random Forest algorithm introduces extra randomness when growing trees; instead of searching for the very best feature when splitting a node, it searches for the best feature among a random subset of features.
* This results in a greater tree diversity, which trades a higher bias for a lower variance, generally yielding an overall better model.
* When you are growing a tree in a Random Forest, at each node only a random subset of the features is considered for splitting.

**Feature Importance**
* Another great quality of Random Forests is that they make it easy to measure the relative importance of each feature.
* sklearn measures a feature's importance by looking at how much the tree nodes that use that feature reduce impurity on average (across all trees in the forest).
    * More precisely, it is a weighted average, where each node's weight is equal to the number of training samples that are associated with it.
* sklearn computes this score automatically for each feature after training, then it scales the results so that the sum of all importances is equal to 1.
* Feature Importance Example:
```
from sklearn.datasets import load_iris
from sklearn.enseamble import RandomForestClassifier

iris = load_iris()

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf_clf.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], rf_clf.feature_importances_):
    print(name, score)
```

**Boosting**
* Boosting (originally called hypothesis boosting) refers to any ensemble method that can combine several weak learners into a strong learner.
* The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor.
* The two most popular boosting methods are AdaBoost and Gradient Boosting.
* The one drawback of boosting methods is they cannot be parallelized, since each predictor can only be trained after the previous predictor has been trained and evaluated.
    * I.e., it doesn't scale as well as other models.

**AdaBoost**
* To build an AdaBoost classifier, a first base classifier (such as a Decision Tree) is trained and used to make predictions on the training set. The relative weight of misclassified training instances is then increased. A second classifier is trained using the updated weights and gain it makes predictions on the training set, weights are updated and so on.

**Gradient Boosting**
* Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor.
* However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual error make by the previous predictor.
* In order to find the optimal number of trees, you can use early stopping.
* Stochastic Gradient Descent speeds up the training phase of Gradient Boosting.

**Stacking**
* Stacking is short for stacked generalization.
* It is based on a simple idea: instead of using trivial functions (such as hard voting) to aggregate the predictions of all predictors in an ensemble, why don't we train a model to perform this aggregation?
* This final model is called a blender.
* Unfortunately, sklearn does not support stacking directly.

---
#### Chapter 8 | Dimensionality Reduction
**The Curse of Dimensionality**
* Most points in a high-dimensional hypercube are very close to the extremes in at least one feature.
    * Fun fact: anyone you know is probably an extremest in at least one dimension (e.g., how much sugar they put in their coffee), if you consider enough dimensions.
* High dimensional datasets are at risk of being very sparse: most training instances are likely to be far away from each other.
    * This means that a new instance will likely be far away from any training instance, making predictions much less reliable than in lower dimensions, since they will be based on much larger extrapolations.
* The more dimensions the training set has, the greater the risk of overfitting it.

**Main Approaches for Dimensionality Reduction**
* There are two main approaches to reducing dimensionality:
    1) Projection
        * In most real-world problems, training instances are not spread out uniformly across all dimensions.
        * Many features are almost constant, while others are highly correlated.
        * As a result, all training instances actually lie within (or close to) a much lower-dimensional subspace of the high-dimensional space.
        * Thus, we can squash higher dimensions into lower dimensions.
    2) Manifold Learning
        * However, in some settings (i.e., the swiss roll dataset) it does not make sense to squash higher dimensions into lower dimensions.
            * You instead want to unroll the data.
* If you reduce the dimensionality of your training set before training a model, it will definitely speed up training, but it may not always lead to a better or simpler solution; it all depends on the dataset.

**PCA**
* Principal Component Analysis (PCA) is by far the most popular dimensionality reduction algorithm.
* PCA identifies the axis that accounts for the largest amount of variance in the training set. It also finds a second axis, orthogonal to the first one, that accounts for the largest amount of remaining variance. If it were a higher-dimensional dataset, PCA would also find a third axis, orthogonal to both previous axes, and a fourth, and a fifth, and so on - as many axes as the number of dimensions in the dataset.
* The unit vector that defines the i^th^ axis is called the i^th^ principal component.
* We can find principal components using a standard matrix factorization technique called Singular Value Decomposition (SVD) that can decompose the training set matrix **X** into the dot product of three matrices **U**, **E**, and **V**, where **V** contains all the principal components.
* PCA in sklearn:
```
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
```
* To access the explained variance of each principle component:
```
pca.explained_variance_ratio_
```
* Instead of arbitrarily choosing the number of dimensions to reduce down to, it is generally preferable to choose the number of dimensions that add up to a sufficiently large portion of the variance (e.g., 95%).
    * Unless you are reducing dimensionality for data visualization - in that case you will generally want to reduce the dimensionality down to 2 or 3.
    * You can do this via:
    ```
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    ```
    * An alternative way to do this would be to visualize a scree plot.
* Other types of dimensionality reduction:
    * t-Distribution Stochastic Neighbor Embedding (t-SNE): reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. It's mostly used for visualize clusters of instances in high-dimensional space.
    * Linear Discriminant Analysis (LDA): Actually a classification algorithm but during training it learns the most discriminative axes between the classes and these axes can then be used to define a hyperplane onto which to project the data.

---
#### Chapter 9 | Up and Running with TensorFlow
**Introduction**
* TensorFlow is a powerful open source software library for numerical computation, particularly well suited and fine-tuned for large-scale Machine Learning.
* It's basic principle is simple: you first define in Python a graph of computations to perform and then TensorFlow takes that graph and runs it efficiently using optimized C++ code.
    * Most importantly, it is possible to break up the graph into several chunks and run them in parallel across multiple CPUs or GPUs.
* TensorFlow also supports distributed computing, so you can train colossal neural networks on humongous training sets in a reasonable amount of time by splitting the computations across hundreds of servers.
* Developed by the Google Brain team and it powers many of Google's large-scale services.
* TensorFlow's clean design, scalability, flexibility, and great documentation (not to mention Google's name) quicklly boosted it to the top of the list of best open source Deep Learning libraries.
    * TensorFlow was designed to be flexible, scalable, and production-ready, and existing frameworks arguably hit only two out of the three of these.

**TensorFlow Highlights**
* IT runs not only on Windows, Linux, and macOS, but also on mobile devices, including both iOS and Android.
* It provides a very simple Python API, which is compatible with sklearn.
* It also provides another simple API to simplify building, training, and evaluating neural networks.
*  Several other high-level APIs have been built independently on top of TensorFlow, such as Keras and Pretty Tensor.
* It's main Python API offers much more flexibility (at the cost of higher complexity) to create all sorts of computations, including any neural network architecture you can think of.
* It includes highly efficient C++ implementations of many ML operations, particularly those needed to build neural networks.
* It provides several advanced optimization nodes to search for the parameters that minimize a cost function. These are very to use since TensorFlow automatically takes care of computing the gradients of the functions you define.
    * This is called automatic differentiating (or autodiff).
* It also comes with a great visualization tool called TensorBoard that allows you to browse through the computation graph, view learning curves, and more.
* Google also launched a cloud services to run TensorFlow graphs.

**Creating Your First Graph and Running It in a Session**
* Create a first graph to represent the function `f(x,y) = x**2*y+y+2`
```
import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2
```
* The previous code does not actually perform any computation, it just creates a computation graph. Even the variables are not initialized yet.
* To actually evaluate this graph, you need to open a TensorFlow session and use it to initialize the variables and evaluate `f`.
    * A TensorFlow session takes care of placing the operations onto devices such as CPUs and GPUs and running them, and it holds all the variable values.
* To create a session, initialize the variables, evaluate `f`, and close the session:
```
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()
```
* To create a more reproducible session (that closes automatically):
```
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
```
* Instead of manually running the initializer for every single variable you can use the `gloabal_variables_initializer()` function:
```
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
```
* An alternative to the above regular `Session` is to use an `InteractiveSession` when in a Jupyter notebook or python shell:
    * This makes it so you don't have to use the `with` statement
```
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()
```
* A TensorFlow program is typically split into two parts: the first part builds a computation graph (this is called the construction phase), and the second part runs it (this is the execution phase).
    * This construction phase typically builds a computation graph representing the ML model and the computations required to train it.
    * The execution phase generally runs a loop that evaluates a training step repeatedly (for example, one per mini-batch), gradually improving the model parameters.

**Managing Graphs**
* Any node you create is automatically added to the default graph:
```
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()
```
* If you instead want to manage multiple independent graphs:
```
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph
x2.graph is tf.get_default_graph()
```
* To reset the default graph:
```
tf.reset_default_graph()
```

**Lifecycle of a Node Value**
* When you evaluate a node, TensorFlow automatically determines the set of nodes that it depends on and it evaluates these nodes first.
* A variable starts its life when its initializer is run, and it ends when the session is closed.

**Linear Regression with TensorFlow**
* TensorFlow operations (aka ops) can take any number of inputs and produce any number of outputs.
    * For example, the addition and multiplication ops each take two inputs and produce one output.
* Constants add variables take no input (they are called source ops).
* The inputs and outputs are multidimensional arrays, called tensors (hence the name 'tensorflow').
    * Just like NumPy arrays, tensors have a type and a shape.
    * The Python API tensors are simply represented by NumPy ndarrays.
    * They typically contain floats, but you can also use them to carry strings.
* Linear Regression Example on California Housing Data:
```
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape

# Add a bias input feature to all training instances
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
```

**Implementing Gradient Descent**
```
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias)

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')

y_pred = tf.matmul(X, theta, name='prediction')
error = y_pred - 1

mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE = ', mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
```

**Using autodiff**
* TensorFlow's autodiff feature automatically and efficiently computes the gradient for you.
* Just replace the `gradients = ...` line in the previous example with:
```
gradients = tf.gradients(msw, [theta])[0]
```

**Using an Optimizer**
* To use an optimizer out of the box, replace the preceding `gradients = ...` and `training_op = ...` lines with:
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```

**Feeding Data to the Training Algorithm**
* To use placeholder nodes:
```
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_Dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
```

**Saving and Restoring Models**
* Once you have trained your model, you should save its parameters to disk so you can come back to it whenever you want it, use it in another program, and compare it to other models.
* TensorFlow makes saving and restoring a model very easy.
    * Just create a `Saver` node at the end of the construction phase (after all variable nodes are created).
    * Then in the execution phase, just call its `save()` method whenever you want to save the model.
* Example:
```
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')

init = tf.global_variables_initializer()
saver = ft.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        # checkpoint save every 100 epochs
        if epoch % 100 == 0:
            save_path = saver.save(sess, '/tmp/my_model.ckpt')

        sess.run(training_op)

    best_theta = theta.eval()
    save_path = saver.save(sess, 'tmp/my_model_final.ckpt')
```
* Restoring a model is easy, you create a `Saver` at the end of the construction phase just like before, but then at the beginning of the execution phase, instead of initializing the variables using the `init` node, you call the `restore()` method of the `Saver` object:
```
with tf.Session() as sess:
    saver.response(Sess, '/tmp/my_model_final.ckpt')
```
* By default a `Saver` saves and restores all variables under their own name, but if you need more control, you can specify which variables to save or restore, and what names to use.
    * For example, in the below we will save and restore only the theta variable under the name 'weights':
    ```
    saver = tf.train.Saver({'weights': theta})
    ```
* Be default the `save()` method also saves the structure of the graph in a second file with the same name plus a `.meta` extension.
* You can load this graph struture using `tf.train.import_meta_graph()`.
* This adds the graph to the default graph, and retuns a `Saver` instance that you can then use to restore the graph's state (i.e., the variable values):
```
saver = tf.train.import_meta_graph('/tmp/my_model_final.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, '/tmp/my_model_final.ckpt')
```
* This allows you to fully restore a saved model, including both the graph structure and the variable values, without having to search for the code that built it.

**Visualizing the Graph and Training Curves Using TensorBoard**
* TensorBoard displays interactive visualizations of training stats in your web browser (e.g., learning curves).
* You can also provide it the graph's definition and it will give you a great interface to browse through it.
    * This is very useful to identify errors in the graph, to find bottlenecks, and so on.
* To run TensorBoard:
```
$ tensorboard --logdir tf_logs/
```
* Then navigate to `http://0.0.0.0:6006`

**Name Scopes**
* When dealing with more complex models such as neural networks, the graph can easily become cluttered with thousands of nodes.
    * To avoid this, you can create name scopes to group related nodes.
* Example:
```
with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
```

---
#### Chapter 10 | Introduction to Artificial Neural Networks
**Introduction**
ANNs are at the very core of Deep Learning. They are versatile powerful, and scalable, making them ideal to tackle large and highly complex Machine Learning tasks.

**From Biological to Artificial Neurons**
* ANNs were first introduced back in 1943 as a way to present a simplified computational model of how biological neurons might work together in animal brains to perform complex computations using propositional logic.
* However, their popularity and use is only now increasing because:
    * There is now a huge quantity of data available to train neural networks, and ANNs frequently outperform other ML techniques on very large and complex problems.
    * The tremendous
