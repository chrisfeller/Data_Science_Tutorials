### Data School scikit-learn Tutorial
#### December 2017

---
#### Video 1: What is machine learning, and how does it work?
* Machine learning is the semi-automated extraction of knowledge from data.
* Two main categories of machine learning:
    1) Supervised Learning
        * AKA Predictive modeling
        * Process of making predictions using data
        * Example: Is a given email 'ham' or 'spam'?
        * In supervised learning, there is an outcome we are trying to predict.
    2) Unsupervised Learning
        * Process of extracting structure from data
        * Example: Segment grocery store shoppers into clusters that exhibit similar behaviors
        * There are no 'right answers' in unsupervised learning
* High-level steps of supervised learning:
    1) Train a machine learning model using *labeled data*
        * Labeled data has been labeled with the outcome
        * Machine learning model learns the relationship between the attributes of the data and its outcome.
    2) Make predictions on new data for which the label is unknown
    **The primary goal of supervised learning is to build a model that 'generalizes', which means it accurately predicts the future rather than the past.**
* [Examples of Supervised/Unsupervised/Reinforcement Learning](http://work.caltech.edu/library/014.html)
---
#### Video 2: Setting up Python for machine learning: sciki-learn and Jupyter Notebook
* Benefits of scikit-learn:
    * Provides a consistent interface to many machine learning models
    * Provides many tuning parameters but with sensible defaults
    * Exceptional documentation
    * Rich set of functionality for companion tasks (i.e. munging, scaling, etc.)
    * Active community for development and support
* Drawbacks of scikit-learn:
    * Harder (than R) to get started but easier to go deeper.
    * Less emphasis (than R) on model interpretability.
    * scikit-learn emphasizes accuracy over interpretability when compared with R.
* To render full static notebooks from websites: http://nbviewer.jupyter.org/

---
#### Video 3: Getting started with scikit-learn with the famous iris dataset
* To load the iris dataset from sklearn:
~~~
from sklearn.datasets import load_iris
iris = load_iris()
iris.data
~~~
* Machine learning terminology:
    * Each row is an *observation* (also known as sample, example, instance, record)
    * Each column is an *feature* (also known as predictor, attribute, independent variable, input, regressor)
    * Each value we are predicting is the response (also known as target, outcome, label, dependent variable)
* To print out the feature names of iris:
~~~
iris.feature_names
~~~
* To print out the target (in integer not names):
~~~
iris.target
~~~
* To print out the target (in names):
~~~
iris.target_names
~~~

**Classification vs. Regression:**
* Classification: Supervised learning in which the response is categorical
* Regression: Supervised learning in which the response is ordered and continuous

**Four Requirements for working with data in scikit-learn:**
1) Features and response are separate objects
    ~~~
    X = iris.data
    y = iris.target
    ~~~
2) Features and response should be numeric
3) Features and response should be NumPy arrays
4) Features and response should have specific shapes

**scikit-learn conventions:**
* Feature data should be stored in an object named `X`
    * Should be capitalized since it is a matrix
* Response data should be stored in an object named `y`
    * Should not be capitalized since it is an array

---
#### Video 4: Training a machine learning model with scikit-learn
* We will use the iris dataset to classify the flower type.
**k-Nearest Neighborts (KNN) Classification Steps:**
1) Pick a value for k (neighbors)
2) Search for the k observations in the training data that are 'nearest' to the measurements of the unknown iris
3) Use the most popular response value from the k-Nearest Neighbors as the predicted response value for the unknown iris.

**Load the Data:**
~~~
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target
~~~

**Step 1: Import the class you plan to use:**
~~~
from sklearn.neighbors import KNeighborsClassifier
~~~

**Step 2: Instantiate the estimator:**
* 'Estimator' is scikit-learn's term for model
* 'Instantiate' means 'make an instance of'
~~~
knn = KNeighborsClassifier(n_neighbors=5)
~~~
* Name of the object (i.e. knn) does not matter
* We can specify tuning parameters (aka hyperparameters) during this step
* All parameters not specified are set to their defaults

**Step 3: Fit the model with data (aka 'model training'):**
~~~
knn.fit(X, y)
~~~
* Model is learning the relationship between X and y
* This occurs in-place, which means we don't need to assign it to anything.

**Step 4: Predict the response for a new observation:**
~~~
knn.predict([[3,5,4,2]])
~~~
* Returns a numpy array
* Can predict for multiple observations at once

**Redo Process but for Logistic Regression:**
~~~
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,y)
log_reg.predict([[3,5,4,2]])
~~~

---
#### Video 5: Comparing machine learning models in scikit-learn
* An overview of model evaluation procedures
* First, load data:
~~~
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target
~~~

**Evaluation Procedure #1: Train and test on the entire dataset:**
1) Train the model on the entire dataset
2) Test the model on the same dataset, and evaluate how well we did by comparing the predicted response values with the true response values.
* Using Logistic Regression:
~~~
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,y)
y_pred = logreg.predict(X)
~~~
* To evaluate the model, we will use accuracy, which is the proportion of correct predictions.
    * Accuracy is a common evaluation metric for classification problems.
~~~
from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred)
~~~
* This output is known as the *training accuracy* as we tested the model on the same data we used to train the model.
* Problems with training and testing on the same data:
    * Goal is to estimate likely performance of a model on out-of-sample data
    * But, maximizing training accuracy rewards overly complex models that won't necessarily generalize
    * Unnecessarily complex models, like this one, overfit the training data

**Evaluation Procedure #2: Train/Test Split:**
* Split the dataset into two pieces: a training set and a testing set
* Train the model on the training set
* Test the model on the testing set, and evaluate how well we did.
* To train/test split your data:
~~~
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
~~~
* Model can now be trained and tested on different data
* Response values are known for the training set, and thus predictions can be evaluated
* Testing accuracy is a better estimate than training accuracy of out-of-sample performance
* The typical testing holdout is 20-40% of the entire dataset.
* Model Logistic Regression using train/test split procedure:
~~~
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy_score(y_test, y_pred)
~~~
* Training accuracy rises as model complexity increases.
* Testing accuracy  penalizes models that are too complex or not complex enough.
* For kNN models, complexity is determined by the value of K (lower value = more complex)
* Downsides with train/test split:
    * Provides a high-variance estimate of out-of-sample accuracy
        * K-fold cross-validation overcomes this limitation
    * But, train/test split is still useful because of its flexibility and speed.

---
#### Video 6: Data science in python: pandas, seaborn, sciki-learn
**Types of Supervised Learning:**
* Classification: Predict a categorical response
* Regression: Predict a continuous response

**Pandas:**
* Popular python library for data exploration, manipulation, and analysis
* Import convention:
~~~
import pandas as pd
~~~
* Reading in data with pandas:
~~~
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
~~~
* Primary object types in pandas:
    1) DataFrame: rows and columns (like a spreadsheet)
    2) Series: a single column

**Seaborn:**
* Python library for statistical data visualization build on top of matplotlib
* Import convention:
~~~
import seaborn as sns
~~~
* Visualize the relationship between each of our features and our response variable using scatterplots:
~~~
sns.pairplot(data,x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7)
~~~
* To visualize those same relationships with a line of best fit and 95% confidence band:
~~~
sns.pairplot(data,x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')
~~~
* To use the fivethirtyeight plot style input the following before you create any graphs:
~~~
plt.style.use('fivethirtyeight')
~~~

**Linear Regression:**
* Pros: fast, no tuning required, highly interpretable, well-understood
* Cons: unlikely to produce the best predictive accuracy (presumes a linear relationship between the features and response)
* Linear relationship 'learns' the model coefficients (betas) during the model fitting step using the 'least squares' criterion. Then, the fitted model can be used to make predictions.

**Preparing X and y using pandas:**
* X is called the 'feature matrix'
* y is called the 'response vector'
* scikit-learn expects X and y to be numpy arrays.
    * However, pandas is built on top of numpy so DataFrames and Series are actually numpy arrays.
    * So X can be a pandas DataFrame and y can be a Series and scikit-learn will understand them.
~~~
X = data[['TV', 'radio', 'newspaper']]
y = data['sales']
~~~

**Splitting X and y into training and testing sets:**
~~~
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
~~~
* This defaults to using 25% of the data for testing and 75% for training.

**Fit a Linear Regression Model:**
~~~
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
~~~

**Display and Interpret Model Coefficients:**
~~~
linreg.intercept_
linreg.coef_
~~~
* The trailing underscore is sklearn's convention for anything that is estimated by a model.
* For display the coefficients with their corresponding feature names:
~~~
list(zip(X_train.columns, linreg.coef_))
~~~

**Linear Regression Interpretation:**
* Coefficients and intercepts are statements of association, not causation.

**Make Predictions:**
~~~
y_pred = linreg.predict(X_test)
~~~

**Evaluate Model:**
* Unlike classification, accuracy is not a useful metric to evaluate regression models.
* Instead we use:
    1) Mean Absolute Error (MAE)
        * The mean of the absolute value of the errors.
        * Can think of this as the average error of the predictions.
        * Easiest to understand, because it's the average error.
            ~~~
            from sklearn.metrics import mean_absolute_error(y_test, y_pred)
            ~~~
    2) Mean Squared Error (MSE)
        * The mean of the squared errors.
        * More popular than MAE, because MSE 'punishes' larger errors.
            ~~~
            from sklearn.metrics import
            mean_squared_error(y_test, y_pred)
            ~~~
    3) Root Mean Squared Error (RMSE)
        * The square root of the mean of the squared errors.
        * Even more popular than MSE, because RMSE is interpretable in the 'y' units.
            ~~~
            from sklearn.metrics import np.sqrt(mean_squared_error(y_test, y_pred))
            ~~~

---
#### Video 7: Selecting the best model in scikit-learn using cross-validation
**Review of model evaluation procedures:**
* Motivation: We need a way to choose between machine learning models
    * Goal is to estimate likely performance of a model on out-of-sample data
* Initial Idea: Train and test on the same data
    * But, maximizing training accuracy rewards overly complex models which overfit the training data
* Alternative idea: Train/test split
    * Split the dataset into two pieces, so that the model can be trained and tested on different data
    * Testing accuracy is a better estimate than training accuracy of our-of-sample performance
    * but, it provides a high variance estimate since changing which observations happen to be in the testing set can significantly change testing accuracy.
* Testing accuracy is called a 'high-variance' estimate, meaning it changes based on which data we test on.
    * Cross-validation is the solution to this.

**Steps for K-fold cross-validation:**
1) Split the dataset into K equal partitions (or 'folds')
2) Use fold 1 as the testing set and the union of the other folds as the training set.
3) Calculate testing accuracy.
4) Repeat steps 2 and 3 K times, using a different fold as the testing set each time.
5) Use the average testing accuracy as the estimate of the out-of-sample accuracy.
*We are dividing the observations into folds NOT the features*

![5-Fold Cross Validation Diagram ](http://blog-test.goldenhelix.com/wp-content/uploads/2015/04/B-fig-1.jpg)

**Visualizing 5-Fold Cross Validation in Python:**
* Simulate splitting a dataset of 25 observations into 5 Folds:**
~~~
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=5, shuffle=False)
~~~
* Print the contents of each training and testing set:
~~~
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{} {} {}'.format(iteration, data[0], data[1]))
~~~
* Dataset contains 25 observations (numbered 0 through 24)
* 5-fold cross-validation, thus it runs for 5 iterations
* For each iteration, every observation is either in the training set or the testing set, but not both
* Every observation is in the testing set exactly once.

**Comparing Cross-Validation to Train/Test Split:**
* Advantages of cross-validation:
    * More accurate estimate of out-of-sample accuracy
    * More 'efficient' use of data (every observation is used for both training and testing)
* Advantages of train/test split:
    * Runs K times faster than K-fold cross-validation
    * Simpler to examine the detailed results of the testing process

**Cross-validation Recommendations:**
1) K can be any number, but K=10 is generally recommended
2) For classification problems, stratified sampling is recommended for creating the folds
    * Each response class should be represented with equal proportion in each of the K folds
    * scikit-learn's `cross_val_score` function does this by default

**10-Fold Cross Validation Example:**
* Imports:
~~~
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
~~~
* Load data:
~~~
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target
~~~
* Run 10-fold cross-validation with K=5 for KNN
~~~
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
~~~
* Return the average of the 10-cross val scores:
~~~
scores.mean()
~~~

**Cross-validation example: parameter tuning:**
* Goal: Select the best tuning parameters (aka 'hyperparameters' for KNN on the iris dataset)
* Imports:
~~~
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
~~~
* Load data:
~~~
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target
~~~
* Search for optimal value of k for KNN:
~~~
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
~~~
* Plot the value of K for KNN (x-axis) versus cross-validation accuracy (y_axis)
~~~
import matplotlib.pyplot as plt
plt.plot(k_range, k_scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-Validation Accuracy')
plt.show()
~~~
* From the graph, the optimal k for KNN appears to be 13.
* This is an example of the bias-variance tradeoff, low values of k produce a model with low bias and high variance, while high values of k produce a model with high bias and low variance. The best model is found in the middle because it balances bias and variance and is most likely to generalize to out-of-sample data.

**Cross-validation example: model  selection:**
* Goal: Compare the best KNN model with logistic regression on the iris dataset.
* Imports:
~~~
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
~~~
* Load data:
~~~
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target
~~~
* 10-fold cross-validation with best KNN model (from above):
~~~
knn = KNeighborsClassifier(n_neighbors=13)
cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()
~~~
* 10-fold cross validation with logistic regression:
~~~
logreg = LogisticRegression()
cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean()
~~~
* In this case, we would choose the KNN model with K=13 over the logistic regression model as it's 10-fold cross-validation mean accuracy is higher.

**Cross-validation example: feature selection:**
* Goal: Select whether the newspaper feature should be included in the linear regression model on the advertising dataset.
* Imports:
~~~
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
~~~
* Load data:
~~~
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
~~~
* Create feature matrix `X` and target vector `y`:
~~~
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales
~~~
* 10-fold cross-validation with all three features:
~~~
lm = LinearRegression()
scores = -cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
~~~
* Calculate RMSE for the above:
~~~
np.sqrt(scores).mean()
~~~
* Now, do the same for 10-fold cross validation for just 'TV' and 'Radio' features:
~~~
feature_cols = ['TV', 'radio']
X = data[feature_cols]
print(np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')).mean())
~~~
* From this, we would choose the model which excludes newspaper since it has a lower RMSE.

**Improvements to Cross-Validation:**
* Repeated cross-validation:
    * Repeat cross-validation multiple times (with different random splits of the data) and average the results
    * More reliable estimate of out-of-sample performance by reducing the variance associated with a single trial of cross-validation
* Create a hold-out set:
    * 'Hold out' a portion of the data before the model building process
    * Locate the best model using cross-validation on the remaining data, and test it using the hold-out set
    * More reliable estimate of out-of-sample performance since hold-out is truly out-of-sample
* Feature engineering and selecting within cross-validation iterations
    * Normally, feature engineering and selection occurs before cross-validation.
    * Instead, perform all feature engineering and selection within each cross-validation iteration.
    * More reliable estimate of out-of-sample performance since it better mimics the application of the model to out-of-sample data

---
#### Video 8: How to find the best model parameters in scikit-learn
**Review of K-fold Cross-Validation:**
* Steps for cross-validation:
    1) Dataset is split into K 'folds' (often 10) of equal size
    2) Each folds acts as the testing set 1 time, and acts as the training set K-1 times
    3) Average testing performance is used as the estimate of out-of-sample performance.
* Benefits of cross-validation:
    * More reliable estimate of out-of-sample performance than train/test split
    * Can be used for selecting tuning parameters, choosing between models, and selecting features
* Drawbacks of cross-validation
    * Can be computationally expensive.

**Review of parameter tuning using `cross_val_score`:**
* Imports:
~~~
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
~~~
* Load data:
~~~
iris = load_iris()
X = iris.data
y = iris.target
~~~
* 10-fold cross-validation with K=5 for KNN (the n_neighbors parameters)
~~~
knn =KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()
~~~
* Search for an optimal value for K:
~~~
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(score.mean())
~~~
* Plot the value of k for KNN (x-axis) versus the cross-validation accuracy (y-axis):
~~~
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validation Accuracy')
plt.show()
~~~
* This same process can be simplified/automated with `GridSearchCV`

**More Efficient Parameter tuning using `GridSearchCV`:**
* Allows you to define a grid of parameters that will be searched using K-fold cross-validation
* Go through the same process as above but with `GridSearchCV`:
* Imports:
~~~
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
~~~
* Load data:
~~~
iris = load_iris()
X = iris.data
y = iris.target
~~~
* Define the parameter values that should be searched:
~~~
k_range = list(range(1, 31))
~~~
* Create a parameter grid: map the parameter names to the values that should be searched:
~~~
param_grid = dict(n_neighbors=k_range)
~~~
* Instantiate the grid:
~~~
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
~~~
*You can set n_jobs=-1 in the above to run computations in parallel (if supported by your computer and OS)*
* Fit the grid with data:
~~~
grid.fit(X, y)
~~~
* View the complete results:
~~~
grid.grid_scores_
~~~
* Examine the first tuple:
~~~
grid.grid_scores_[0].parameters
grid.grid_scores_[0].cv_validation_scores
grid.grid_scores_[0].mean_validation_score
~~~
* Create a list of the mean scores:
~~~
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
~~~
* Plot the results:
~~~
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validation Accuracy')
plt.show()
~~~
* Examine the best model:
~~~
grid.best_score_
grid.best_params_
grid.best_estimator_
~~~

**Searching multiple parameters simultaneously:**
* Example: Tuning `max_depth` and `min_samples_leaf` for a DecisionTreeClassifier
* Could tune parameters independently: change `max_depth` while learning `min_samples_leaf` at its default value, and vice versa
* But, best performance might be achieved when neither parameter is at its default value.
* To tune multiple parameters in GridSearchCV
* Imports:
~~~
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
~~~
* Load data:
~~~
iris = load_iris()
X = iris.data
y = iris.target
~~~
* Define the parameter values that should be searched:
~~~
k_range = list(range(1,31))
weight_options = ['uniform', 'distance']
~~~
* Create a parameter grid: map the parameter names to the values that should be searched:
~~~
param_grid = dict(n_neighbors=k_range, weights = weight_options)
~~~
* Instantiate and fit the grid:
~~~
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)
~~~
* View the complete results:
~~~
grid.grid_scores_
~~~
* To examine the best model:
~~~
grid.best_score_
grid.best_params_
~~~
**GridSearchCV has a shortcut that allows you to predict on the model with the best parameters:**
~~~
grid.predict(X_test)
~~~

**Reducing computational expense using `RandomizedSearchCV`:**
* Searching many different parameters at once may be computationally infeasible
* `RandomizedSearchCV` searches a subset of the parameters, and you control the computational 'budget'
* Imports:
~~~
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV
~~~
* Load data:
~~~
iris = load_iris()
X = iris.data
y = iris.target
~~~
* Define the parameter values that should be searched:
~~~
k_range = list(range(1,31))
weight_options = ['uniform', 'distance']
~~~
* Specify 'parameter distributions' rather than a 'parameter grid':
~~~
param_dist = dict(n_neighbors=k_range, weights=weight_options)
~~~
*Important: Specify a continuous distribution (rather than a list of values) for any continuous parameters*
* Instantiate and fit the grid:
~~~
knn = KNeighborsClassifier()
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
rand.fit(X,y)
rand.grid_scores_
~~~
* Examine the best model:
~~~
rand.best_score_
rand.best_params_
~~~

---
#### Video  9: How to evaluate a classifier in scikit-learn
**Review of model evaluation:**
* Goal of model evaluation: We need a way to choose between different model types, tuning parameters, and features
* Use a model evaluation procedure to estimate how well a model will generalize to out-of-sample data
* Requires a model evaluation metric to quantify the model performance

**Model evaluation procedures:**
1) Training and testing on the same data
    * Rewards overly complex models that 'overfit' the training data and won't necessarily generalize
2) Train/test split
    * Split the dataset into two pieces, so that the model can be trained and tested on different data
    * Better estimate of out-of-sample performance, but still a 'high variance' estimate
    * Useful due to its speed, simplicity, and flexibility
3) K-fold Cross-Validation
    * Systematically create 'K' train/test splits and average the results together
    * Even better estimate of out-of-sample performance
    * Runs 'K' times slower than train/test split

**Model Evaluation Metrics:**
* Regression problems: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error
* Classification problems: Classification accuracy

**Classification Accuracy:**
* Review of classification accuracy from Prima Indian Diabetes dataset
* Imports:
~~~
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
~~~

* Load data:
~~~
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)
~~~
* Question: Can we predict the diabetes status of a patient given their health measurements?
* Define the feature matrix `X` and target vector `y`:
~~~
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
X = pima[feature_cols]
y = pima.label
~~~
* Train/Test Split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
~~~
* Train a logistic regression model on the training set:
~~~
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
~~~
* Predict on the model with the test data:
~~~
y_pred_class = logreg.predict(X_test)
~~~
* Calculate classification accuracy:
~~~
accuracy_score(y_test, y_pred_class)
~~~

**Null Accuracy:**
* Classification Accuracy: Percentage of correct predictions
* Null accuracy: Accuracy that could be achieved by always predicting the most frequent class
* To calculate the null accuracy of the above example, first examine the class distribution of the testing set (using a Pandas Series method):
~~~
y_test.value_counts()
~~~
* We observe that there are 130 0's and 62 1's in the target vector.
* Calculate the percentage of ones and zeroes
~~~
y_test.mean() #ones
1 - y_test.mean() #zeroes
~~~
* Calculate the null accuracy (for binary classification problems coded 0/1):
~~~
max(y_test.mean(), 1 - y_test.mean())
~~~
* Calculate the null accuracy (for multi-classification problems):
~~~
y_test.value_counts().head(1)/len(y_test)
~~~
* In this case, we are better off predicting the most frequent class (0).

**Conclusion:**
* Classification accuracy is the easiest classification metric to understand
* But, it does not tell you the underlying distribution of response values
* And, it does not tell you what 'types' of errors your classifier is making.

**Confusion Matrix:**
* Definition: Table that describes the performance of a classification model.
* Import:
~~~
from sklearn.metrics import confusion_matrix
~~~
* To display the confusion matrix:
~~~
confusion_matrix(y_test, y_pred_class)
~~~
* Every observation in the testing set is represented in exactly one box of the confusion matrix.
* The confusion matrix is a 2x2 matrix because there are 2 response classes

**Confusion Matrix Terminology:**
* True Positives (TP): We correctly predicted that they do have diabetes
* True Negative (TN): We correctly predicted that they don't have diabetes
* False Positive (FP): We incorrectly predicted that they do have diabetes (a 'Type I Error')
* False Negatives (FN): We incorrectly predicted that they don't have diabetes (a 'Type II Error')

**Metrics Computed From a Confusion Matrix:**
* To save a confusion matrix for later use:
~~~
confusion = confusion_matrix(y_test, y_pred_class)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
~~~
* Classification Accuracy: Overall, how often is the classifier correct?
~~~
(TP + TN)/(TP + TN + FP + FN)
 # OR
accuracy_score(y_test, y_pred_class)
~~~
* Classification Error: Overall, how often is the classifier incorrect?
~~~
(FP + FN)/(TP + TN + FP + FN)
 # OR
1 - accuracy_score(y_test, y_pred_class)
~~~
* Sensitivity: When the actual value is positive, how often is the prediction correct?
    * How 'sensitive' is the classifier to detecting positive instances?
    * Also known as 'True Positive Rate' or 'Recall'
~~~
TP / (TP + FN)
 # OR
from sklearn.metrics import recall_score
recall_score(y_test, y_pred_class)
~~~
* Specificity: When the actual value is negative, how often is the prediciton correct?
    * How 'specific' (or 'selective') is the classifier in predicting positive instances?
~~~
TN / (TN + FP)
~~~
* False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
~~~
FP / (TN + FP)
~~~
* Precision: When a positive value is predicted, how often is the prediction incorrect?
    * How 'precise' the classifier is when predicting positive instances.
~~~
TP / (TP + FP)
 # OR
from sklearn.metrics import precision_score
precision_score(y_test, y_pred_class)
~~~

**Conclusion:**
* Confusion matrix gives you a more complete picture of how your classifier is performing
* Also allows you to compute various classification metrics, and these metrics can guide your model selection
* Sensitivity and specificity have an inverse relationship.

**Which metrics should you focus on?**
* Choice of metric depends on your business objective.
* Spam filter (positive class is 'spam'): Optimize for precision or specificity because false negatives (spam goes to the inbox) are more acceptable than false positives (non-spam is caught by the spam filter).
* Fraudulent transaction detector (positive class is 'fraud'): Optimize for sensitivity because false positives (normal transactions that are flagged as possible fraud) are more acceptable than false negatives (fraudulent transactions that are not detected).

**Adjusting the classification threshold:**
* Print the first 10 predicted responses:
~~~
logreg.predict(X_test)[0:10]
~~~
* Print the first 10 predicted probabilities of class membership:
~~~
logreg.predict_proba(X_test)[0:10, :]
~~~
* Print the first 10 predicted probabilities for class 1:
~~~
logreg.predict_proba(X_test)[0:10, 1]
~~~
* Store the predicted probabilities for class 1
~~~
y_pred_prob = logreg.predict_proba(X_test)[:,1]
~~~
* Predict diabetes if the predicted probability is greater than 0.3
~~~
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob], 0.3)[0]
~~~


**ROV Curves and Area Under the Curve (AUC):**
* To see how sensitivity and specificity are affected by various thresholds, without actually changing the threshold, plot the ROC Curve.
* To plot ROC Curve:
~~~
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1,0])
plt.title('ROC Curve for Diabetes Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
~~~
* ROC curve can help you choose a threshold that balances sensitivity and specificity in a way that makes sense for your particular context.
* You can't actually see the thresholds used to generate the curve on the ROC curve itself.
* To define a function that accepts a threshold and prints sensitivity and specificity:
~~~
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
~~~
* To calculate AUC:
~~~
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_prob)
~~~
* AUC is useful as a single summary of classifier performance.
* If you randomly chose one positive and one negative observation, AUC represents the likelihood that your classifier will assign a higher predicted probability to the positive observation.
* AUC is useful even when there is high class imbalance (unlike classification accuracy)

**Summary:**
* Confusion matrix advantages:
    * Allows you to calculate a variety of metrics
    * Useful for multi-class problems (more than two response classes)
* ROC/AUC advantages:
    * Does not require you to set a classification threshold
    * Still useful when there is high class imbalance

---
#### Video  10: Tutorial Machine Learning in scikit-learn
**Part 1: Model building in scikit-learn:**
* Load iris dataset:
~~~
from sklearn.datasets import load_iris
iris = load_iris()
~~~
* Store the feature matrix `X` and response vector `y`:
~~~
X = iris.data
y = iris.target
~~~
* 'Features' are also known as predictors, inputs, or attributes.
* 'Response' is also known as the target, label, or output.
* 'Observations' are also known as samples, instances, or records.
* In order to build a model, the features must be numeric and every observation must have the same features in the same order.
* The four steps of modeling in scikit-learn are:
    1) Import
    2) Instantiate
    3) Fit
    4) Predict
* Do this process with k-nearest neighbord model on the iris data:
~~~
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X, y)
knn.predict(X)
~~~
* In order to make a prediction, the new observation must have the same features as the training observations, both in number and meaning.

**Part 2: Representing text as numbers:**
* Example text:
~~~
simple_train = ['call you tonight', 'Call me a cab', 'please call me...Please']
~~~
* Text analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithm themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.
* We will use `CountVectorizer` to convert text into a matrix of token counts:
~~~
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(simple_train)
~~~
* To examine the fitted vocabulary:
~~~
vect.get_feature_names()
~~~
* To transform training data into a 'document-term matrix':
~~~
simple_train_dtm = vect.transform(simple_train)
~~~
* Convert sparse matrix to dense matrix:
~~~
simple_train_dtm.toarray()
~~~
* To examine the vocabulary and document-term matrix together:
~~~
import pandas as pd
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
~~~
* We call vectorization the general process of turn a collection of documents into numeric feature vectors. This specific strategy (tokenization, counting, and normalization) is called Bag of Words or Bag of n-grams representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.
* Example text for model testing:
~~~
simple_test = ["please don't call me"]
~~~
* In order to make a prediction, the new observation must have the same features as the training observations, both in number and meaning.
* To transform testing data into a document-term matrix:
~~~
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.to_array()
~~~
* Summary:
    * `vect.fit(train)` learns the vocabulary of the training data
    * `vect.transform(train)` uses the fitted vocabulary to build a document-term matrix from the training data
    * `vector.transform(test)` uses the fitted vocabulary to build a document-term matrix from the testing data (and ignores tokens it hasn't seen before)

**Part 3: Reading a text-based dataset into pandas:**
* Read in file:
~~~
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(url, header=None, names=['label', 'message'])
~~~
* To view the class distribution:
~~~
sms['label'].value_counts()
~~~
* To convert label to a numerical variable:
~~~
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
~~~
* To define feature matrix `X` and target vector `y`:
~~~
X = sms['message']
y = sms['label_num']
~~~
* To split X and y into train/test split:
~~~
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
~~~
*It is important that you train/test split prior to vectorizing*

**Part 4: Vectorizing our dataset:**
* To instantiate the vectorizer:
~~~
vect = CountVectorizer()
~~~
* To learn training data vocabulary:
~~~
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)

 # OR the same can be done via:
X_train_dtm = vect.fit_transform(X_train)
~~~
* To examine the document-term matrix:
~~~
X_train_dtm
~~~
* To transform testing data (using fitted vocabulary) into a document-term:
~~~
X_test_dtm = vect.transform(X_test)
~~~
*We don't ever fit our test data, only transform*

**Part 5: Building and evaluating a model:**
* We will use multinomial Naive Bayes classifier, which is suitable for classification with discrete features (e.g., word counts for text classification).
* To import and instantiate the model:
~~~
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
~~~
* To train the model using `X_train_dtm` because we needed a transformed version of `X_train`:
~~~
nb.fit(X_train_dtm, y_train)
~~~
* To make class predictions for `X_test_dtm`:
~~~
y_pred_class = nb.predict(X_test_dtm)
~~~
* To calculate accuracy:
~~~
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_class)
~~~
* To show confusion matrix:
~~~
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_class)
~~~
* To calculate predicted probabilities for `X_test_dtm`:
~~~
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
~~~
* To calculate AUC:
~~~
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_prob)
~~~

**Part 6: Comparing models:**
* We will compare multinomial Naive Bayes with logistic regression.
* Go through same process with logistic regression:
~~~
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
y_pred_proba = logreg.predict_proba(X_test_dtm)[:,1]
accuracy_score(y_test, y_pred_class)
roc_auc_score(y_test, y_pred_prob)
~~~

**Part 7: Examining a model for further insight:**
* We will examine our trained Naive Bayes model to compare 'spamminess' of each token.
* To store the vocabulary of `X_train`:
~~~
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)
~~~
* Naive Bayes conounts the number of times each token appears in each class:
~~~
nb.feature_count_
~~~
* Number of times each token appears across all HAM messages:
~~~
ham_token_count = nb.feature_count_[0,:]
~~~
* Number of times each token appears across all SPAM messages:
~~~
spam_token_count = nb.feature_count_[1,:]
~~~
* Create a DataFrame of tokens with their separate ham and spam counts:
~~~
tokens = pd.DataFrame({'token': X_train_tokens, 'ham': ham_token_count, 'spam': spam_token_count})
~~~

**Part 9: Tuning the vectorizer:**
* The vectorizer is worth tuning, just like a model is worth tuning.
* A few parameters that you might want to tune:
    * `stop_words`: words that don't contain a lot of meaning. sklearn has a built-in 'english' `stop_words` argument that includes words like 'a', 'and', 'the'.
    * `ngram_range`: combinations of n-length word combinations
    * `max_df`: ignore terms that occur above a given frequency within a document.
    * `min_df`: ignore terms that occur below a given frequency within a document.
