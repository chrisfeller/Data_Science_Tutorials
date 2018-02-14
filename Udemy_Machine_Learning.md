### Udemy Machine Learning
####

---
#### Introduction (Section 1)
**Curriculum:**
* Regression: Linear Regression, Decision Trees, and Random Forest Regression
* Classification: Logistic Regression, K-Nearest Neighbors (kNN), Support Vector Machines (SVM), and Naive Bayes
* Clustering: K-Means and Hierarchical
* Association Rule Learning
* Dimensionality Reduction
* Neural Networks

---
#### Download and Install Python Anaconda (Section 2)
**Python:**
* Python is a general purpose programming language.
* Packages used in course: Numpy, Scipy, Matplotlib

**Anaconda:**
* Anaconda is an open source Python distribution.
* Provides a python interpreter, together with popular python packages.

**Conda:**
* Anaconda's package management system.
* Similar to pip, but for all packages within Anaconda distribution
* To install a new package:
~~~
$ conda install package_name
~~~
* To see all of the packages installed via Anaconda:
~~~
$ conda list
~~~
* To search to see if a specific package is installed:
~~~
$ conda search package_name
~~~
* To update a package:
~~~
$ conda update package_name
~~~

---
#### 'Hello World' in Jupyter Notebook (Section 3)
* Jupyter Notebook is a JSON document containing an ordered list of input/output cells which can contain code, text, mathematics, plots, and rich media.
* Useful for reproducible research.
* You can download Jupyter Notebooks as .html, .py, .md, or even .pdf files.

---
#### Mac Anaconda & IPython Installation (Section 4)
* To shut down a Jupyter Notebook:
~~~
CTRL + C
~~~

---
#### Datasets, Python Notebooks and Scripts For the Course (Section 5)
* The course material can be cloned from Github: https://github.com/Jojo666/PythonML_is_fun

---
#### Regression (Section 6)
* Regression allows you to make predictions from data by learning the relationship between features of your data and some observed, continuous-valued response.
* Three main regression techniques:
    1) Linear Regression
    2) Decision Trees (CART)
    3) Random Forest Regression

---
#### Linear Regression - Theory (Section 7)
* Dependent Variable: Values you want to explain or forecast.
    * Denoted as `y`
* Independent Variables (AKA Explanatory): Values that explain the dependent variable.  
    * Denoted as `X`
* Linear Regression: Line of best fit
    * `Y = MX + B`

---
#### Linear Regression - Practical Labs (Section 8)
* Load Libraries:
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from sklearn import linear_model
~~~
* Load Data into pandas DataFrame:
~~~
iris = sns.load_dataset('iris')
iris.head()
~~~
* Predict `petal_width` from `petal_length`. First, create y (target variable) and X (explanatory variable).
~~~
X = iris[['petal_length']]
y = iris['petal_width']
~~~
* Model in statsmodel:
~~~
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
~~~
* To get an intercept (and thus not force the regression line through 0), you must add a constant:
~~~
X = iris[['petal_length']]
X = sm.add_constant(X, prepend=False)
y = iris['petal_width']

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
~~~
* Now, run a multiple linear regression using `petal_length` and `sepal_length` as explanatory variables:
~~~
X = iris[['petal_length', 'sepal_length']]
X = sm.add_constant(X, prepend=False)
y = iris['petal_width']

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
~~~
* To run a multiple linear regression using a categorical variable (`species`), first dummy `species` and add those dummies back to the original `iris` dataframe:
~~~
dummies = pd.get_dummies(iris['species'])
iris = pd.concat([iris, dummies], axis=1)

X = iris[['petal_length', 'sepal_length', 'setosa', 'versicolor', 'virginica']]
X = sm.add_constant(X)
y = iris['petal_width']

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
~~~
* Linear model using sklearn:
~~~
model = linear_model.LinearRegression()
results = model.fit(X, y)
print(results.intercept_, results.coef_)
~~~
* sklearn is not the best choice for linear regression, instead use statsmodel.

---
#### Decision Tree - Theory (Section 9)
* Decision trees can solve both classification and regression problems.
    * AKA `Cart` - Classification and regression tree

**Advantages:**
* Simple to understand, interpret, and visualize
* Implicitly performs variable screening or feature selection
* Can handle both numerical and categorical data
* Little effort for data preparation

**Disadvantages:**
* Commonly leads to overfitting
* Can be unstable because small variations in the data (high variance models)
* Decision Trees are a greedy algorithm, which do not guarantee to return the globally optimal decision tree.
    * Greedy algorithm: Always makes the choice that seems to be the best at that moment. This means that it makes a locally-optimal choice in the hope that his choice will lead to a globally-optimal solution.

**Difference Between Classification and Regression Trees:**
* Regression trees are used when dependent variable is continuous.
    * Predications are made using the mean/average of the values obtained in terminal nodes
* Classification trees are used when dependent variable is categorical.
    * Predictions are made using the mode of the values obtained in terminal nodes

**Common Decision Tree Algorithms:**
1) Gini Index
2) Chi-Square
3) Information Gain
4) Reduction in Variance

---
#### Decision Tree - Practical Lab (Section 10)
**Decision Tree on Boston Housing Data:**
* Imports:
~~~
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
~~~
* Import data
~~~
data = datasets.load_boston()
~~~
* Define X and y:
~~~
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['MEDV'])
~~~
* Train/Test Split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
~~~
    * The `test_size` argument specifies what percentage of the data should be held for the test set.
* Instantiate a decision tree model with a depth of 2 to avoid overfitting:
~~~
dt = DecisionTreeRegressor(max_depth=2)
~~~
* Fit the model with your training data:
~~~
dt.fit(X_train, y_train)
~~~
* Predict test data:
~~~
y_pred = dt.predict(X_test)
~~~
* To calculate the mean squared error of the predictions:
~~~
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
~~~

**Cross Validation:**
* Instantiate decision tree model:
~~~
regression_tree = DecisionTreeRegressor(min_samples_split=30, min_samples_leaf=10)
~~~
* Set up cross validation:
~~~
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
crossvalidation = KFold(n=X.shape[0], n_folds=10, shuffle=True, random_state=1)
~~~
* Fit model:
~~~
regression_tree.fit(X, y)
~~~
* Predict and Score model:
~~~
score = np.mean(cross_val_score(regression_tree, X, y, scoring='mean_squared_error', cv=crossvalidation, n_jobs=1))
~~~

**How to Decide on the Max Depth:**
* To iterate through max depths between 1-9:
~~~
for depth in range(1,10):
    regression_tree = DecisionTreeRegressor(min_samples_split=30, min_samples_leaf=10)
    if regression_tree.fit(X,y).tree_.max_depth < depth:
        break
        score = np.mean(cross_val_score(tree_classifier, X, y, scoring='accuracy', cv=crossvalidation, n_jobs=1))
    print('Depth: {0} Accuracy: {1:0.03f}'.format(depth, score))
~~~

---
#### Random Forest - Theory (Section 11)
* In comparison with decision tree models, random forests grow multiple decision trees.
* Individual observations are classified by each tree and the forest chooses the final classification for the observation by selecting the category for which the object was most classified to.
    * In the case of regression, it takes the average of each tree.
* Advantages of Random Forests:
    * Can handle both classification and regression tasks
    * Handles missing data well and maintains accuracy for missing data.
    * Won't overfit the model, inn contrast to individual decision trees.
    * Handles large data sets with higher dimensionality well.
* Disadvantages:
    * Works well for classification but not as well for regression
    * You have very little control on what the model does.
    * May overfit noisy datasets.
* Random Forest Pseudocode:
    1) Assume number of cases in the training set is N. Then, a sample of these N cases is taken at random but with replacement.
    2) If there are M input variables of features, a number m<M is specified such that at each node, m variables are selected at random out of the M.  The best split of these m is used to split the node. The value of m is held constant while we grow the forest.
    3) Each tree is grown to the largest extent possible and there is no pruning.
    4) Predict new data by aggregating the predications of the n trees
        * majority vote for classification, average for regression.
* Random Forests are known as an ensemble machine learning algorithm, because they divide and conquer.
* Terms and Definitions:
    * Bagging (AKA Bootstrap Aggregating): A machine learning ensemble meta-algorithm designed to improve stability and accuracy of the model
        * Reduces variance
        * Helps to avoid overfitting
    * Boosting: A machine learning ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning.

---
#### Random Forest - Practical Lab (Section 12)
**Random Forest Regression on Boston Housing Data:**
* Imports:
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
~~~
* Load Data:
~~~
data = datasets.load_boston()
~~~
* Assign feature matrix and labels:
~~~
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['MEDV'])
~~~
* Test/Train Split
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
~~~
* Fit and train model
~~~
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2)
rf.fit(X_train, y_train)
~~~
* To look at feature importance:
~~~
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print('{0} feature {1} ({2:0.03f})'.format((1+f), indices[f], importances[indices[f]]))
~~~
* Make predictions:
~~~
y_pred = rf.predict(X_test)
~~~
* Evaluate model using MSE:
~~~
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
~~~
* Evaluate model using R-Squared:
~~~
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
~~~

---
#### Classification (Section 13)
* Classification is the problem of identifying to which of a set of categories (sub-populations) a new observations belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known.
* Classification is an example of pattern recognition
* Types of classification techniques:
    * Logistic Regression
    * K-nearest neighbors
    * Support Vector Machine
    * Naive Bayes
* Example: 'ham' or 'spam' emails

---
#### Logistic Regression - Theory (Section 14)
**Background of Logistic Regression:**
* Borrowed from statistics
* For classification, not regression
* Similar to linear regression, except targets are limited to [0, 1]
* Can be used to fit complex, non-linear datasets

**Estimating Coefficients of the Logistic Function:**
* Use gradient descent or stochastic gradient descent
    * Given each training instance:
        1) Calculate a prediction using the current values of the coefficients.
        2) Calculate new coefficient values based on the error in the prediction

**Online Machine Learning:**
* Batch learning:
    * Scan all data before building a model
    * Data must be stored in memory or storage
* Online learning:
    * Model will be updated by each data sample
    * Sometimes with theory that the online model converges to the batch model

**Logistic Regression vs. Decision Trees:**
* Both algorithms are really fast
* Logistic regression will work better if there's a single decision boundary
* Decision trees work best if the class labels roughly lie in hyper-rectangular regions
* Logistic regression has low variance and so is less prone to over-fitting
* Decision trees can be scaled up to be very complex, are more liable to overfitting

---
#### Logistic Regression - Practical Lab (Section 15)
* Logistic regression on the titanic dataset:
* Imports:
~~~
import pandas as pd
from sklearn.linear_model import LogisticRegression
~~~
* Load Data, using a subset of columns:
~~~
train = pd.read_csv('train.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age', 'Fare'])
test = pd.read_csv('test.csv')
~~~
* Turn the `Sex` features into a dummy variable:
~~~
train['Sex'] = train['Sex'].map({'male': 1, 'female':0})
~~~
* To impute the media value for missing values in the `Age` and `Fare` features:
~~~
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
~~~
* Split training data into feature matrix `X` and target vector `y`:
~~~
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
~~~
* Instantiate and fit a logistic regression model:
~~~
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
~~~
* Clean the test data in the same manner in which we cleaned the train data:
~~~
test = test[['Pclass', 'Sex', 'Age', 'Fare']]
test['Sex'] = test['Sex'].map({'male':1, 'female':0})
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
~~~
* Predict on the test data using the logistic regression from above:
~~~
y_pred = logreg.predict(test)
~~~
* Calculate Probability of Survival:
~~~
preds = logreg.predict_proba(test)
~~~

---
#### K-Nearest Neighbors - Theory (Section 16)
* Method for classifying objects based on the closest training examples in the features space.
* KNN is a type of instance-based or lazy learning where the function is only approximated locally and all computation is delayed until classification.
* Simplest classification technique to be used when there is little or no prior knowledge about the distribution of the data.
* The k in KNN refers to the number of nearest neighbors the classifier will use to make its prediction.
* k should always be an odd number to break ties when classifying a new observation
* Can also be used for regression

**Proximity Metrics:**
* Euclidean distance
* Hamming distance
* Manhattan distance (city block)
* Minkowsky distance
* Chebychev distance

**Advantages:**
* Robust to noisy training data
* Effective if training data is large
* There is no training phase
* Learns complex models easily

**Disadvantages:**
* Need to determine the value of the parameter k
* Struggles with high-dimensional data
    * Low computational efficiency
    * Data sparsity
    * Large amount of data and storage required
    * False intuition
    * Distance between data objects become less distinct
* Not clear which type of distance metric to use
* Computation cost is high  

---
#### K-Nearest Neighbors - Practical Lab (Section 17)
* Imports:
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
~~~
* Load data:
~~~
glass = pd.read_csv('glassClass.csv')
~~~
* Create feature matrix `X` and target vector `y`:
~~~
X = glass.drop('Type', axis=1)
y = glass['Type']
~~~
* Train/test split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=25)
~~~
* Instantiate and fit a KNN model with k=3:
~~~
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
~~~
* Predict responses:
~~~
y_pred = knn.predict(X_test)
~~~
* Calculate model accuracy:
~~~
accuracy_score(y_test, y_pred)
~~~

**Parameter Tuning with Cross-Validation:**
* Imports:
~~~
from sklearn.model_selection import cross_val_score
~~~
* Cross validate with neighbors range 1-50:
~~~
neighbors = list(range(1,50))
cv_score = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
cv_score
~~~

---
#### Support Vector Machines - Theory (Section 18)
* Suited for extreme cases
* Extreme data points are known as support vectors
* SVM is a frontier which best segregates the two classes (hyperplane/line)
* SVM implies that only support vectors are important whereas other training examples are ignorable.
* Use kernel trick to transform data into higher dimensions

**Popular Kernel Types:**
* Polynomial Kernel
* Radial Basis Function RBF Kernel
* Sigmoid Kernel

**Advantages:**
* Effective in high dimensional spaces
    * Still effective in cases when our number of features are greater than our number of observations (wide data)
* Memory efficient
* Versatile - different kernel functions for various decision functions

**Disadvantages:**
* Poor performance when # features > # observations
* SVMs do not provide probability estimates

---
#### Support Vector Machines (Linear SVM Classification) - Practical Lab (Section 19)
* Imports:
~~~
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
~~~
* Load data:
~~~
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
~~~
* Set hyperparameters:
~~~
h = .02 # Step size in the mesh
C = 1.0 # SVM regularization parameter
~~~
* Train/Test Split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=25)
~~~
* Instantiate and fit linear SVM model:
~~~
svc = svm.SVC(kernel='linear', C=C)
svc.fit(X_train, y_train)
~~~
* Make predictions:
~~~
y_pred = svc.predict(X_test)
~~~
* Calculate accuracy:
~~~
accuracy_score(y_test, y_pred)
~~~
* Display confusion matrix:
~~~
confusion_matrix(y_test, y_pred)
~~~

---
#### Support Vector Machines (Non-Linear SVM Classification) - Practical Lab (Section 20)
* Imports:
~~~
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
~~~
* Load data:
~~~
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
~~~
* Set hyperparameters:
~~~
h = .02 # step size in the mesh
C = 1 # SVM regularization parameter
~~~
* Train/Test split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=25)
~~~
* Instantiate, fit, predict, and score a SVM model with an rbf kernel:
~~~
svc1 = svm.SVC(kernel='rbf', gamma=0.7, C=C)
svc1.fit(X_train, y_train)
y_pred = svc1.predict(X_test)
accuracy_score(y_test, y_pred)
~~~
* Now, instantiate, fit, predict and score a SVM model with a polynomial kernel:
~~~
svc2 = svm.SVC(kernel='poly', degree=3, C=C)
svc2.fit(X_train, y_train)
y_pred = svc2.predict(X_test)
accuracy_score(y_test, y_pred)
~~~

---
#### Naive Bayes - Theory (Section 21)
* Naive Bayes is based on Bayes Rule, which is also known as conditional theorem.
    * Bayes theorem is a mathematical formula for how much you should trust the evidence.

**Bayes Terminology:**
* Prior Probability: Describes the degree to which we believe the model accurately describes reality based on all of our prior information.
* Likelihood: Describes how well the model predicts the data
* Normalizing constant: The constant that makes the posterior density integrate to one
* Posterior Probability: Represents the degree to which we believe a given model accurately describes the situation given the available data and all of our prior information

**Advantages:**
* Easy and fast to predict a class of test data set
* Naive bayes classifier preforms better compared to other models assuming independence
* Performs well in the case of categorical input variables compared to numerical variables

**Disadvantages:**
* Zero frequency
    * To account for zero frequency we can use laplace estimation or adding 1 for simple cases to avoid dividing by zero.
* Naive bayes is known as a bad estimator
* Assumes independent predictors (in real life this is rarely true)

---
#### Naive Bayes - Practical Lab (Section 22)
* Imports:
~~~
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
~~~
* A gaussian naive bayes algorithm is a special type of naive bayes algorithm
    * It's specifically used when the features have continuous values
    * It assumes that all features are following a gaussian distribution (i.e., normal distribution)
* Load data:
~~~
iris = sns.load_dataset('iris')
X = iris.drop('species', axis=1)
y = iris['species']
~~~
* Train/Test Split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=25)
~~~
* Instantiate, fit, and predict on default Gaussian NB model:
~~~
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
~~~
* Create confusion matrix:
~~~
confusion_matrix(y_test, y_pred)
~~~
* Calculate accuracy:
~~~
accuracy_score(y_test, y_pred)
~~~

**Another Example:**
* Imports:
~~~
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
~~~
* Load Data:
~~~
glass = pd.read_csv('glassClass.csv')
X = glass.drop('Type', axis=1)
y = glass['Type']
~~~
* Train/Test Split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=25)
~~~
* Instantiate, fit, and predict on default Gaussian NB model:
~~~
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
~~~
* Create confusion matrix:
~~~
confusion_matrix(y_test, y_pred)
~~~
* Calculate accuracy:
~~~
accuracy_score(y_test, y_pred)
~~~

**Multinomial NB:**
* Multinomial naive bayes is suitable for classification with discrete features.
* Imports:
~~~
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
~~~
* Load Data:
~~~
glass = pd.read_csv('glassClass.csv')
X = glass.drop('Type', axis=1)
y = glass['Type']
~~~
* Train/Test Split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=25)
~~~
* Instantiate, fit, and predict on Multinomial NB model:
~~~
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
~~~
* Create confusion matrix:
~~~
confusion_matrix(y_test, y_pred)
~~~
* Calculate accuracy:
~~~
accuracy_score(y_test, y_pred)
~~~

---
#### Clustering (Section 23)
* Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters).
* It is a main task of exploratory data mining, and a common technique for statistical data analysis, used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, bioinformatics, data compression, and computer graphics.
* Types:
    1) K-means Clustering
    2) Hierarchical Clustering

---
#### K-Means Clustering - Theory (Section 24)
* The K-Means algorithm is a traditional, simple, machine learning algorithm that is trained on a test dataset and then able to classify a new dataset using a k number of predefined clusters
* The K in K-Means is the number of clusters we would like to group our data into
* K-Means is an unsupervised machine learning algorithm

**K-Means Algorithm Flow Chart:**
1) Initialize the center of the clusters
2) Attribute the closest cluster to each data point
3) Set the position of each cluster to the mean of all data points belonging to that cluster
4) Repeat steps 2-3 until convergence

**What is Clustering?**
* Process of quantitatively partitioning a group of data points into smaller subsets.

**Deciding Number of Clusters:**
* She be based on the data itself.
* An incorrect choice of the number of clusters will invalidate the whole process.
* An empirical way to find the optimal clusters is to cross-validate on the number of clusters and measure the resulting sum of squares.
    * 'Elbow' plot

**When to use K-Means Clustering?:**
* Best when datasets are distinct or well separated from each other in a linear fashion.
* Best to use when the number of cluster centers, is specified due to a well-defined list of types shown in the data.
* Will not perform well when there is heavily overlapping data.

**The Equation:**
* Goal of minimizing an objective function, which in this case is the squared error function.

**Advantages:**
* Faster than hierarchical clustering if k is small
* May produce tighter clusters than hierarchical clustering

**Disadvantages:**
* Difficult in comparing quality of the clusters produced
* Fixed number of clusters can make it difficult to predict what k should be
* Strong sensitivity to outliers and noise
* Doesn't work well with non-circular cluster shapes
* Low capability to pass the local optimum

---
#### K-Means Clustering (Iris) - Practical Lab (Section 25)
* Imports:
~~~
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, confusion_matrix
~~~
* Load Data:
~~~
iris = sns.load_dataset('iris')
~~~
* Label encoding of species column numerically:
~~~
le = LabelEncoder()
le.fit(iris['species'])
iris['species'] = le.transform(iris['species'])
~~~
* Create matrix:
~~~
iris_matrix = pd.DataFrame.as_matrix(iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
~~~
* Create model:
~~~
cluster_model = KMeans(n_clusters=3, random_state=10)
cluster_model.fit(iris_matrix)
~~~
* View clusters:
~~~
cluster_model.labels_
~~~
* Create a column with predicted clusters:
~~~
iris['pred'] = cluster_model.fit_predict(iris_matrix)
~~~
* Plot clusters:
~~~
sns.FacetGrid(iris, hue='species', size=5).map(plt.scatter, 'sepal_length', 'sepal_width').add_legend()
plt.show()
~~~
* Compute accuracy score:
~~~
accuracy_score(iris['species'], iris['pred'])
~~~
* Compute adjusted rand score:
~~~
adjusted_rand_score(iris['species'], iris['pred'])
~~~
* Create confusion matrix:
~~~
confusion_matrix(iris['species'], iris['pred'])
~~~

---
#### K-Means Clustering (Glass) - Practical Lab (Section 26)
* Imports:
~~~
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, confusion_matrix
~~~
* Load Data:
~~~
glass = pd.read_csv('glassClass.csv')
~~~
* Label encoding of Type column numerically:
~~~
le = LabelEncoder()
le.fit(glass['Type'])
glass['Type'] = le.transform(glass['Type'])
~~~
* Create matrix:
~~~
glass_matrix = pd.DataFrame.as_matrix(glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']])
~~~
* Create model:
~~~
cluster_model = KMeans(n_clusters=7, random_state=10)
cluster_model.fit(glass_matrix)
~~~
* Create a column with predicted clusters:
~~~
glass['pred'] = cluster_model.fit_predict(glass_matrix)
~~~
* Plot clusters:
~~~
sns.FacetGrid(glass, hue='pred', size=5).map(plt.scatter, 'RI', 'Na').add_legend()
plt.show()
~~~
* Compute accuracy:
~~~
accuracy_score(glass['Type'], glass['pred'])
~~~
* Compute adjusted rand score:
~~~
adjusted_rand_score(glass['Type'], glass['pred'])
~~~

---
#### Hierarchical Clustering - Theory (Section 27)
**Hierarchical Clustering:**
* Algorithm that builds hierarchy of clusters

**Difference from K-Means:**
* HCA can't handle big data as well as
* Results are reproducible in hierarchical clustering unlike k-means
* K-means words better with globular clusters

**Distance Metrics:**
* Euclidean
* Squared Euclidean
* Manhattan
* Maximum
* Manhalanobis

**Linkage:**
* Methods differ in respect to how they define proximity between any two clusters at every step.
    1) Simple linkage: distance between two closest points of separate clusters
    2) Average linkage: distance between average points of separate clusters
    3) Complete linkage: distance between two most distant points of separate clusters



**HCA Types:**
1) Agglomerative Clustering (Bottom-up Approach)
2) Divisive Clustering (Top-down Approach)
    * Les popular
* General practice for selecting clusters: Use midpoint of longest branch

---
#### Hierarchical Clustering - Practical Lab (Section 28)
* Imports:
~~~
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
~~~
* Load data:
~~~
glass = pd.read_csv('glassClass.csv')
~~~
* Label encoding of Type column numerically:
~~~
le = LabelEncoder()
le.fit(glass['Type'])
glass['Type'] = le.transform(glass['Type'])
~~~
* Create matrix:
~~~
glass_matrix = pd.DataFrame.as_matrix(glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']])
~~~
* Model: Bottom-up algorithms treat each unit as a singleton cluster at the outset and then successively merge (or agglomerate) pairs of clusters until all clusters have been merged into a single cluster that contains all documents. Bottom-up hierarchical clustering is therefore called hierarchical agglomerative clustering or HAC.
* Create mode:
~~~
cluster_model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
glass['pred'] = cluster_model.fit_predict(glass_matrix)
~~~
* Calculate accuracy:
~~~
accuracy_score(glass['Type'], glass['pred'])
~~~
* Calculate adjusted_rand_score:
~~~
adjusted_rand_score(glass['Type'], glass['pred'])
~~~
* Visualize Clusters:
~~~
cluster_vis = sns.clustermap(glass.corr())
plt.show()
~~~

**Create Dendrogram:**
* A Dendrogram illustrates how each cluster is composed by drawing a U-shaped link between a non-singleton cluster and its children
* Imports:
~~~
from scipy.cluster.hierarchy import linkage, dendrogram
~~~
* Load Data:
~~~
happy = pd.read_csv('2015.csv', usecols=['Happiness Score', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual'])
~~~
* Create linkage matrix:
~~~
Z = linkage(happy, 'ward')
~~~
* Plot dendrogram:
~~~
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.xlabel('sample index')
plt.ylabel('distance')
plt.show()
~~~
* To truncate full dendrogram (in this case to last 12 clusters):
~~~
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=8., show_contracted=True)
plt.xlabel('sample index')
plt.ylabel('distance')
plt.show()
~~~

---
#### Associated Rule Learning (Section 30)
* Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases.
* It is intended to identify strong rules discovered in databases using some measures of interestingness
* Two main algorithms:
    1) Eclat
    2) Aprior

---
#### Aprior - Theory (Section 31)
**Terminology:**
* Antecedent: 'If' (Left-hand side)
* Consequent: 'Then' (Right-hand side)

**Example:**
* Grocery shopping: 'If milk and sugar, then coffee'

**Apriori Algorithm:**
* Useful in recommender systems and market basket analysis
* Rules:
    1) All subsets of a frequent item sets must be frequent
    2) Similarly, for any infrequent item set, all its supersets must be infrequent too

**Support:**
* Proportion of transactions in the dataset in which the item set appears.
* Signifies the popularity of an item set.
* Equation: `supp(X) = Number of transactions in which X appears/Total number of transactions`

**Confidence:**
* Signifies the likelihood of item Y being purchased when item X is purchased.
* Equation: `conf(X -> Y) = supp(X U Y)/supp(X)`
* Only accounts for popularity of X and not Y. If Y is equally as popular as X then there is a high probability that transactions containing X will contain Y, thus increasing confidence.
    * Lift accounts for this.

**Lift:**
* Signifies the likelihood of an item Y being purchased when item X is purchased while taking in account of the popularity of item Y.
* Equation: `lift(X -> Y) = supp(X U Y)/supp(X) * supp(Y)`
* If lift > 1, Y is likely to be bought with X
* If lift < 1, Y is unlikely to be bought with X

**Conviction:**
* Equation: `conv(X -> Y) = 1 - supp(Y)/1 - conf(X -> Y)`

**Advantages:**
* Easy to implement and understand
* Can be used in large datasets
* Can be parallelized

**Disadvantages:**
* Computationally expensive

---
#### Apriori - Practical Lab (Section 32)
* Imports:
~~~
import numpy as np
import pandas as pd
~~~
* Load Data:
~~~
all_ratings = pd.read_csv('ml-100k/u.data', delimiter='\t', header=None, names=['UserID', 'MovieID', 'Rating', 'Datetime'])
~~~
* Change all_rating['Datetime'] to datetype dtype:
~~~
all_ratings['Datetime'] = pd.to_datetime(all_ratings['Datetime'], unit='s')
~~~
* Create a column that specifies if the rating was favorable or not (rating > 3):
~~~
all_ratings['Favorable'] = all_ratings['Rating'] > 3
~~~
* Sample the dataset:
~~~
sample_ratings = all_ratings[all_ratings['UserID'].isin(range(200))]
~~~
* Create a dataset of each user's favorable reviews:
~~~
favorable_ratings = sample_ratings[sample_ratings['Favorable']]
~~~
* To find only the viewers with more than one review:
~~~
favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_ratings.groupby("UserID")["MovieID"])
len(favorable_reviews_by_users)
~~~
* To find out how many movies have favorable ratings:
~~~
num_favorable_by_movie = sample_ratings[['MovieID', 'Favorable']].groupby('MovieID').sum()
num_favorable_by_movie.sort_values(by='Favorable', ascending=False)
~~~
* Create initial frequent itemset to work with items that have minimum support:
~~~
from collections import defaultdict

def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])
~~~
* Identify films with more than 50 favorable reviews:
~~~
import sys
frequent_itemsets = {}  # itemsets are sorted by length
min_support = 50 #cut-off

# k=1 candidates are the isbns with more than min_support favourable reviews
frequent_itemsets[1] = dict((frozenset((movie_id,)), row["Favorable"])
                                for movie_id, row in num_favorable_by_movie.iterrows()
                                if row["Favorable"] > min_support)

print("There are {} movies with more than {} favorable reviews".format(len(frequent_itemsets[1]), min_support))
sys.stdout.flush()
for k in range(2, 20):
    # Generate candidates of length k, using the frequent itemsets of length k-1
    # Only store the frequent itemsets
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k-1],
                                                   min_support)
    if len(cur_frequent_itemsets) == 0:
        print("Did not find any frequent itemsets of length {}".format(k))
        sys.stdout.flush()
        break
    else:
        print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
        #print(cur_frequent_itemsets)
        sys.stdout.flush()
        frequent_itemsets[k] = cur_frequent_itemsets
# We aren't interested in the itemsets of length 1, so remove those
del frequent_itemsets[1]
~~~
* Apriori algorithm to identify the frequent sets. Use these to build our association rules.
~~~
candidate_rules = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset: #iterate over each movie- conclusion
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))
print("There are {} candidate rules".format(len(candidate_rules)))
~~~
* Now, compute the confidence of each of these rules by creating a dictionary to store the number of times the premise leads to the conclusion:
~~~
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
rule_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
              for candidate_rule in candidate_rules}
~~~
* Choose only rules above a minimum confidence level:
~~~
min_confidence = 0.8
~~~
* Filter out the rules with poor confidence:
~~~
rule_confidence = {rule: confidence for rule, confidence in rule_confidence.items() if confidence > min_confidence}
from operator import itemgetter
sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)
~~~
* Print Rules:
~~~
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise, conclusion))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")
~~~
* Do the same but with movie names:
~~~
movie_name_data = pd.read_csv('ml-100k/u.item', delimiter="|", header=None, encoding = "mac-roman")
movie_name_data.columns = ["MovieID", "Title", "Release Date", "Video Release", "IMDB", "<UNK>", "Action", "Adventure",
                           "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                           "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def get_movie_name(movie_id):
   title_object = movie_name_data[movie_name_data["MovieID"] == movie_id]["Title"]
   title = title_object.values[0]
   return title

for index in range(5):
   print("Rule #{0}".format(index + 1))
   (premise, conclusion) = sorted_confidence[index][0]
   premise_names = ", ".join(get_movie_name(idx) for idx in premise)
   conclusion_name = get_movie_name(conclusion)
   print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
   print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
   print("")  
~~~

---
#### Eclat - Theory (Section 33)
**Introduction:**
* ECLAT: Equivalence CLAss Transformation
* An algorithm for discovering itemset (group of items) mining occurring frequently in a transaction database (frequent itemsets).
    * Itemset mining: finding frequent patterns in data; often for consumer goods purchased in tandem.
        * AKA: Association Rule
    * Useful in recommender systems.

**Overview of Eclat:**
* Depth-first algorithm as opposed to breadth-first search
* Suitable for parallel execution
* Locality enhancing properties

**Mining Association Rules:**
* Find rules the occurrence of an item based on the occurrences of other items in the transaction.

**Eclat Algorithm Characteristics:**
* Efficient frequent itemset generation
* Represents data in vertical format instead of horizontal
* Improves on Apriori Algorithm
    * Other alternatives to scale Apriori:
        * FPGrowth
        * Mining Close Frequent Patterns
        * Max Patterns Eclat

**Advantages of Eclat:**
* Depth-first search reduces memory requirements
* No need to scan the database to find the support of (K+1) itemsets, for k >= 1.

**Disadvantages of Eclat:**
* TID-sets can be quite long, hence expensive

---
#### Eclat - Practical Lab (Section 34)
* Not worth notes.

---
#### Dimensionality Reduction (Section 35)
* Dimensionality reduction is the process of reducing the number of random variables under consideration, via obtaining a set of principal variables.
* It can be divided into feature selection and feature extraction.

---
#### Principle Component Analysis - Theory (Section 36)
**Problems in Higher Dimensions:**
* It involves high time and space complexity
* Significant computing power and thus cost
* There is risk of over-fitting
* Not all features in our dataset are relevant to our problems, some features are more relevant than others.

**Eigenevectors and Eigenvalues:**
* An eigenvector is a direction
* An eigenvalue is a number, telling you how much variance there is in the data in that direction.
*The eigenvector with the highest eigenvalue is therefore the principal component.*
* Eigenvectors and eigenvalues that exist = number of dimensions
* Eigenvectors must be orthogonal to one another

**Dimension Reduction:**
* Remove what is unnecessary
* k-D to 3-D or 2-D or 1-D

**PCA Flowchart:**
Step 1: Standardize Data
Step 2: Calculate Covariance of Matrix
Step 3: Deduce Eigen's
Step 4: Re-Orient and Plot Data
Step 5: Bi-Plot

**Benefits of Dimensionality Reduction:**
* Sorts out multi-collinearity
* Improves the model performance
* Removes redundant features

---
#### Principal Component Analysis - Practical Lab (Section 36)
* Imports:
~~~
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
~~~
* Load data:
~~~
iris = sns.load_dataset('iris')
~~~
* Create feature matrix X:
~~~
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
~~~
* Scale values:
~~~
X = scale(X)
~~~
* Describe Four Principal Components:
~~~
pca = PCA(n_components=4)
pca.fit(X)
~~~
* Display the amount of variance that each PC explains:
~~~
pca.explained_variance_ratio_
~~~
* Plot the variance from above:
~~~
plt.plot(var)
plt.show()
~~~
* Plot the cumulative variance:
~~~
cumvar = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(cumvar)
plt.show()
~~~

---
#### Linear Discriminant Analysis - Theory (Section 37)
**Introduction:**
* Another dimensionality reduction technique that serves as an alternative to PCA.
    * PCA: Component axes that maximize the variance
    * LDA: Maximizing the component axes for class-separation.

**PCA vs. LDA:**
* Both LDA and PCA are used for dimensionality reduction
* PCA is an unsupervised algorithm
* LDA is a supervised algorithm
    * LDA is superior to PCA for multi-class classification tasks when the class labels are known.  

**LDA Step-by_Step:**
Step 1: Calculate the separability between the different classes.
    * The between-class variance.
Step 2: Calculate the distance between the means and the samples within each class.
    * The within-class variance.
Step 3: Construct a lower-dimension space which maximizes the between-class variance and the within-class variance.
Step 4: Project original data using the calculated lower-dimensional space.

**LDA Disadvantages:**
1) Small Sample Size (SSS)
    * Solved with regularization (RLDA) subspace and null space methods.
2) Linearity Problems
    * Solved with kernel trick (Used in SVM)

---
#### Linear Discriminant Analysis - Practical Lab (Section 39)
* Imports:
~~~
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
~~~
* Import data:
~~~
iris = sns.load_dataset('iris')
~~~
* Create feature matrix X and target vector y:
~~~
X = iris.drop('species', axis=1)
y = iris['species']
~~~
* Perform train/test split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=25)
~~~
* Perform LDA for classification:
~~~
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
~~~
* Use LDA for dimensionality reduction and plot those components:
~~~
lda2 = LinearDiscriminantAnalysis(n_components=2)
components = lda2.fit(X, y).transform(X).T

plt.scatter(components[0], components[1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('LDA')
plt.show()
~~~

---
#### Artificial Neural Networks (Section 40)
* Artificial neural networks (ANNs), a form of connectionism, are computing systems inspired by the biological neural networks.
* Such systems learn (progressively improving performance) to do tasks by considering examples, generally without task-specific programming.
* Best for applications in which traditional computer algorithms using rule-based programming fails.
* Types:
    * Artificial Neural Networks
    * Convolutional Neural Networks
    * Recurrent Neural Networks

---
#### Artificial Neural Networks - Theory (Section 41)
**Parts of a Perceptron:**
* Inputs
* Weights
* Summation and Bias
* Activation Function
* Output

**Layers of a Neural Network:**
* A traditional neural network consists of three layers:
    1) Input layer (Features)
    2) Hidden Layer (Additional Neurons)
    3) Output Layer (Value to Predict)
* A neural network with multiple hidden layers is called deep learning.

**Activation Function:**
* Popular activation functions:
    * Threshold/Step Function
    * Sigmoid/Logistic Function - Most commonly used
    * Hyperbolic Tangent Function
    * Rectified Linear Unit (RELU) - Not as computationally expensive

**Gradient Descent vs. Stochastic Gradient Descent:**
* Gradient Descent: Computes the gradient using the whole dataset.
    * May get stuck in the local minima
* SGD: Computes the gradient using the single sample from dataset
    * Can converge faster

**Learning Types:**
* Supervised learning: the desired output is known
* Unsupervised learning: network classifies input data
* Reinforcement learning: the output is unknown, network provides feedback.
* Offline learning: Weight vector and threshold adjustments done after training set is presented
    * AKA: Batch learning
* Online learning: weight vector and threshold adjustments done after each sample

---
#### Artificial Neural Network (Perceptron) - Practical Lab (Section 42)
* Imports:
~~~
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
~~~
* Import data and create feature matrix X and target vector y:
~~~
cancer = load_breast_cancer()
X = cancer['data']
y = cancer['target']
~~~
* Perform train/test split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=25)
~~~
* Create Perceptron:
~~~
per = Perceptron(random_state=40)
per.fit(X_train, y_train)
y_pred = per.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
~~~

---
#### Artificial Neural Network (Multi-Layer Perceptron) - Practical Lab (Section 43)
* Imports:
~~~
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
~~~
* Import data and create feature matrix X and target vector y:
~~~
cancer = load_breast_cancer()
X = cancer['data']
y = cancer['target']
~~~
* Perform train/test split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=25)
~~~
* Standardize Data:
~~~
scaler = StandardScaler() # Fit only to training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
~~~
* Create Multi-Layer Perceptron:
~~~
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
~~~

---
#### Artificial Neural Network (Multi-Layer Perceptron) - Practical Lab (Section 44)
* Imports:
~~~
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
~~~
* Import data and create feature matrix X and target vector y:
~~~
glass = pd.read_csv('glassClass.csv')
X = glass.drop('Type', axis=1)
y = glass['Type']
~~~
* Perform train/test split:
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y)
~~~
* Scale Data:
~~~
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
~~~
* Create Multi-Layer Perceptron:
~~~
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
~~~
* Instead, customize the Multi-Layer Perceptron:
~~~
mlp2 = MLPClassifier(activation='logistic', solver='lbfgs', alpha=0.0001, random_state=1)
mlp2.fit(X_train, y_train)
y_pred = mlp2.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
~~~
* Again, customize the Multi-Layer Perceptron:
~~~
mlp3 = MLPClassifier(activation='logistic', solver='adam', hidden_layer_sizes=(200, 25), alpha=0.001, random_state=1, max_iter=300)
mlp3.fit(X_train, y_train)
y_pred = mlp3.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
~~~

---
#### Convolutional Neural Networks - Theory (Section 45)
**Convnet Architecture:**
1) Convolution Layer
2) Non-Linearity also as (Relu) rectified linear unit
3) Pooling or sub sampling
4) Classification or fully connected layer

![CNN](https://www.mathworks.com/content/mathworks/www/en/discovery/convolutional-neural-network/jcr:content/mainParsys/image_copy.adapt.full.high.jpg/1508999490138.jpg)

**Other:**
* A CNN will typically have more hyperparameters than an MLP.
    * Batch size: represents the number of training examples being used simultaneously during a single iteration of the gradient descent algorithm.
    *  Epochs: The number of times the training algorithm will iterate over the entire training set before terminating
    * Kernel sizes in the convolutional layers
    * Pooling size in the pooling layers
    * Number of kernals in the convolutional layers
    * Dropout probability (apply dropout after each pooling, and after the fully connected layer) -- prevent overfitting.
    * Number of neurons in the fully connected layer of the MLP.

---
#### Convolutional Neural Network - Practical Lab (Section 46)
* Imports:
~~~
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
import numpy as np
~~~
* Set hyperparameters:
~~~
batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 200 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per convolutional layer
conv_depth_2 = 64 # switch to 64 kernels after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons
~~~
* Perform test/train split and create feature matrix and target vector:
~~~
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10
num_test = X_test.shape[0] # There are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # there are 10 image classes

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalize data to [0, 1] range
X_test /= np.max(X_test) # Normalize data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels
~~~
* Build model:
    * Four convolution2D layers, with a MaxPooling2D layer following after the second and the fourth convolution
    * The output of the second pooling layer is flattened to 1D (via the Flatten layer), and passed through two fully connected (Dense) layers
    * ReLU activations will once again be used for all layers except the output dense layer, which will use a softmax activation (for purposes fo probabilistic classification)
    * Dropout uses regularization to precent overfitting
~~~
inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)
~~~
* Model:
~~~
model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
optimizer='adam', # using the Adam optimiser
metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train, # Train the model using the training set...
batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!
~~~

---
#### Recurrent Neural Networks - Theory (Section 47)
**RNN Overview:**
* Make use of sequential information
* RNN's have some sort of memory
* Commonly used in NLP

**Vanishing and Exploding Gradients:**
* When you're training an deep neural network, your gradients can become either exponentially large (exploding) or exponentially small (vanishing).
* Solutions
    * Gradient Clipping (for exploding gradients)
    * ReLU activation function (for vanishing gradients)
    * Long-term short-term memory
    * Identify initialization

---
#### Reccurrent Neural Network - Practical Lab (Section 48)
* Imports:
~~~
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
import numpy
numpy.random.seed(7)
~~~
* Import Keras:
~~~
from keras import backend as K
K.set_image_dim_ordering('th')
~~~
* Load Data and test/train split:
~~~
from keras.datasets import imdb
from matplotlib import pyplot
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=5000)

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
~~~
* Build Model:
~~~
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(5000, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64)
~~~
* Get scores:
~~~
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
~~~
