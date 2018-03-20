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
