### Pandas Profiling Tutorial
#### February 2018

#### Introduction
* Library, which creates HTML profiling reports form pandas DataFrame objects
* Similar to df.describe() but on steroids
* For each column in a DataFrame, it provides the following statistics - if relevant for the column type:
    * Essentials: type, unique values, missing values
    * Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range
    * Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
    * Most frequent values
    * Histogram
    * Correlations highlighting of highly correlated variables, Spearman and Pearson matrixes
* Repo: https://github.com/pandas-profiling/pandas-profiling

#### Installation
* pip: `pip install pandas-profiling`
* conda: `conda install pandas-profiling`

#### Usage
* Recommends using Jupyter Notebook for pandas profiling:
~~~
import pandas as pd
import pandas_profiling
df = pd.read_csv('/Users/chrisfeller/Desktop/sandbox/Meteorite_Landings.csv')
pandas_profiling.ProfileReport(df)
~~~
* To save the HTML report file:
~~~
profile = pandas_profiling.ProfileReport(df)
profile.to_file(outputfile='/Users/chrisfeller/Desktop/sandbox/profile.html')
~~~
