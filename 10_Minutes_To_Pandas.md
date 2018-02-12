### 10 Minutes to pandas
#### October 2017

---
**Object Creation:**
* The four biggest import conventions:
~~~
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
~~~

**Object Creation:**
* To create a Series, pass a list to Series:
~~~
s = pd.Series([1, 3, 5, np.nan, 6, 8])
~~~
* By default, pandas will create a default integer index.
* To create a DataFrame with a timeseries as its index:
~~~
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
~~~
* To create a DataFrame from a dictionary of objects:
~~~
df = pd.DataFrame({'A': 1., 'B': pd.Timestamp('20130102'), 'C': pd.Series(1, index=list(range(4)), dtype='float32'), 'D': np.array([3] * 4, dtype='int32'), 'E': pd.Series(['test', 'train', 'test', 'train'], dtype='category'), 'F': 'foo'})
~~~


**Viewing Data:**
* To check the datatypes of columns within a DataFrame:
~~~
df.dtypes
~~~
* To view the top n rows of a DataFrame or Series (n defaults to 5):
~~~
s.head(n)
df.head(n)
~~~
* To view the bottom n rows of a DataFrame or Series (n defaults to 5):
~~~
s.tail(n)
df.tail(n)
~~~
* To display the index of a DataFrame or Series:
~~~
s.index
df.index
~~~
* To view the column names of a DataFrame:
~~~
df.columns
~~~
* To view the values of a Series or DataFrame:
~~~
s.values
df.values
~~~
* To view a quick statistic summary of each column in a DataFrame:
~~~
df.describe()
~~~
* To view a quick statistic summary of one specific column of a DataFrame:
~~~
df['column'].describe()
~~~

**Transposing and Sorting DataFrames:**
* To transpose a DataFrame:
~~~
df.T
~~~
* To sort a DataFrame by its index:
~~~
df.sort_index(axis=1, ascending=True)
~~~
* To sort a DataFrame by its column names:
~~~
df.sort_index(axis=0, ascending=True)
~~~
* To sort a DataFrame by the values within a specific columns:
~~~
df.sort_values(by='column')
~~~

**Selection:**
* To select a single column of a DataFrame, which returns a Series:
~~~
df['column']
~~~
* To select a single row of a DataFrame:
~~~
df.loc[5, :]
~~~
* To select multiple columns of a DataFrame:
~~~
df.loc[:['column1', 'column2']]
~~~
* To select multiple rows of a DataFrame:
~~~
df.loc[5:10, :]
~~~
* To slice rows and columns:
~~~
df.loc[5:10, ['column1', 'column2']]
~~~

**Boolean Indexing (Fancy Indexing):**
* To get all of the values within a DataFrame where a boolean condition is met:
~~~
df[df > 0]
~~~
* To get all of the rows in a DataFrame with a value greater than zero in a specific column:
~~~
df[df.column > 0]
~~~
* To select all of the rows that have specific values in a specific row:
~~~
df[df.column].isin(['value', 'value2'])
~~~

**Setting:**
* Setting a new column automatically aligns the data by the indexes:
~~~
s = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
df['new_column'] = s
~~~

**Missing Data:**
* pandas primarily uses the value `np.nan` to represent missing data, which by default is not included in computation.
* To drop any rows that have missing data in one or more columns:
~~~
df.dropna(how='any')
~~~
* To drop any rows that have missing data in all columns:
~~~
df.dropna(how='all')
~~~
* To fill missing data:
~~~
df.fillna(value=5)
~~~
* To get the boolean mask where values are `nan`:
~~~
pd.isnull(df)
~~~

**Operations:**
* To perform a descriptive statistic on all columns of a DataFrame:
~~~
df.mean()
~~~
* To perform a descriptive statistic on all rows of a DataFrame:
~~~
df.mean(axis=1)
~~~

**Apply:**
* To apply a function to a DataFrame:
~~~
df.apply(np.cumsum)
~~~
* To apply an anonymous function to a DataFrame:
~~~
df.apply(lambda x: x.max() - x.min())
~~~

**Counting Values:**
* To return a count of each value within a column:
~~~
df['column'].value_counts()
~~~

**String Methods:**
* To apply a string method to a column in a DataFrame:
~~~
df['column'].str.upper()
~~~

**Merge:**
* To concatenate different DataFrames with the same columns by row (meaning add to the bottom of the rows):
~~~
df.concat(df2)
~~~

**Join:**
* To join DataFrames in an SQL-style merge:
~~~
pd.merge(df, df2, on='column')
~~~

**Append:**
* To append rows to a DataFrames:
~~~
df.append(x, ignore_index=True)
~~~

**Grouping:**
* By 'groupby' we are referring to a process involving one or more of the following steps:
    1) Splitting the data into groups based on some criteria
    2) Applying a function to each group independently
    3) Combining the results into a data structure
* To group by the categorical values in one column and then return a descriptive statistic on those categories:
~~~
df.groupby('column').mean()
~~~
* To group by multiple columns, which forms a hierarchical index:
~~~
df.groupby(['column1', 'column2']).sum()
~~~

**Reshaping:**
* The `stack()` method 'compresses' a level in the DataFrame's columns:
~~~
df.stack()
~~~
* The `unstack()` method unstacks the last level:
~~~
df.unstack()
~~~

**Pivot Tables:**
* To create a pivot table of a DataFrame:
~~~
pd.pivot_table(df, values='columns', index=['column2', 'column3', columns='columns4'])
~~~

**Time Series:**
* To create a DateTime index:
~~~
rng = pd.date_range('1/1/2012', periods=100, freq='S')
~~~
* To create a timeseries:
~~~
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
~~~

**Categoricals:**
* To create a column with categorical variables from an existing columns:
~~~
df['new_column'] = df['column'].astype('category')
~~~
* To rename existing categories:
~~~
df['new_column'].cat.categories = ['very good', 'good' , 'bad']
~~~

**Plotting:**
* To plot a column of a DataFrame:
~~~
df['column'].plot()
~~~

**Getting Data In/Out:**
* To read data from a .csv:
~~~
pd.read_csv('file.csv')
~~~
* To output to a .csv:
~~~
df.to_csv('file.csv')
~~~
* To read data from an excel:
~~~
pd.read_excel('file.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
~~~
* To output to an excel:
~~~
df.to_excel('file.xlsx', sheet_name='Sheet1')
~~~
