### Data School Top 25 Pandas Tricks
#### August 2019

#### 1) Show Installed Versions and Dependencies
* Installed version:
```
import pandas as pd
pd.__version__
```
* Versions of pandas dependencies:  
```
pd.show_versions()
```

#### 2) Create An Example DataFrame
* Via dictionary:
```
df = pd.DataFrame({'col one': [100, 200], 'col two': [300, 400]})
```
* Via numpy random generator:
```
df = pd.DataFrame(np.random.rand(4, 8))
```
* Via numpy random generator with column names :
```
df = pd.DataFrame(np.random.rand(4, 8), columns=list('abdcefgh'))
```

#### 3) Rename Columns
* Rename method:
```
df = df.rename({'old_name': 'new_name'}, axis='columns')
```
* List method:
```
df.columns = ['new_name']
```
* Add prefix to all columns:
```
df.add_prefix('X_')
```
* Add suffix to all columns:
```
df.add_suffix('_Y')
```

#### 4) Reverse Row Order
* Keep same index
```
df.loc[::-1]
```
* The same with new index:
```
df.loc[::-1].reset_index(drop=True)
```

#### 5) Reverse Column Order
```
df.loc[:, ::-1]
```

#### 6) Select Columns by Data Type
```
df.select_dtypes(include='number')

# OR

df.select_dtypes(exclude='object')

# OR

df.select_dtypes(include['number', 'object'])
```

#### 7) Convert Strings to Numbers
```
df = pd.DataFrame({'col_one': ['1.1', '2.2', '3.3'],
                   'col_two': ['4.4', '5.5', '6.6'],
                   'col_three': ['7.7', '8.8', '-']})

df = df.astype({'col_one': 'float',
                'col_two': 'float'})

# OR

df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
```

#### 8) Read In Only A Select Number of Columns
```
df = pd.read_csv(path, usecols=column_list)
```

#### 9) Build a DataFrame From Multiple Files (Row-Wise)
```
from glob import glob
file_list = sorted(glob(path/*.csv))

df = pd.concat((pd.read_csv(file) for file in file_list), ignore_index=True)
```

#### 10) Build a DataFrame From Multiple Files (Column-Wise)
```
from glob import glob

file_list = sorted(glob(path*.csv))

df = pd.concat((pd.read_csv(file) for file in file_list) axis='columns')
```

#### 11) Create a DataFrame From the Clipboard
```
df = pd.read_clipboard()
```

#### 12) Split a DataFrame Into Two Random Subsets
```
df_1 = df.sample(frac=0.75, random_state=10)
df_2 = df.drop(df_1.index)
```

#### 13) Filter a DataFrame By Multiple Columns
* To include only certain columns:
```
df[df.column.isin(['a', 'b', 'c'])]
```
* To exclude only certain columns:
```
df[~df.column.isin(['a', 'b', 'c'])]
```

#### 14) Filter a DataFrame By Largest Categories
```
# Counts by category in a column
counts = df.column.value_counts()
df[df.column.isin(counts.nlargest(3).index)]
```
#### 15) Handle Missing Values
* To get a count of null values in each column:
```
df.isna().sum()
```
* To find the percent of nulls in each column:
```
df.isna().mean()
```
* To drop columns with more than a certain percent of nulls:
```
df.dropna(thresh=len(df)*.9, axis=columns)
```

#### 16) Split a String Into Multiple Columns
* Split a string column by comma into three separate columns
```
df[['col1', 'col2', 'col3']] = df.column.str.split(',', expand=True)
```
* Do the same as above but only create a new column for the last item
```
df['col1'] = df.column.str.split(',', expand=True)[-1]
```

###3 17) Expand a Series of Lists Into a DataFrame
```
df = pd.DataFrame({'col_one': ['a', 'b', 'c'], 'col_two': [[10, 40], [20, 50], [30, 60]]})

df_new = df.col_two.apply(pd.Series)

df_og = pd.concat([df, df_new], axis='columns')
```

#### 18) Aggregate by Multiple Functions
```
df = df.groupby('col1').col2.agg(['sum', 'mean'])
```

#### 19) Combine the Output of an Aggregation with a DataFrame
* **This is super cool!** Appends aggregate amounts back onto original dataframe
```
df = df.groupby('col1').col2.transform('sum')
```

#### 20) Select a Slice of Rows and Columns
```
df.describe().loc['min':'max', 'col1': 'col5']
```

#### 21) Reshape a MultiIndexed Series
```
df.groupby(['col1', 'col2']).col3.mean().unstack()
```

#### 22) Create a Pivot Table
```
df.pivot_table(index='col1', columns='col2', values='col3', aggfunc='mean')
```

#### 23) Convert Continuous Data Into Categorical Data
```
pd.cut(df.col1, bins=[0, 10, 20, 30], labels='0-10', '11-20', '21-30', '30+')
```

#### 24) Change Display Options
```
pd.set_option('display.float_format', '{:.2}'.format)
```

#### 25) Style a DataFrame
```
format_dict = {'col1': '{:.2f}', 'col2': '{:.2f}'}

df.style.format(format_dict)

df.style.format(format_dict)
  .hide_index()
  .highlight_min('col1', color='red')
  .highlight_max('col2', color='green')
  .set_caption('Example Styled DataFrame')
 ```
