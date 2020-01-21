### Data School Pandas Best Practices (PyCon 2019)
#### January 2020

---
**Link**
* [Video](https://www.youtube.com/watch?v=dPwLlJkSHLo)
* [Data](https://github.com/justmarkham/pycon-2019-tutorial)

**1. Intro to the Dataset**
```
# Imports
import pandas as pd
import matplotlib.pyplot as plt

# Read in data
ted = pd.read_csv('ted.csv')

# Number of columns and rows
ted.shape

# Datatypes
ted.dtypes

# Check number of nulls in each column
# Documentation is moving to `isna()` instead of `isnull()`
ted.isna().sum()

# Check percent of nulls in each column
ted.isna().sum()/len(ted)
```

**2. Which Talks Provoke the Most Online Discussion**
```
# Sort the dataframe by the number of comments in descending order
ted.sort_values(by='comments', ascending=False).head()[['description', 'comments']]

# Normalize by the number of views (comment to view ratio)
ted['comment_view_ratio'] = ted['comments']/ted['views']
ted.sort_values(by='comment_view_ratio', ascending=False).head()[['description', 'comment_view_ratio']]

# Re-do the last command with chaining syntax
ted.assign(comment_view_ratio=ted['comments']/ted['views'])\
   .sort_values(by='comment_view_ratio', ascending=False)\
   .head()[['description', 'comment_view_ratio']]
```

**3. Visualize the distribution of comments**
```
import seaborn as sns
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(12, 5))
sns.distplot(ted.loc[:, 'comments'], kde=False)
ax.set(xlabel='Number of Comments', ylabel='Log')
ax.set_yscale('log')
plt.title('Distribution of Comments')
plt.tight_layout()
plt.show()
```
