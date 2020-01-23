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

**4. Plot the number of talks that took place each year**
```
import seaborn as sns
plt.style.use('fivethirtyeight')

ted_agg = ted.assign(year=
                    pd.to_datetime(ted['film_date'], unit='s').dt.year)\
             .groupby('year')\
             .count()\
             .reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(x='year', y='comments', data=ted_agg, color='royalblue')
ax.set(xlabel='Year', ylabel='Event Count')
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.title('Number of Ted Talks Per Year')
plt.tight_layout()
plt.show()
```

**5. What were the 'best' events in TED history to attend?**
```
ted_agg = ted.groupby('event')\
             .views\
             .agg(['count', 'sum'])\
             .sort_values(by='sum', ascending=False)
```

**6. Unpack the ratings data**
```
import ast

ted['ratings_list'] = ted.ratings.apply(lambda x: ast.literal_eval(x))
```

**7. Count the total number of ratings received by each talk**
```
def get_num_ratings(list_of_dicts):
    num = 0
    for d in list_of_dicts:
        num = num + d['count']
    return num

ted['num_ratings'] = ted.ratings_list.apply(lambda x: get_num_ratings(x))
```

**8. Which occupations deliver the funniest TED talks on average?**
```
# 1. Get count of funny ratings
def get_funny_ratings(list_of_dicts):
    for d in list_of_dicts:
        if d['name']=='Funny':
            return d['count']

ted['funny_rating'] = ted['ratings_list'].apply(lambda x: get_funny_ratings(x))

# 2. Create a rate of funny comments to total comments
ted['funny_rate'] = ted['funny_rating'] / ted['num_ratings']

# 3. Analyze the funny rate by occupation
ted.groupby('speaker_occupation')['funny_rate'].mean()\
            .sort_values(ascending=False)

# 4. Focus on occupations that are well-represented in the data
occupation_counts = ted['speaker_occupation'].value_counts()
top_occupations = occupation_counts[occupation_counts >=5].index

```
