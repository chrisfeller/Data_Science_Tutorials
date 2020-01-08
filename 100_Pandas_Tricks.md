### Data School 100 Pandas Tricks
#### January 2020

**1) 5 useful `.read_csv()` parameters**
1. `names`: specify column names
2. `usecols`: which columns to use
3. `dtype`: specify data types
4. `nrows`: number of rows to read
5. `na_values`: strings to recognize as NaN

**2) Use these `.read_csv()` parameters if you have bad data (or empty rows) at teh top of your csv file**
* `header`: row number of header (zero indexed)
* `skiprows`: list of row numbers to skip
* Example: `pd.read_csv('data.csv', header=2, skiprows=[3, 4])`

**3) Two easy ways to reduce DataFrame memory usage**
1. Only read in columns you need
2. Use `category` data type with categorical data
* Example: `pd.read_csv('data.csv', usecols=['A', 'C', 'D'], dtype={'D':'category'})`

**4) You can read or write directly from/to a compressed file**
* Read: `pd.read_csv('https://h.co/3JAwA8h7FJ')`
* Write: `df.to_csv('https://h.co/3JAwA8h7FJ')`

**5) Avoid having a column named `Unnamed: 0` by setting the first column as the index (when reading) or don't save the index to the file (when writing)**
* Read: `pd.read_csv('data.csv', index_col=0)`
* Write: `df.to_csv('data.csv', index=False)`

**6) Combine dataset rows spread across multiple files into single DataFrame**
* Use `glob` to list your files and then use a generator expression to read files and `concat()` to combine them.
* Example:
    ```
    import pandas as pd
    from glob import glob

    # Step 1: Use glob function to list all files that match a pattern
    stock_files = sorted(glob('data/stocks*.csv'))

    # Step 2: Use a generator expression to read the files and use concat to combine them
    pd.concat((pd.read_csv(file) for file in stock_files), ignore_index=True)
    ```

**7) Combine dataset rows spread across multiple files into single DataFrame, while keeping track of which rows come from which files**
* Use `glob` to list your files and then read files with generator expression, creating a new column with `assign()`, and combine with `concat()`
* Example:
    ```
    import pandas as pd
    from glob import glob

    # Step 1: Use glob function to list all files that match a pattern
    stock_files = sorted(glob('data/stocks*.csv'))

    # Step 2: Use a generator expression to read the files, assign() to create a new column, and concat() to combine the DataFrames
    pd.concat((pd.read_csv(file).assign(filename=file) for file in stock_files), ignore_index=True)
    ```

**8) Read in a huge dataset into pandas by sampling rows during the file read**
* Randomly sample the dataset during the file reading by passing a function 'skiprows'.
* Example:
    ```
    df = pd.read_csv('big_data.csv', skiprows = lamba x: x > 0 and np.random.rand() > 0.01)
    ```
* How it works:
    - `skiprows` accepts a function that is evaluated against the integer index
    - `x > 0` ensures that the header row is **not** skipped
    - `np.random.rand() > 0.01` returns True 99% of the time, thus skipping 99% of the rows.

**9) Read in data from Excel or Google Sheets via the clipboard**
* Copy data to the clipboard and then `pd.read_clipboard()`

**10) Extract tables from a PDF into a DataFrame**
* Use `tabula-py` library
* Example:
    ```
    from tabula import read_pdf
    df = read_pdf('test.pdf', pages='all')
    ```

**11) Read a JSON from the web**
* Use `read_json()` to read it directly from a URL into a DataFrame
* Example:
    `df = pd.read_csv('https://api.github.com/users.justmarkham/repos?per_page=100')`

**12) Scrape a table from a web page**
* Try `read_html()` before trying `BeautifulSoup` and `Selenium`
* Example: `df = pd.read_html('https://finance.yahoo.com/quote/AAPL?p-APPL')`

**13) Use `read_html()` to scrape a webpage containing multiple tables**
* If you're using `read_html()` to scrape a webpage but it's returning too many tables, use the `match` parameter to find tables that contain a particular string.
* Example:
    ```
    import pandas as pd

    url = 'https://en.wikipedia.org/wiki/Twitter'

    tables = pd.read_html(url)
    # returns 17 tables
    len(tables)

    matching_tables = pd.read_html(url, match='Followers')
    # returns one table
    len(matching_tables)
    ```

**14) Create an example DataFrame**
* Three ways:
    1. `pd.DataFrame({'col_one': [10, 20], 'col_two': [30, 40]})`
    2. `pd.DataFrame(np.random.rand(2, 3), columns=list('abc'))`
    3. `pd.util.testing.makeMixedDataFrame()`

**15) Create a DataFrame for testing**
* Three ways:
    1. `pd.util.testing.makeDataFrame()` # random values
    2. `pd.util.testing.makeMissingDataframe()` # includes missing values
    3. `pd.util.testing.makeTimeDataFrame()` # has DateTime index
    4. `pd.util.testing.makeMixedDataFrame()` # mixed data types

**16) Create a time series dataset for testing**
* You can create a timeframe index in an existing dataframe:
    ```
    df = pd.DataFrame({'col_one': [10, 20], 'col_two': [30, 40]})

    df.index = pd.util.testing.makeDateIndex(len(df), freq='H')
    ```

**17) Create new columns or overwrite existing columns within a method chain**
* Use `assign()`
    ```
    (df.assign(col1 = df.col1.lower(),
               col2 = df.col1.upper()))
    ```

**18) Create a bunch of new columns based on existing columns**
* Use this pattern:
    ```
    for col in df.columns:
    df[f'{col}_new'] = df[col].apply(my_function)
    ```
* Example:
    ```
    for col in df.columns:
        df[f'{col}_fixed'] = df[col].str.upper()
    ```

**19) Remove a column from a DataFrame and store it was a seperate Series**
* Use `pop()`
* Example:
    ```
    label = iris.pop('species')
    ```

**20) Insert a new column into a DataFrame at a specific location**
* Use `insert()`
* Example:
    ```
    df.insert(3, 'C2', df['C'] * 2)
    ```

**21) Three ways to rename a column**
1. Most flexible option:
    `df = df.rename({'A': 'a', 'B': 'b'}, axis='columns')`
2. Overwrite all columns names:
    `df.columns = ['a', 'b']`
3. Apply string method:
    `df.columns = df.columns.str.lower()`

**22) Add a prefix or suffix to all of your column names**
* Prefix:
    `df.add_prefix('X_')`
* Suffix:
    `df.add_suffix('_Y')`

**23) Rename all columns in the same way**
* Replace spaces with an underscore:
    `df.columns = df.columns.str.replace('', '_')`
* Make lowercase and remove trailing whitespace:
    `df.columns = df.columns.str.lower().rstrip()`

**24) Use f-strings to select columns in a DataFrame**
* You can use f-strings (Python 3.6+) when selecting a Series from a DataFrame:
    `df[f'{}_servings]` # selects all columns with `serving` in name

**25) Select multiple rows/columns using `.loc`**
* Select a slice (inclusive):
    `df.loc[0:4, 'col_A':'col_D']`
* Select a list:
    `df.loc[[0, 3], ['col_A', 'col_C']]`
* Select a condition:
    `df.loc[df.col_A=='val', 'col_D']`

**26) Select by both label and position**
* 'loc' selects by label and 'iloc' selects by position. To select by label and position you can still use 'loc' and 'iloc'.
* Use `loc` with a mix of label and position:
    `df.['a', df.columns[0]]` or `df.loc[df.index[0], 'col_one']`
* Use `iloc` with a mix of label and position:
    `df.iloc[df.index.get_loc('a'), 0]`
* Don't use `ix` as it was deprecated.

**27) Chain 'loc' and 'iloc' together**
* To select from a DataFrame by label and position you can chain together `loc` and `iloc`:
    `drinks.iloc[15:20, :].loc[:, 'beer_servings':'wine_servings']`

**28) Reverse column and row order**
* Reverse column order in a DataFrame:
    `df.loc[:, ::-1]`
* Reverse row order:
    `df.loc[::-1]`
* Reverse row order and reset the index:
    `df.loc[::-1].reset_index(drop=True)`

**29) Select multiple slices of columns from a DataFrame**
1. Use `df.loc` to select and `pd.concat` to combine:
    `pd.concat([df.loc[:, 'A':'C'], df.loc[:, 'F'], df.loc[:, 'J': 'K']], axis='columns')`
2. Slice `df.columns` and select using brackets:
    `df[list(df.columns[0:3]) + list(df.columns[5]) + list(df.columns[9:11])]`
3. Use `np.r_` to combine slices and `df.iloc` to select:
    `df.iloc[:, np.r_[0:3, 5, 9:11]]`

**30) Filter DataFrame by multiple OR conditions**
* Traditional way:
    `df[(df.color=='red') | (df.color=='green') | (df.color=='blue')]`
* Shorter way:
    `df[df.color.isin(['red', 'green', 'blue'])]`
* Invert the filter above:
    `df[~df.color.isin(['red', 'green', 'blue'])]`

**31) Count or percentage of rows that match a condition**
* To find the count of rows that match a condition:
    `(movies.content_rating=='PG').sum()`
* To find the percent of rows that match a condition:
    `(movies.content_rating=='PG').mean()`

**32) Filter a DataFrame to only include the largest categories**
* to filter a DataFrame to only include the largest categories?
1. Save `value_counts()` output:
    `counts = movies.genre.value_counts()`
2. Get the index of its `head()`
    `largest_categories = counts.head(3).index`
3. Use that index with `isin()` to filer the DataFrame:
    `movies[movies.genre.isin(largest_categories)]`

**33) Combine the smaller categories in a Series into a single category called 'other'**
1. Save the index of the largest values of `value_counts()`
2. Use `where()` to replace all other values with 'Other'
* Example:
    ```
    top_four = genre.value_counts().nlargest(4).index
    genre_updated = genre.where(genre.isin(top_four), other='Other')
    ```

**35) Combine the small categories in a Series (<10% frequency) into a single category**
1. Save the normalized value counts
2. Filter by frequency and save the index
3. Replace small categories with 'Other'
* Example:
    ```
    frequencies = genre.value_counts(normalize=True)
    small_categories = frequencies[frequencies < 0.10].index
    genre_updated = genre.replace(small_categories, 'Other')
    ```

**36) Simplify filtering with multiple conditions**
* Instead of filtering a DataFrame using lots of criteria, save the criteria as objects and use them to filter.
    ```
    crit1 = df.continent =='Europe'
    crit2 = df.beer_servings > 200
    crit3 = df.wine_servings > 200
    crit4 = df.spirit_servings > 100

    df[crit1 & crit2 & crit3 & crit4]
    ```

* Alternatively, use `reduce()` to combine the criteria:
    ```
    from functools import reduce
    criteria = reduce(lambda x: x, y: x & y, [crit1, crit2, crit3, crit4])

    df[criteria]
    ```

**37) Filter a DataFrame that doesn't have a name**
* Instead of creating an intermediate variable such as:
    ```
    temp = stocks.groupby('Symbol').mean()
    temp[temp.Close < 100]
    ```
* Use `query()`:
    ```
    stocks.groupby('Symbol').mean().query('Close < 100')
    ```

**38) Refer to a local variable with a `query()` string?**
* Prefix it with an @ symbol:
    ```
    mean_volumne = stocks.Volume.mean()
    stocks.query('Volume > @mean_volume')
    ```

**39) Use `query()` on a column name containing a space**
* Surround it with backticks:
    ```
    df.query('`column name` > 40')
    ```

**40) Concatenate two string columns**
1. Use a string method:
    `ufo.City.str.cat(ufo.State, sep=', ')`
2. Use a plus sign:
    `ufo.city + ', ' + ufo.State`

**41) Split a string into multiple columns**
* Use `str.split()` method, `expand=True` to return a DataFrame, and assign it to the original DataFrame:
    ```
    df[['first', 'middle', 'last']] = df.name.str.split(' ', expand=True)
    ```
* To do the same but only save the first name:
    ```
    df['first'] = df.name.str.split(' ', expand=True)[0]
    ```

**42) Split names of variable length into first_name and last_name**
1. Use `str.split(n=1)` to split only once (returns a Series of lists)
2. Chain `str[0]` and `str[1]` on the end to select the list elements
    ```
    df['first_name'] = df.name.str.split(n=1).str[0]
    df['last_name'] = df.name.str.split(n=1).str[1]
    ```

**43) Count the number of words in a Series**
* Use a string method to count the spaces and add 1
    `df['word_count'] = df.messages.str.count(' ') + 1`

**44) Convert string columns to numeric**
* If you have numbers stored as strings:
    `df.astype({'col1': 'int', 'col2': 'float})`
* However, the above will fail if you have any invalid input.
* Instead use:
    `df.apply(pd.to_numeric, errors='coerce')`

**45) Select columns by data type**
* Examples:
    ```
    df.select_dtypes(include='number')
    df.select_dtypes(include=['number', 'category', 'object'])
    df.select_dtypes(exclude=['datetime', 'timedelta'])
    ```

**46) Save a massive amount of memory by fixing data types**
* Use `int8` for small integers
* Use `category` for strings with few unique values
* Use `Sparse` if most values are 0 or NaN
* Example:
    ```
    df = df.astype({'Pclass': 'int8',
                    'Sex': 'category',
                    'Parch': 'Sparse[int]',
                    'Cabin': 'Sparse[str]'})
    ```

**47) Check to see if object column contains mixed data types**
* Use `df.col.apply(type).value_counts()` to check.

**48) Clean an object column with mixed data types**
* Use `replace` (not str.replace) and regex:
    `df['sales'] = df.sales.replace('[$,]', '', regex=True).astype('float')`

**49) Use logic to order categories**
* Create ordered category:
    ```
    cat_type = CategoricalDtype(['good', 'very_good', 'excellent'], ordered=True)

    df['quality'] = df.quality.astype(cat_type)
    ```
* Now you can sort and compare categories logically:
    ```
    df.sort_values('quality')

    df[df.quality > 'good']
    ```

**50) Ensure correct data types for many columns**
1. Create CSV of column names & dtypes
2. Read it into a DataFrame
3. Convert it to a dict
4. Use dict to specify dtypes of dataset
