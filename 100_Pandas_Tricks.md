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
