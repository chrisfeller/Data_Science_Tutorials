### Data School Web Scraping Tutorial
#### October 2017


**Example: Web Scrape the list of lies told by President Trump in the July 21st, 2017 New York Times article *Trump's Lies***
* The link to the article is: https://www.nytimes.com/interactive/2017/06/23/opinion/trumps-lies.html

#### Part 1: Getting Started
**What is Web Scraping?:**
* The process of extracting information from a web page by taking advantage of patterns in the web pages underlying code.
* To view a web pages underlying (HTML) code:
    1) Right click on the web page
    2) Select "View Page Source"

**Three Facts to Know HTML:**
1) HTML consists of tags, specified using angle brackets (<>) that mark up the text.
    * HTML stands for Hypertext Markup Language.
    * Example: `<strong>`, which means use bold formatting on the text "Jan. 21"
        ~~~
        <strong>Jan. 21&nbsp;</strong>
        ~~~
2) HTML tags can have attributes, which are specified in the opening tag.
    * Example: `<span class="short-desc">` - The `span` tag has a class attribute with a value of shot-desc.
        * You don't need to understand the meaning of this but only that these attributes exist.
3) Tags can be nested.
    * Example:
        ~~~
        Hello <strong><em>Data School</em> students</strong>
        ~~~
        * Outputs as hello ***Data School*** **students**
        * The italics tag is nested within the bold tags.
        * Tags mark up text from wherever they open to wherever they close regardless of whether they are nested within other tags.

#### Part 2: Parsing HTML with BeautifulSoup
* To read an article into python:
~~~
import requests
r = requests.get('https://www.nytimes.com/interactive/2017/06/23/opinion/trumps-lies.html')
~~~
* `r` is called a response object, which has a text attribute, which contains the same HTML code we saw when viewing the source of the web page from our web browser:
~~~
print(r.text[0:500])
~~~
* To parse using the BeautifulSoup library:
~~~
from bs4 import BeautifulSoup
soup = BeautifulSoup(r.text, 'html.parser')
~~~
* By looking at the HTML we notice that each of the records (lie's) we are trying to extract following the following format:
~~~
<span class="short-desc"><strong> DATE </strong> LIE <span class="short-truth"> <a href="URL"> EXPLANATION </a></span></span>
~~~
* To find all of the records, we must search for all of the span tags with the class attribute set to short-desc:
~~~
results = soup.find_all('span', attrs={'class':'short-desc'})
~~~
* `results` acts as a python list so we can check the number of records:
~~~
len(results)
~~~
* We can also slice it to look at the first few results:
~~~
results[0:3]
~~~
* We now need to separate the records four parts: the date, the lie, the explanation, and the link.

#### Part 3: Building a Dataset
* To extract the date, let's practice with the first record in the results object. We'll iterate over all records later.
~~~
first_result = results[0]
first_result.find('strong').text #Returns a python string.
~~~
* To remove the escape characters at the end of the python string:
~~~
first_result.find('strong').text[0:-1]
~~~
* To add the year to each date:
~~~
first_result.find('strong').text[0:-1] + ', 2017'
~~~
* To extract the lie:
~~~
first_result.contents
~~~
* The first_result tag has an attribute `.contents`, which is a list of its children.
    * Children are tags and strings that are nested within a tag.
* To get the lie, which is the second child:
~~~
first_result.contents[1]
~~~    
* To slice off the curly quotation marks and the space at the end:
~~~
first_result.contents[1][1:-2]
~~~
* To extract the explanation of the lie we can either search for the surrounding tag, like we did when searching for the date, or we can slice the contents attribute when extracting the lie. We will use the tag:
~~~
first_result.find('a').text[1:-1]
~~~
* To extract the URL that proves the the lie:
~~~
first_result.find('a')['href']
~~~

**Recap: Beautiful Soup Methods and Attributes:**
* You can apply these two methods to either the initial soup object or a Tag object:
    1) `find()`: searches for the first matching tag, and returns a Tag object.
    2) `findall()`: searches for all matching tags, and returns a ResultSet object, which you can treat like a list of tags.
* You can extract information from a Tag object using these two attributes:
    1) `text`: extracts the text of a Tag, and returns a string.
    2) `contents`: extracts the children of a Tag, and returns a list of Tags and strings.

**Iterate Through The Entire Document:**
~~~
records = []
for result in results:
    date = result.find('strong').text[0:-1] + ', 2017'
    lie = result.contents[1][1:-2]
    explanation = result.find('a').text[1:-1]
    url = result.find('a')['href']
    records.append((date, lie, explanation, url))
~~~

#### Part 4: Exporting a CSV with pandas
* To apply a tabular data structure to our results:
~~~
import pandas as pd
df = pd.DataFrame(records, columns=['date', 'lie', 'explanation', 'url'])
~~~
* To change the 'date' column to datetime:
~~~
df['date'] = pd.to_datetime(df['date'])
~~~
* To export the dataframe as a .csv file:
~~~
df.to_csv('trump_lies.csv', index=False, encoding='utf-8')
~~~

**Web Scraping Advice:**
* Web scraping works best with static, well-structured web pages.
* Web scraping is a 'fragile' approach for building a dataset.
* If you can download the data you need from a website, or if the website provides an API with data access, those approaches are preferable to scraping.
* If you are scraping a lot of pages from the same website, it's best to insert delays in your code.
* Before scraping a website, you should review its robots.txt file. 
