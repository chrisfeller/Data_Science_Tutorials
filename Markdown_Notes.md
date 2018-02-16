<center> <h1>Markdown Notes</h1> </center>

Markdown is a plain text formatting syntax, which allows you to quickly write content for reporting, and have it seamlessly converted to clean, structured HTML.

### Headers
Headings in Markdown are in line which is prefixed with a # symbol. The number of hashes indicates the level of the heading.
# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6
    # Heading 1
    ## Heading 2
    ### Heading 3
    #### Heading 4
    ##### Heading 5
    ###### Heading 6

### Text
*italic*
**bold**
***bold-italic***
[link](http://example.com)

    *italic*
    **bold**
    ***bold-italic***
    [link](http://example.com)

[Example Link](http://example.com)
To add a link: wrap the text which you want to be linked in square brackets, followed by the URL to be linked to in parenthesis.
[button] (place to go when button is clicked)

### Images

![Planet Money](http://media.npr.org/assets/img/2015/12/18/planetmoney_sq-c7d1c6f957f3b7f701f8e1d5546695cebd523720-s300-c85.jpg)

Markdown images have exactly the same formatting as a link, except they're prefixed with a !.
    ![Image Description] (http://linktoimage.com)

This also works for images on your machine.

    ![Image Description] (/Users/chrisfeller/Downloads/shotchart.png)

### Lists
* Milk
* Bread
    * Wholegrain
* Butter


    * Milk
    * Bread
        * Wholegrain
    * Butter

1. Tidy the kitchen
2. Prepare ingredients
3. Cook delicious things


    1. Tidy the kitchen
    2. Prepare ingredients
    3. Cook delicious things

* Monday
    1. Prepare Post-Game Report
    2. Draft Model
* Tuesday
    1. Positional Cluster Analysis


    * Monday
        1. Prepare Post-Game Report
        2. Draft Model
    * Tuesday
        1. Positional Cluster Analysis

### Quotes
> The only thing holding you back is you are worried about what other people have to say. My friends... You have one life. You have to do you.
- Gary Vaynerchuck


Code:
    > The only thing holding you back is you are worried about what other people have to say. My friends... You have one life. You have to do you.
    - Gary Vaynerchuk

### Horizontal Rules

Want to throw-down a quick divider in your report to denote a visual separation between different sections of text? No problem, 3 dashes produce:

---

    ---


### Code Snippets
Some text with an inline `code = 5` snippet

    Some text with an inline `code = 5` snippet

Use a single back-tick around a word or phrase in a sentence, to show a quick `code` snippet.
Indenting by 4 spaces will turn an entire paragraph into a code block.

    [float(x) for x in [team_advance[team_advance.TEAM_CITY=='Denver']['DEF_RATING']]][0]

### Reference Lists & Titles
To reference a link at the bottom of your markdown. Use the following.

    The two books I recommended to Ben Falk were [The Cubs Way][1] and [Principles: Life and Work][2].

    [1]: https://www.amazon.com/Cubs-Way-Building-Baseball-Breaking/dp/0804190011
    [2]: https://www.amazon.com/Principles-Life-Work-Ray-Dalio/dp/1501124021

### Escaping
To escape Markdown characters use a back-slash `\`

*literally* vs. \*literally\*

    \*literally\*

### Superscript and Subscript
32^nd^
H~2~O

### Text Highlighting
==Text to be highlighted==

### Text Highlighting
[⋅] Unchecked Item

[X] Checked Item
