### R Shiny Tutorial
#### August 2019

---
#### How to Start Shiny Video Series
#### Part 1: How to Build a Shiny App
**Official Tutorial**
* Link: https://shiny.rstudio.com/tutorial/

**Fundamental Pieces**
* There are two fundamental pieces of a shiny app
1. Server Instructions (`server`)
    - Instructions to the server of where to host the app
2. User Interface (`ui`)
    - Instructions on how to display the app

**App Template**
* The shortest viable shiny app
* Use this template to start all projects
```
library(shiny)
ui <- fluidPage()

server <- function(input, output) {}

shinyApp(ui = ui, server = server)
```

**Inputs and Outputs**
* Build your app around *inputs* and *output*
    - Inputs are things your user can toggle or add values to
    - Outputs are the R objects that the user sees in the app display
* Inputs and outputs are both functions added to the `fluidPage()` argument in the ui
    - Anything you add to `fluidPage()` will end up on the app display

**Adding Input Functions**
* Example of an Input Function (Displaying a Slider):
```
sliderInput(inputId = 'num',
            label = 'Choose a number',
            value = 25, min = 1, max = 100)
```
* This would go inside of the `fluidPage()` argument in the shiny template

**Input Function Types**
* Shiny provides 12 different kinds of input functions:
1. Buttons
    ```
    actionButton()
    submitButton()
    ```
2. Single Checkbox
    ```
    checkboxInput()
    ```
3. Checkbox Group
    ```
    checkboxGroupInput()
    ```
4. Date Input
    ```
    dateInput()
    ```
5. Date Range
    ```
    dateRangeInput()
    ```
6. File Input
    ```
    fileInput()
    ```
7. Numeric Input
    ```
    numericInput()
    ```
8. Password Input
    ```
    passwordInput()
    ```
9. Radio Buttons
    ```
    radioButtons()
    ```
10. Select Box
    ```
    selectInput()
    ```
11. Sliders
    ```
    sliderInput()
    ```
12. Text Input
    ```
    textInput()
    ```

**Input Function Syntax**
* Each input function takes in the same basic syntax:
    ```
    sliderInput(inputId = 'num', label = 'Choose a number', ...)
    ```
* `inputId` is the name of the input that you can refer to in your script
* `label` is the label to display that the user sees in the app
* To see the rest of the arguments for an input function view the documents by using the helper function:
    ```
    ?sliderInput
    ```

**Adding Output Functions**
* These are things like plots, tables, texts

**Output Function Types**
* There are eight output functions
1. Interactive Table
    ```
    dataTableOutput()
    ```
2. raw HTML
    ```
    htmlOutput()
    ```
3. Image
    ```
    imageOutput()
    ```
4. Plot
    ```
    plotOutput()
    ```
5. Table
    ```
    tableOutput()
    ```
6. Text
    ```
    textOutput()
    ```
7. A Shiny UI Element
    ```
    uiOutput()
    ```
8. Text
    ```
    verbatimTextOutput()
    ```
**Output Function Syntax**    
* To display output, add it to the `fluidPage()` with an `*Output()` function
    ```
    plotOutput(outputId = 'hist')
    ```
* `plotOutput` is the type of output to display
* `outputId` is the name of the output object to refer to later in your script

**Assembling Inputs Into Outputs via the Server**
* The server function transforms inputs into outputs
* The syntax for the server function is:
    ```
    server <- function(input, output){}
    ```
* There are three rules to the server function:
    1. Save objects to display to output$
        ```
        server <- function(input, output){
            output$hist <- #code
        }
        ```
        - This would be saving an output to the input function called `hist`
    2. Build objects to display with `render*()`
        ```
        server <- function(input, output){
            output$hist <- renderPlot({ #code})
        }
        ```
        - What you save into the output should be rendered
    3. Use `input` values with `input$`:
        ```
        server <- function(input, output){
            output$hist <- renderPlot({
                hist(rnorm(input$num))})
        }
        ```


**Render Functions**
* Use the `render*()` function to create the type of output you wish to make
* There are seven types of render functions
1. Interactive table (from a dataframe, matrix or other table-like structure)
    ```
    renderDataTable()
    ```
2. An image (saved as a link to a source file)
    ```
    renderImage()
    ```
3. A plot
    ```
    renderPlot()
    ```
4. A code block of printed out
    ```
    renderPrint()
    ```
5. A table (from a dataframe, matrix or other table-like structure)
    ```
    renderTable()
    ```
6. A character string
    ```
    renderText()
    ```
7. A Shiny UI element
    ```
    renderUI
    ```

**Sharing a Shiny App**
* Shiny Apps should be saved in one directory with:
    - `app.R`: script which ends with a call to `shinyApp()`
    - datasets
    - images
    - css
    - helper scripts
* The actual app must be called `app.R` for the server to run it.

**Using shinyapps.io**
* shinyapps.io is a free service from RStudio that allows you to share apps online
* The service is free, easy to use, secure, and scalable
* Will need to sign up at shinyapps.io which you will then need to link to RStudio IDE through a token process
* [shinyapps.io Tutorial](shiny.rstudio.com/articles/shinyapps.html)

**Shiny Server**
* A backend program that builds a linux web server specifically designed to host Shiny Apps
* An alternative if you don't want to host apps on RStudio's servers
* [Link](www.rstudio.com/products/shiny/shiny-server)
* Free and open source
* There is also Shiny Server Pro with more advanced features but isn't free

---
#### Part 2: How to Customize Reactions
**What is Reactivity?**
* Example: an excel cell that reacts to the input of an adjacent cell.
* Takes inputs and triggers outputs

**Terminology**
* *Reactive values* act as the data streams that flow through your app.
* The `input` list is a list of reactive values. The values show the current state of the inputs.
* Reactive values notify while reactive functions respond.

**Reactive Values**
* Reactive values work together with reactive functions. You cannot call a reactive value from outside of one.
* Reactive value example: `input$num`
* Reactive function example: `renderPlot()`
* Example putting the two together: `renderPlot({hist(rnorm(100, input$num))})`
* You need both reactive values and reactive functions to work together to makes things work in a shiny app.
* You can only call a reactive value from a function that is designed to work with one.

**Reactive Functions**
1. Use a code chunk to build (and rebuild) an object
2. The object will respond to changes in a set of reactive values

**Reactive Toolkits**
* There are only seven reactive functions you need to build an app.

**Display Output with `render*()`**
* Family of functions that create outputs you can use to display in a final shiny app
* Make objects to display
* Always save the result to `output$`
* Render options:
    1. `renderDataTable()`
    2. `renderImage()`
    3. `renderPlot()`
    4. `renderPrint()`
    5. `renderTable()`
    6. `renderText()`
    7. `renderUI()`

**Modularize Code with `reactive()`**
* Builds a reactive object (aka reactive expression) that can be used for any render function so as to avoid rebuilding the same object multiple times
* Example:
    ```
    data <- reactive({rnorm(input$num)})
    ```
* To call a reactive object you must call it as a function via `data()`
* A reactive expression is special in two ways:
    1. You call a reactive expression like a function
    2. Reactive expressions cache their values (the expression will return its most recent value, unless it has become invalidated)

**Prevent Reactions with `isolate()`**
* Returns the result as a non-reactive value
* Makes reactive objects non-reactive
* Example:
    ```
    isolate({input$title})
    ```

**Trigger Code with `observeEvent()`**
* Code is triggered via some sort of action
* Action buttons:
    ```
    actionButton(inputId = 'go', label = 'Click Me!')
    ```
* Action buttons are triggered via `observeEvent()`
* Example:
    ```
    observeEvent(input$clicks, {print(input$clicks)})
    ```
* There is also `observe`
* Example:
    ```
    observe({print(input$clicks)})
    ```
* Recap:
    - `observeEvent()` triggers code to run on the server
    - Specify precisely which reactive values should invalidate the observer
    - Use `observe()` for a more implicit syntax

**Delay Reactions with `eventReactive()`**
* A reactive expression that only responds to specific values
* Example:
    ```
    data <- eventReactive(input$go, {rnorm(input$num)})
    ```
* Recap:
    - Use `eventReactive()` to delay reactions
    - `eventReactive()` creates a reactive expression
    - You can specify precisely which reactive values should invalidate the expression

**Manage State with `reactiveValues()`**
* Creates a list of reactive values to manipulate programmatically
* Example:
    ```
    rv <- reactiveValues(data = rnorm(100))
    ```

---

#### Part 3: How to Customize Appearance
**Working with the HTML UI**
* The `fluidPage()` call is using the R language to assemble HTML code.
* We then use `input*()` and `output*()` functions to add reactive content.
* All of this is then saved to `ui` as the user interface

**Add Static Content**
* When writing R, you add content with tags functions
* Shiny provides R functions to recreate HTML tags
* In R `tags$h1()` is the equivalent of `<h1></h1>` in HTML
* Similarly, in R `tags%a()` is the equivalent of `<a></a>` in HTML

**Tags**
* There are 110 tag options in `tags`
    * To view these: `names(tag)`
* Tags are a list but the elements in the list are functions
    * Example: `tags$h1()` instead of `tags$h1`
* Tags Syntax
    * Example:
    ```
    tags$a(href = 'www.rstudio.com', 'RStudio')
    ```
* To add raw text:
    ```
    fluidPage('This is a block of text')
    ```
* To add a paragraph:
    ```
    fluidPage(
        tags$p('This is paragraph 1'),
        tags$p('This is paragraph 2')
        )
    ```
* To italicize text:
    ```
    fluidPage(
        tags$em('This is italicized text')
        )
    ```
* To add a code snipped:
    ```
    fluidpage(
        tags$code('This is a code snipped')
        )
    ```
* To insert a line break:
    ```
    fluidPage(
        'Line of text 1',
        tags$br(),
        'Line of text 2'
        )
    ```
* To insert a horizontal line:
    ```
    fluidPage(
        'This is a Shiny app',
        tags$hr(),
        'It is also a web page.'
        )
    ```
* To add an image:
    ```
    fluidPage(
        tags$img(height = 100,
                 width = 100,
                 src = 'http://www.rstudio.com/images/RStudio.2x.png')
        )
    ```
    * You can set `src` to a url or a file path which exists inside of a separate folder called `www`. You don't need to preface the path with `www`.

**Nesting Tags**
* To add tags together in one another:
    ```
    fluidPage(
        tags$p('This is a',
            tags$strong('Shiny'),
            'app.')
        )
    ```

**Wrapper Functions**
* Some tags functions come with a wrapper function, so you do not need to call `tags$`
* Functions that have an associated wrapper:
    1. `a()`: hyperlink
    2. `br()`: line break
    3. `code()`: code snippet
    4. `em()`: italicized text
    5. `h1()...h6()`: headers
    6. `hr()`: horizontal line
    7. `img()`: image
    8. `p()`: paragraph
    9. `strong()`: bold

**Passing HTML to R Shiny App**
* You can also pass raw HTML code into `fluidPage()`:
    ```
    fluidPage(HTML(actual html code here))
    ```

**Creating a Layout**
* You can determine where objects appear within the app using layout functions
* Layout functions add HTML that divides the UI into a grid
* The two main functions are:
    1. `fluidRow()`
    2. `column(width = 2)`

**`fluidRow()`**
* `fluidRow()` adds rows to the grid. Each new row goes below the previous rows
* Example:
    ```
    ui <- fluidPage(
            fluidRow(),
            fluidRow()
        )
    ```

**`column()`**
* `column()` adds columns within a row. Each new column goes to the left of the previous column.
* Specify the width and offset of each column out of 12
    * Width is how wide a column is (the max width of a column is 12)
    * Offset is how far to the left the column should appear
    * Width is a mandatory argument, while offset is optional
* Example:
    ```
    ui <- fluidPage(
        fluidRow(
            column(3),
            column(5)),
        fluidRow(
            column(4, offset = 8)
            )
        )
    ```
* To place an element in the grid, call it as an argument of a layout function:
    ```
    fluidRow('In the row')

    # OR

    column(2, plotOutput('hist'))
    ```

**Assemble Layers of Panels**
* Panels are the basic functionality in Shiny to group multiple elements into a single unit with it's own properties.
* Panels are created via the `wellPanel()` function.
    * Groups elements into a grey 'well'
* A visual way to group objects
* There are 12 Panel Types:
    1. `absolutePanel()`: Panel position set rigidly (absolutely), not fluidly
    2. `conditionalPanel()`: A JavaScript expression determines whether panel is visible or not
    3. `fixedPanel()`: Panel is fixed to browser window and does not scroll with the page
    4. `headerPanel()`: Panel for the app's title, used with `pageWithSidebar()`
    5. `inputPanel()`: Panel with grey background, suitable for grouping inputs
    6. `mainPanel()`: Panel for displaying output, used with `pageWithSidebar()`
    7. `navlistPanel()`: Panel for displaying multiple stacked `tabPanels()`. Uses sidebar navigation.
    8. `sidebarPanel()`: Panel for displaying a sidebar of inputs, used with `pageWithSidebar()`
    9. `tabPanel()`: Stackable panel. Used with `navPanel()` and `tabsetPanel()`
    10. `tabsetPanel()`: Panel for displaying multiple stacked `tabPanels()`. Uses tab navigation.
    11. `titlePanel()`: Panel for the app's title, used with `pageWithSidebar()`
    12. `wellPanel()`: Panel with grey background.

**`tabPanel()`**
* `tabPanel()` creates a stackable layer of elements. Each tab is like a small UI of its own.
* Each tab needs a title
* Example:
    ```
    tabPanel('Tab 1 Title', element)
    ```
* Used to work with `tabsetPanel()`, `navlistPanel()`, and `navbarPage()`

**`tabsetPanel()`**
* `tabsetPanel()` combines tabs into a single panel. Uses tabs to navigate between tabs.
* Example:
    ```
    fluidPage(
        tabsetPanel('tab 1', 'contents'),
        tabPanel('tab 2', 'contents'),
        tabPanel('tab 3', 'contents')
        )
    ```

**`navlistPanel()`**
* `navlistPanel()` combines tabs into a single panel. Use links to navigate between tabs.
* Example:
    ```
    fluidPage(
        tabPanel('tab 1', 'contents'),
        tabPanel('tab 2', 'contents'),
        tabPanel('tab 3', 'contents')
        )
    ```

**Shiny Layout Guide**
* [Link Here](http://shiny.rstudio.com/articles/layout-guide.html)

**Use a Prepackaged Layout**
* One of the most common ways to build a Shiny App is with `sidebarLayout()`
* Use `sidebarPanel()` and `mainPanel()` to divide app into two sections.
* Example:
    ```
    ui <- fluidPage(
        sidebarLayout(
            sidebarPanel(),
            mainPanel()
            )
        )
    ```
* Another common approach is to use `fixedPage()` which creates a page that defaults to a width of 724, 940, or 1170 pixels instead of `fluidPage()` or `fluidRow()` which adjusts to browser window.
* Example:
    ```
    ui <- fixedPage(
        fixedRow(
            column(5, ...)
            )
        )
    ```
* A third common approach is `navbarPage()` combines tabs into a single page. `navbarPage` replaces `fluidPage()` and requires a title.
* Example:
    ```
    navbarPage(title = 'Title',
        tabPanel('tab 1', 'contents'),
        tabPanel('tab 2', 'contents'),
        tabPanel('tab 3', 'contents')
        )
    ```

**`dashboardPage()`**
* `dashboardPage()` comes in the shinydashboard package.
* Pre-packed dashboard type layouts/
```
library(shinydashboard)
ui <- dashboardpage(
    dashboardHeader(),
    dashboardSidebar(),
    dashboardBody()
    )
```
* [shinydashboards tutorial](http://rstudio.github.io/shinydashboards)
* [shinydashboards webinar](www.rstudio.com/resources/webinars)

**Style with CSS**
* Cascading Style Sheets (CSS) are a framework for customizing the appearance of elements in a web page.
* You can style a web page in three ways:
    1. Link to an external CSS file
    2. Write a global CSS in header
    3. Write individual CSS in a tag's style attribute
* Shiny uses the Bootstrap 3 CSS framework: getbootstrap.com
* To link to an external CSS file put it in the `www` folder of your app directory. Then to use the theme from that .css file:
```
ui <- fluidPage(
        theme = 'file.css'
    )
```
* To learn CSS go through the [Codecademy Tutorial](http://www.codecademy.com/tracks/web)

---
#### Shiny Written Tutorials
#### Lesson 1: Welcome to Shiny
**Welcome to Shiny**
* Shiny is an R package that makes it easy to build interactive web applications (apps) straight from R,
* To install Shiny:
```
install.packages('shiny')
```

**Examples**
* The Shiny package has eleven built-in examples that each demonstrate how Shiny works.
* Each example is a self-contained Shiny app.
* The 'Hello Shiny' example plots a histogram of R's `faithful` dataset with a configurable number of bins.
```
library(shiny)
runExample('01_hello')
```
* List of other Example Apps:
```
runExample("01_hello")      # a histogram
runExample("02_text")       # tables and data frames
runExample("03_reactivity") # a reactive expression
runExample("04_mpg")        # global variables
runExample("05_sliders")    # slider bars
runExample("06_tabsets")    # tabbed panels
runExample("07_widgets")    # help text and submit buttons
runExample("08_html")       # Shiny app built from HTML
runExample("09_upload")     # file upload wizard
runExample("10_download")   # file download wizard
runExample("11_timer")      # an automated timer
```

**Structure of a Shiny App**
* Shiny apps are always contained in a single script called `app.R`
* That script `app.R` lives in a directory (for example, `newdir/`) and the app can be run with `runApp('newdir')`
* `app.R` has three components:
    1. A user interface object
        - `ui`
        - Controls the layout and appearance of your app.
    2. A server function
        - `server`
        - Contains the instructions that your computer needs to build your app.
    3. A call to the `shinyApp` function
        - Creates Shiny app objects from the `ui`/`server` pair.
* Prior to version 0.10.2, Shiny did not support single-file apps and the `ui` and `server` functions needed to be contained in separate scripts called `ui.R` and `server.R`, respectively.
    * This functionality is still supported in Shiny, however the current best practice is to have both in the same script as demonstrated in this tutorial.
* `ui` example:
```
library(shiny)

# Define UI for app that draws a histogram
ui <- fluidPage(
    # App title
    titlePanel('Hello Shiny!'),
    # Sidebar layout with input and output definitions
    sidebarLayout(
        # sidebarPanel for inputs
        sidebarPanel(
            #Input: Slider for the number of bins
            sliderInput(inputId = 'bins',
            label = 'Number of Bins:',
            min = 1,
            max = 50,
            value = 30)
            ),
    # Main panel for displaying outputs
    mainPanel(
        # Output: Histogram
        plotOutput(outputId = distplot)
        )
    )
)
```
* `server` example to accompany above:
```
# Define server logic required to draw a histogram
server <- function(input, output) {
    # Histogram of the Old Faithful Geyser Data ----
    # with requested number of bins
    # This expression that generates a histogram is wrapped in a call
    # to renderPlot to indicate that:
    #
    # 1. It is "reactive" and therefore should be automatically
    #    re-executed when inputs (input$bins) change
    # 2. Its output type is a plot
    output$distPlot <- renderPlot({
        x <- faithful$waiting
        bins <- seq(min(x), max(x), length.out = input$bins + 1)
        hist(x, breaks = bins, col = '#75AADB', border = 'white',
            xlab = 'Waiting time to next eruption (in mins)',
            main = 'Histogram of waiting times')
        })
}
```
* `shinyApp` is the last function call needed to make the app run.
```
shinyApp(ui = ui, server = server)
```
* Your R session will be busy while the Hello Shiny app is active, so you will not be able to run any R commands. R is monitoring the app and executing the app’s reactions. To get your R session back, hit escape or click the stop sign icon (found in the upper right corner of the RStudio console panel).

**Running an App**
* Every Shiny app has the same structure: an `app.R` file that contains `ui` and `server`.
* You can create a Shiny app by making a new directory and saving an `app.R` file inside it.
    * It's recommended that each app live in its own unique directory.
* You can run a Shiny app by giving the name of its directory to the function `runapp()`.
    * For example, if your Shiny app is in a directory called `my_app`, run it with the following code:
    ```
    library(shiny)
    runApp('my_app')
    ```
    * `runApp` is similar to `read.csv`, `read.table`, and many other functions in R. The first argument of `runApp` is the filepath from your working directory to the app's directory. The code above assumes that the app directory is in your working directory. In this case, the filepath is just the name of the directory.
* By default, Shiny apps display in 'normal' mode. However, the example apps like `runExample('01_hello')` are displayed in 'showcase mode', that displays both the app along with it's associated `app.R` script.
    * To display your app in showcase mode:
    ```
    runApp('app.R', display.mod = 'showcase')
    ```

**Relaunching Apps**
* To relaunch your Shiny App:
    1. Run `runApp('app.R')`
    2. Open the `app.R` script in your RStudio editor. RStudio will recognize the Shiny script and provide a `Run App` button (at the top of the editor). Either click this button to launch your app or use the keyboard shortcut: `Command` + `Shift` + `Enter`
* RStudio will launch the app in a new window by default, but you can also choose to have the app launch in a dedicated viewer pane, or in your external web browser.
    * Make your selection by clicking the icon next to `Run App`

**Recap**
* To create your own Shiny App:
    1. Make a directory named `myapp/` for your app
    2. Save your `app.R` script inside that directory
    3. Launch your app with `runApp` or RStudio's keyboard shortcuts
    4. Exit the Shiny app by clicking escape

#### Lesson 2: Build a User Interface
**Introduction**
* The boilerplate for any Shiny app should look like this:
```
library(shiny)

ui <- fluidPage(

    )

server <- function(input, output){

}

shinyApp(ui = ui, server = server)
```

**Layout**
* Shiny uses the function `fluidPage` to create a display that automatically adjusts to the dimensions of your user's browser window.
    - You lay out the user interface of your app by placing elements in the `fluidPage` function
    - For example, the `ui` function below creates a user interface that has a title panel and a sidebar layout, which includes a sidebar panel and a main panel. Note that these elements are placed within the `fluidPage` function.
    ```
    ui <- fluidPage(
        titlePanel('title panel'),

        sidebarLayout(
            sidebarPanel('sidebar panel'),
            mainPanel('main panel')
            )
        )
    ```
* `titlePanel` and `sidebarLayout` are the two most popular elements to add to `fluidPage`.
    - They create a basic Shiny app with a sidebar.
* `sidebarLayout` always takes two arguments:
    - `sidebarPanel` function output
    - `mainPanel` function output
    - These functions place content in either the sidebar or the main panels.
* The sidebar panel will appear on the left side of your app by default. To move it to the right side either provide `sidebarLayout` the optional argument `position = 'right'` or move the `mainPanel` above the `sidebarPanel` in the code.
* An alternative to `fluidPage` is `navbarPage` which gives your app a multi-page user interface that includes a navigation bar.
    * Or you can use `fluidRow` and `column` to build your layout from a grid system.

**HTML Content**
* You can add content to your Shiny app by placing it inside a `*Panel` function.
* To add more advanced content, use one of Shiny's HTML tag functions.
    - These functions parallel common HTML5 tags.
* Shiny function HTML5 equivalents:
    - `p`: paragraph
    - `h1`: first level header
    - `h2`: second level header
    - `h3`: third level header
    - `h4`: fourth level header
    - `h5`: fifth level header
    - `h6`: six level header
    - `a`: hyper link
    - `br`:  line break
    - `div`: division of text with a uniform style
    - `span`: in-line division of text with a uniform style
    - `pre`: text 'as-is' in a fixed width font
    - `code`: a formatted block of code
    - `img`: an image
    - `strong`: bold text
    - `em`: italicized text
    - `HTML`: directly passes a character string as HTML code

**Headers**
* To create a header element:
    1. Select a header function (e.g., h1 or h5)
    2. Give it the text you want to see in the header
* For example:
```
titlePanel(h1('My title'))
```
    - Can also be passed to `titlePanel`, `sidebarPanel`, `mainPanel`
    - You can place multiple elements in the same panel if you separate them with a comma.

**Formatted Text**
* An example of a layout with multiple formatting tags:
```
ui <- fluidPage(
  titlePanel("My Shiny App"),
  sidebarLayout(
    sidebarPanel(),
    mainPanel(
      p("p creates a paragraph of text."),
      p("A new p() command starts a new paragraph. Supply a style attribute to change the format of the entire paragraph.", style = "font-family: 'times'; font-si16pt"),
      strong("strong() makes bold text."),
      em("em() creates italicized (i.e, emphasized) text."),
      br(),
      code("code displays your text similar to computer code"),
      div("div creates segments of text with a similar style. This division of text is all blue because I passed the argument 'style = color:blue' to div", style = "color:blue"),
      br(),
      p("span does the same thing as div, but it works with",
        span("groups of words", style = "color:blue"),
        "that appear inside a paragraph.")
    )
  )
)
```

**Images**
* Shiny looks for the `img` function to place image files in your app.
* To insert an image, give the `img` function the name of your image file as the `src` argument (e.g., `img(src = "my_image.png")`).
    - You must spell out this argument since `img` passes your input to an HTML tag, and `src` is what the tag expects.
* You can also include other HTML friendly parameters such as height and width.
    - Example: Change height and width of an image (in pixels)
    ```
    img(src = 'my_image.png', height = 72, width = 72)
    ```
* The `img` function looks for your image file in a specific place. Your file must be in a folder named `www` in the same directory as the `app.R` script.
* Example of placing an image in an app:
```
ui <- fluidPage(
  titlePanel("My Shiny App"),
  sidebarLayout(
    sidebarPanel(),
    mainPanel(
      img(src = "rstudio.png", height = 140, width = 400)
    )
  )
)
```

#### Lesson 3: Add Control Widgets
**Add Control Widgets**
* A widget is a web element that users can interact with.
* Widgets provide a way for your users to send messages to the Shiny app.
* Shiny widgets collect a value from your user. When a user changes the widget, the value will change as well (thanks to reactivity).
* Shiny comes with a family of pre-build widgets, each created with a transparently named R function.
* The standard Shiny widgets are:
    - `actionButton`: Action button
    - `checkboxGroupInput`: A group of check boxes
    - `checkboxInput`: A single check box
    - `dateInput`: A calendar to aid date selection
    - `dateRangeInput`: A pair of calendars for selecting a date range
    - `fileInput`: A file upload control wizard
    - `helpText`: Help text that can be added to an input form
    - `numericInput`: A field to enter numbers
    - `radioButtons`: A set of radio buttons
    - `selectInput`: A box with choices to select from
    - `sliderInput`: A slider bar
    - `submitButton`: A submit button
    - `textInput`: A field to enter text

**Adding Widgets**
* You can add widgets to your web page in the same way that you add other types of HTML content in the previous section.
* To add a widget to your app, place a widget function in `sidebarPanel` or `mainPanel` in your `ui` object.
* Each widget function requires several arguments.
        - The first two arguments are:
            1. A name for the widget: The user will not see this name, but you can use it to access the widget's value.
                - The name should be a character string.
            2. A label: This label will appear with the widget in your app.
                - It should be a character string, and can be an empty string `""` if needed.
        - Example: `actionButton('action', label = 'Action')`
        - You can find all arguments for a widget in the widget's help page, which can be accessed via `?selectInput`
* Example of all widgets listed above:
```
library(shiny)

# Define UI ----
ui <- fluidPage(
  titlePanel("Basic widgets"),

  fluidRow(

    column(3,
           h3("Buttons"),
           actionButton("action", "Action"),
           br(),
           br(),
           submitButton("Submit")),

    column(3,
           h3("Single checkbox"),
           checkboxInput("checkbox", "Choice A", value = TRUE)),

    column(3,
           checkboxGroupInput("checkGroup",
                              h3("Checkbox group"),
                              choices = list("Choice 1" = 1,
                                             "Choice 2" = 2,
                                             "Choice 3" = 3),
                              selected = 1)),

    column(3,
           dateInput("date",
                     h3("Date input"),
                     value = "2014-01-01"))   
  ),

  fluidRow(

    column(3,
           dateRangeInput("dates", h3("Date range"))),

    column(3,
           fileInput("file", h3("File input"))),

    column(3,
           h3("Help text"),
           helpText("Note: help text isn't a true widget,",
                    "but it provides an easy way to add text to",
                    "accompany other widgets.")),

    column(3,
           numericInput("num",
                        h3("Numeric input"),
                        value = 1))   
  ),

  fluidRow(

    column(3,
           radioButtons("radio", h3("Radio buttons"),
                        choices = list("Choice 1" = 1, "Choice 2" = 2,
                                       "Choice 3" = 3),selected = 1)),

    column(3,
           selectInput("select", h3("Select box"),
                       choices = list("Choice 1" = 1, "Choice 2" = 2,
                                      "Choice 3" = 3), selected = 1)),

    column(3,
           sliderInput("slider1", h3("Sliders"),
                       min = 0, max = 100, value = 50),
           sliderInput("slider2", "",
                       min = 0, max = 100, value = c(25, 75))
    ),

    column(3,
           textInput("text", h3("Text input"),
                     value = "Enter text..."))   
  )

)

# Define server logic ----
server <- function(input, output) {

}

# Run the app ----
shinyApp(ui = ui, server = server)
```

#### Lesson 4: Display Reactive Output
**Display Reactive Output**
* To add live functionality to your input widgets we will now cover reactive outputs.
* Reactive outputs automatically respond when your user toggles a widget.

**Two Steps**
* You can create reactive outputs with a two-step process:
    1. Add an R object to your user interface.
    2. Tell Shiny how to build the object in the server function. The object will be reactive if the code that builds it calls a widget value.

* Step 1: Add an R object to the UI
    - Shiny provides a family of functions that turn R objects into output for your user interface.
    - Each function creates a specific type of output.
    - Types of functions:
        - `dataTableOutput`: Creates a datatable
        - `htmlOutput`: Creates raw HTML
        - `imageOutput`: Create an image
        - `plotOutput`: Creates a plot
        - `tableOutput`: Creates a table
        - `textOutput`: Creates text
        - `uiOutput`: Creates raw HTML
        - `verbatimTextOutput`: Creates text
    - You can add output to the user interface in the same way that you added HTML elements and widgets.
        - Place the output function inside `sidebarPane` or `mainPanel` in the `ui`.
    - For example, the `ui` object below uses `textOutput` to add a reactive lien of text to the main panel of the Shiny app.
    ```
    ui <- fluidPage(
        titlePanel('censusVis'),

        sidebarLayout(
            sidebarPanel(
                helpText('Create demographic maps with
                          information from the 2010 US Census'),

                selectInput('var',
                            label = 'Choose a variable to display',
                            choices = c('Percent White',
                                        'Percent Black',
                                        'Percent Hispanic',
                                        'Percent Asian'),
                            selected = 'Percent White'),

                sliderInput('range',
                            label = 'Range of interest:',
                            min = 0, max = 100, value = c(0, 100))
                ),

        mainPanel(
            textOutput('selected_var')
            )
        )
    )
    ```
        - Note that `textOutput` takes an argument, the character string `'selected_var'`.
        - Each of the `*Output` functions require a single argument: a character string that Shiny will use as the name of your reactive element. Your user will not see this name, but you will use it later.
* Step 2: Provide R code to build the object
    - Placing a function in `ui` tells Shiny where to display your object. Next, you need to tell Shiny how to build the object, which we do by providing the R code that builds the object in the `server` function.
    - The `server` function plays a special role in the Shiny process; it builds a list-like object named `output` that contains all of the code needed to update the R objects in your app.
        - Each R object needs to have its own entry in the list.
        - You can create an entry by defining a new element for `output` within the `server` function. The element name should match the name of the reactive element that you created in the `ui`.
    - Example:
    ```
    server <- function(input, output) {
        output$selected_var <- renderText({
            'You have selected this'
            })
    }
    ```
        - Notice that `output$selected_var` matches `textOutput('selected_var')` in the `ui`.
    - You do not need to explicitly state in the `server` function to return `output` in its last line of code. R will automatically update `output` through reference class semantics.
    - Each entry to `output` should contain the output of one of Shiny's `render*` function that corresponds to the type of reactive object you are making.
    - Types of Render functions:
        - `renderDataTable`: Creates a Data table
        - `renderImage`: Creates images (saved as a link to the source file)
        - `renderPlot`: Creates plots
        - `renderPrint`: Creates any printed output
        - `renderTable`: Creates data frame, matrix, other table-like structures.
        - `renderText`: Creates a character strings
        - `renderUI`: Creates a Shiny tab object or HTML
    - Each `render*` function takes a single argument: an R expression surrounded by braces `{}`. The expression can be one simple line of text, or it can involve many lines of code, as if it were a complicated function call.
    - Think of R expressions as a set of instructions that you give Shiny to store for later. Shiny will run the instructions when you first launch your app, and then Shiny will re-run the instructions every time it needs to update the object.
        - For this to work, your expression should return the object you have in mind (a piece of text, a plot, a data frame, etc.) You will get an error if the expression does not return an object, or if it returns the wrong type of object.

#### Lesson 5: Use R Scripts and Data
