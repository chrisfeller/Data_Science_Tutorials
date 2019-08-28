### Mastering Shiny
#### August 2019

---
#### Welcome
**Link**
* [Here](https://mastering-shiny.org/index.html)

#### Chapter 1: Introduction
**What is Shiny**
* A framework for creating web applications using R code designed primarily for data scientists.

**Shiny**
* To download the Shiny R package:
    ```
    install.packages('shiny')
    ```
* A few other libraries necessary for the following textbook:
    ```
    install.packages(c('magrittr', 'lubridate', 'readr', 'dplyr', 'ggplot2', 'gt'))
    ```

#### Chapter 2: Your First Shiny App
**Introduction**
* There are two key components of a Shiny App:
    1. User Interface (UI): Defines how your app **looks**
    2. Server Function: Defines how your app **works**

**Loading Shiny**
* To load shiny:
    ```
    library(shiny)
    ```

**Create App Directory and File**
* All shiny app projects should start with a directory containing an `app.R` script. This script will be used to tell Shiny both how your app should look, and how it should behave.
* A beginner script might look something like this:
    ```
    library(shiny)

    ui <- 'Hello, world!'

    server <- function(input, output, session) {}

    shinyApp(ui, server)
    ```
* The above example accomplishes four things:
    1. Calls `library(shiny)` to load the shiny package
    2. Defines the user interface for the app, which in this case it's just the text 'Hello, world!'
    3. Defines the server, which will determine the behavior of the app. For not it is an empty placeholder.
    4. Runs `shinyApp(ui, server)` to construct the shiny app object

**Running and Stopping a Shiny App**
* To run a shiny app in RStudio hit `Run App` in the gui or `Ctrl+Shift+Enter`
* To stop a shiny app in RStudio hit the stop sign icon or `esc`

**Adding UI Controls**
* UI example:
    ```
    ui <- fluidPage(
        selectInput('dataset', label = 'Dataset',
            choices = ls('package:datasets')
            ),
        verbatimTextOutput('summary'),
        tableOutput('table')
        )
    ```
* The example above uses four important functions:
    1. `fluidPage()` is a *layout function* which set up the basic visual structure of the page.
    2. `selectInput()` is an *input control* which lets the user interact with the app by providing a value.
    3. `verbatimTextOutput()` and `tableOutput()` are *output controls* that tell Shiny where to put rendered output.
        - `verbatimTextOutput()` displays code
        - `tableOutput()` displays table

**Adding Behavior**
* Shiny uses a reactive programming framework to make apps interactive. This means we have to tell Shiny *how* to perform a computation.
* As an example, we'll now tell Shiny how to fill the `summary` and `table` outputs.
    ```
    server <- function(input, output, session) {
        output$summary <- renderPrint({
            dataset <- get(input$dataset, 'package:datasets', inherits = FALSE)
            summary(dataset)
            })

        output$table <- renderTable({
            dataset <- get(input$dataset, 'package:datasets', inherits = FALSE)
            dataset
            })    
    }
    ```
* Almost every output you'll write in Shiny will follow this same pattern:
    ```
    outout$IDENTIFIER <- renderTYPE({
        # Expression that generates whatever kind of output
        # renderTYPE expects
        })
    ```
    - `output$IDENTIFIER` indicates that you're providing the recipe for the Shiny output with the matching ID.
    - `renderType()` specifies a specific *render function* to wrap a code block that you provide. Each render function is designed to work with a particular type of output that is passed to an output function.
* Output are reactive which means that Shiny automatically knows when to recalculate them. Anytime a user changes one of these the app will automatically update.

**Reducing Duplication w/ Reactive Expressions**
* To reduce duplicate code we use a mechanism called *reactive expressions* which are blocks of code wrapped in `reactive({...})`. You assign this reactive expression object o a variable and then call the variable like a function to retrieve it's value.
* To update the above example to remove the duplicate `get(input$dataset, 'package:datasets', inherits = FALSE)` line:
    ```
    server <- function(input, output, session) {
        dataset <- reactive({
            get(input$dataset, 'package:datasets', inherits = FALSE)
            })

        output$summary <- renderPrint({
            summary(dataset())
            })

        output$table <- renderTable({
            dataset()
            })
    }
    ```

**Exercise #1**
* Suppose your friend wants to design an app that allows the user to set a number (x) between 1 and 50, and displays the result of multiplying this number by 5. Their first attempt has a functional slider, but yields the following error: `Error: object 'x' not found`
    ```
    ui <- fluidPage(
      sliderInput(inputId = "x",
                  label = "If x is",
                  min = 1, max = 50,
                  value = 30),
      "then, x multiplied by 5 is",
      textOutput("product")
    )

    server <- function(input, output, session) {
      output$product <- renderText({
        x * 5
      })
    }
    ```
    - Answer: x should be referenced via `input$x` in the server function.

**Exercise #2**
* Extend the app from the previous exercise to allow the user to set the value of the multiplier, y, so that the app yields the value of x * y.
    ```
    ui <- fluidPage(
      sliderInput(inputId = "x",
                  label = "If x is",
                  min = 1, max = 50,
                  value = 30),
      sliderInput(inputId = 'y',
                  label = 'If y is',
                  min = 1, max = 50,
                  value = 5),
      "then, x multiplied by y is",
      textOutput("product")
    )

    server <- function(input, output, session) {
      output$product <- renderText({
        input$x * input$y
      })
    }
    ```

**Exercise #3**
* Replace the UI and server components of your app from the previous exercise with the UI and server components below, run the app, and describe the app’s functionality. Then, enhance the code with the use of reactive expressions to avoid redundancy.
    ```
    ui <- fluidPage(
      sliderInput(inputId = "x", label = "If x is",
                  min = 1, max = 50, value = 30),
      sliderInput(inputId = "y", label = "and y is",
                  min = 1, max = 50, value = 5),
      "then, (x * y) is", textOutput("product"),
      "and, (x * y) + 5 is", textOutput("product_plus5"),
      "and (x * y) + 10 is", textOutput("product_plus10")
    )

    server <- function(input, output, session) {
      output$product <- renderText({
        product <- input$x * input$y
        product
      })
      output$product_plus5 <- renderText({
        product <- input$x * input$y
        product + 5
      })
      output$product_plus10 <- renderText({
        product <- input$x * input$y
        product + 10
      })
    }
    ```
    - Answer:
        ```
        ui <- fluidPage(
          sliderInput(inputId = "x", label = "If x is",
                      min = 1, max = 50, value = 30),
          sliderInput(inputId = "y", label = "and y is",
                      min = 1, max = 50, value = 5),
          "then, (x * y) is", textOutput("product"),
          "and, (x * y) + 5 is", textOutput("product_plus5"),
          "and (x * y) + 10 is", textOutput("product_plus10")
        )

        server <- function(input, output, session) {
          x <- reactive({input$x})

          y <- reactive({input$y})

          output$product <- renderText({
            product <- x() * y()
            product
          })
          output$product_plus5 <- renderText({
            product <- x() * y()
            product + 5
          })
          output$product_plus10 <- renderText({
            product <- x() * y()
            product + 10
          })
        }
        ```

**Exercise #4 Spot and Correct the Bug**
```
library(shiny)
library(HistData)

ui <- fluidPage(
  selectInput(inputId = "dataset", label = "Dataset", choices = ls("package:HistData")),
  verbatimTextOutput("summary"),
  plotOutput("plot")
)

server <- function(input, output, session) {
  dataset <- reactive({
    get(input$dataset, "package:HistData", inherits = FALSE)
  })
  output$summary <- renderPrint({
    summary(dataset())
  })
  output$plot <- renderPlot({
    plot(dataset())
  })
}

shinyApp(ui, server)
```
- Answer: I have no idea everything looks and run fine.

#### Chapter 3: Basic UI
**Introduction**
* Shiny encourages separation of the code that generates the user interface, or frontend, from the code that drives the app's behavior, or backend.
* The UI controls the front end while the server controls the behavior.

**Inputs**
* App's intake inputs via functions like `sliderInput()`, `selectInput()`, `textInput()`, and `numericInput()`

**Common Structure**
* All input functions have the same first argument: `inputId`.
    - This is the identifier used to connect the frontend with the backend
    - Example: If your UI specification creates an input with ID `name`, you'll access it in the server function via `input$name`
* The `inputId` has two constraints:
    1. It must be a simple string that contains only letters, numbers, and underscores (no spaces, dashes, periods, or other special characters allowed). Name it like you name variables in R.
    2. It must be unique. If it's not unique, you'll have no way to refer to this control in your server function.
* Most input functions have a second parameter called `label`.
    - This is used to create a human-readable label for the control.
    - There is no restriction on this string
* The third parameter is typically `value`, which, where possible, lets you set the default value.
* The remaining parameters are unique to the control.
* When creating an input, supply the `inputId` and `label` arguments by position, and all other arguments by name.
    - Example: `sliderInput('min', 'Limit (minimum)', value = 50, min = 0, max = 100)`

**Free Text**
* Example of how to collect small amounts of text with `textInput()`, passwords with `passwordInput()`, and paragraphs with `textAreaInput()`
    ```
    ui <- fluidPage(
        textInput('name', "What's your name?"),
        passwordInput('password', "What's your password?"),
        textAreaInput('story', 'Tell me about yourself', rows = 3, cols = 80)
        )
    ```

**Numeric Inputs**
* To collect numeric values, create a slider with `sliderInput()` or a constrained textbox with `numericInput()`.
* Example:
    ```
    ui <- fluidPage(
        numericInput('num', 'Number one', value = 0, min = 0, max = 100),
        sliderInput('num2', 'Number two', value = 50, min = 0, max = 100),
        sliderInput('rng', 'Range', value = c(10, 20), min = 0, max = 100)
        )
    ```
* Generally, you should only use sliders for small ranges, where the precise value is not so important.

**Dates**
* Collect a single date with `dateInput()`, or a range of two days with `dateRangeInput()`. These provide a convenient calendar picker, and additional arguments like `datesdisabled` and `daysofweekdisabled` allow you to restrict the set of valid inputs.
* Example:
    ```
    ui <- fluidPage(
        dateInput('dob', 'When were you born?'),
        dateRangeInput('holiday', 'When do you want to go on vacation next?')
        )
    ```
* The defaults for date format, language, and day on which the week starts adhere to how calendars are generally formatted in the United States.

**Limited Choices**
* There are two difference approaches to allow the user to choose from a prespecified set of options:
    1. `selectInput()`
    2. `radioButtons()`
* Example:
    ```
    animals <- c('dog', 'cat', 'mouse', 'bird', 'other')
    ui <- fluidPage(
        selectInput('state', "What's your favorite state?", state.name),
        radioButtons('animal', "What's your favorite animal?", animals)
        )
    ```
* Radio buttons have two nice features:
    1. They show all possible options, making them suitable for short lists
    2. They can display options other than plain text via `choiceNames`/`choiceValues`
    * Example:
        ```
        ui <- fluidPage(
            radioButtons('rb', 'Choose one:',
                choicesNames = list(
                    icon('angry'),
                    icon('smile'),
                    icon('sad-tear')
                    ),
                choiceValues = list('angry', 'happy', 'sad'))
            )
        ```
* Dropdowns created with `selectInput()` take up the same amount of space, regardless of the number of options, making them more suitable for longer options. You can also set `multiple = TRUE` to allow the user to select multiple elements from the list of possible values.
    * Example:
        ```
        ui <- fluidPage(
            selectInput(
                'state', "What's your favorite state?", state.name, multiple = TRUE
                )
            )
        ```
* There's no way to select multiple values with radio buttons, but there's an alternative that's conceptually similar: `checkboxInputGroup()`
    * Example:
        ```
        ui <- fluidPage(
            checkboxGroupInput('animal', 'What animals do you like?', animals)
            )
        ```
* To allow a user to select variables from a dataframe to be used in summary, plots, etc. you can use the `varSelectInput()`:
    * Example:
        ```
        ui <- fluidPage(
            varSelectInput('variable', 'Select a variable', mtcars)
            )
        ```
    * Note that the result of `varSelectInput()` is different from selecting a variable name from a character vector of names of variables.

**Yes/No Questions**
* If you want a single checkbox for a single yes/no question, use `checkboxInput()`
    * Example:
        ```
        ui <- fluidPage(
            checkboxInput('cleanup', 'Clean up?', value = TRUE),
            checkboxInput('shutdown', 'Shutdown?')
            )
        ```

**File Uploads and Action Buttons**
* To upload a file use `fileInput()`
* To create action buttons use `actionButton()`

**Exercise #1**
* When space is at a premium, it’s useful to label text boxes using a placeholder that appears inside the text entry area. How do you call `textInput()` to generate the UI below?
    - Answer:
    ```
    ui <- fluidPage(
      textInput('name', "What's your name?", value = 'Your name')
      )

    server <- function(input, output, session) {}
    ```

**Exercise #2**
* Carefully read the documentation for sliderInput() to figure out how to create a date slider, as shown below.
    - Answer:
    ```
    ui <- fluidPage(
      sliderInput('delivery',
                  'When should we deliver?',
                  value = as.Date('2019-08-10'),
                  min = as.Date('2019-08-09'),
                  max = as.Date('2019-08-16'))
    )

    server <- function(input, output, session) {}
    ```        

**Exercise #3**
* If you have a moderately long list, it’s useful to create sub-headings that break the list up into pieces. Read the documentation for `selectInput()` to figure out how. (Hint: the underlying HTML is called `<optgroup>`.)
    - Answer:
    ```
    ui <- fluidPage(
      selectInput('state',
                  'Where are you from?',
                  choices = list('East Coast' = list('NY', 'NJ', 'CT'),
                                 'West Coast' = list('WA', 'OR', 'CA'),
                                 'Midwest' = list('MN', 'WI', 'IA'))))

    server <- function(input, output, session) {}
    ```

**Exercise #4**
* Create a slider input to select values between 0 and 100 where the interval between each selectable value on the slider is 5. Then, add animation to the input widget so when the user presses play the input widget scrolls through automatically.
    - Answer:
    ```
    ui <- fluidPage(
        sliderInput('number',
                    'Select a Range',
                    value = c(10, 20),
                    min = 0,
                    max = 100,
                    step = 5,
                    animate = TRUE)
        )

    server <- function(input, output, session) {}
    ```

**Exercise #5**
* Using the following numeric input box the user can enter any value between 0 and 1000. What is the purpose of the step argument in this widget?
    ```
    numericInput("number", "Select a value", min = 0, max = 1000, step = 50)
    ```
    - Answer: Interval to use when stepping between min and mix

**Outputs**
* Output functions in the UI specification create placeholders that are filled by the server function.
* Like inputs, outputs take a unique ID as their first argument.
    - If your UI specification creates an output with ID `plot`, you'll access it in the server function with `output$plot`.
* Each `output` function on the frontend is coupled with a `render` function in the backend, like `output$plot <- renderPlot({...})`
* There are three main types of outputs, corresponding to the three things you usually include in a report:
    1. Text
    2. Tables
    3. Plots

**Text**
* Output regular text with `textOutput()` and code with `verbatimTextOutput()`
* Example:
    ```
    ui <- fluidPage(
        textOutput('text'),
        verbatimTextOutput('code')
        )

    server <- function(input, output, session {
        output$text <- renderText({
            'Hello friend!'
            })
        output$code <- renderPrint({
            summary(1:10)
            })
        })
    ```
* Note that there are two render functions that can be used with either of the text output functions:
    1. `renderText()`: displays text *returned* by the code.
    2. `renderPrint()`: displays text *printed* by the code.

**Tables**
* There are two options for displaying dataframes in tables:
    1. `tableOutput()` and `renderTable()` render a static table of data, showing all the data at once.
        - Most useful for small, fixed summaries
    2. `dataTableOutput()` and `renderDataTable()` render a dynamic table, where only a fixed number of rows are shown at once, and the user can interact to see more.
        - More appropriate if you want to expose a complete dataframe to a user.
* Example:
    ```
    ui <- fluidPage(
        tableOutput('static'),
        dataTableOutput('dynamic')
        )

    server <- function(input, output, session) {
        output$static <- renderTable({head(mtcars)})
        output$dynamic <- renderDataTable({mtcars}, options = list(pageLength = 5))
    }
    ```

**Plots**
* You can display any type of R graphic with `plotOutput()` and `renderPlot()`
* Example:
    ```
    ui <- fluidPage(
        plotOutput('plot', width = '400px')
        )

    server <- function(input, output, session) {
        output$plot <- renderPlot({plot(1:5)})
    }
    ```
* By default, `plotOutput()` will take up the full width of the element it's embedded within and will be 400 pixels high. You can override these defaults with the `height` and `width` arguments.
* Plots are special because they can also act as inputs. `plotOutput()` has a number of arguments like `click`, `dbclick`, and `hover`. If you pass these a string, like `click = 'plot_click'`, they'll create a reactive input (`input$plot_click`) that you can use to handle user interaction on the plot.

**Exercise #1**
* Re-create the Shiny app from the plots section, this time setting height to 300px and width to 700px.
    - Answer:
    ```
    ui <- fluidPage(
        plotOutput('plot', width = '700px', height = '300px')
        )

    server <- function(input, output, session) {
        output$plot <- renderPlot({plot(1:5)})
    }
    ```

**Exercise #2**
* Update the options for renderDataTable() below so that the table is displayed, but nothing else.
    - Answer:
    ```
    ui <- fluidPage(
        dataTableOutput('table')
        )

    server <- function(input, output, session) {
        output$table <- renderDataTable({mtcars}, options = list(dom = 't'))
    }
    ```

**Layouts**
* Layout functions allow you to arrange inputs and outputs on a page.
* Provide a high-level visual structure of an app.
* As we've seen in previous examples `fluidPage()` is the most common layout function.

**Layout Overview**
* Layouts are created by a hierarchy of function calls, where the hierarchy in R matches the hierarchy in the output.
* When you see complex code layout code like this:
```
fluidPage(
    titlePanel('Hello Shiny!'),
    sidebarLayout(
        sidebarPanel(
            sliderInput('obs', 'Observations:', min = 0, max = 1000, value = 500)
            ),
        mainPanel(
            plotOutput('distPlot')
            )
        )
    )
```
First skim it by focusing on the hierarchy of the function calls:
    1. `fluidPage()`
    2. `titlePanel()`
    3. `sidebarLayout()`
        - `sidebarPanel()`
            - `sliderInput()`
        - `mainPanel()`
            - `plotOutput()`
Even without knowing anything about the layout functions you can read the function names to guess what this app is going to look like.
    - A classic app design with a title bar at the top, followed by a sidebar (continuing a slider), and a main panel containing a plot.

**Page Functions**
* The page function sets up all the HTML, CSS, and JS that Shiny needs.
* The most important, but least interesting, layout function is `fluidPage()`
* `fluidPage()` uses a layout system balled Bootstrap that provides attractive defaults.
* Technically, `fluidPage()` is all you need for an app, because you can put inputs and outputs directly inside of it. While this is fine to learn the basics of Shiny, dumping all the inputs and outputs in one place doesn't look very good, so for more complicated apps, you need to learn more layout functions.

**Page with Sidebar**
* `sidebarLayout()` makes it easy to create a two-column layout with inputs on the left and outputs on the right.
* Example:
    ```
    fluidPage(
        headerPanel(
            # app title/description
            ),
        sidebarLayout(
            sidebarPanel(
                # inputs
                ),
            mainPanel(
                # outputs
                )
            )
    ```     
* Example: Uses the same layout to create a simple app demonstrating the Central Limit Theorem
    ```
    ui <- fluidPage(
        headerPanel('Central Limit Theorem'),
        sidebarLayout(
            sidebarPanel(
                numericInput('m', 'Number of samples:', 2, min = 1, max = 100)),
            mainPanel(
                plotOutput('hist')
                )
            )
        )

    server <- function(input, output, server) {
        output$hist <- renderPlot({
            means <- replicate(1e4, mean(runif(input$m)))
            hist(means, breaks = 20)
            })
    }
    ```

**Multi-Row**
* Under the hood, `sidebarLayout()` is built on top of a flexible multi-row layout, which you can use this directly to create more visually complex apps.
* As usual, you start with `fluidPage()`. Then you create rows with `fluidRow()`, and columns with `column()`
* The basic code structure looks like this:
```
fluidPage(
    fluidRow(
        column(4,
            ...
            ),
        column(8,
            ...
            )
        ),
    fluidRow(
        column(6,
            ...
            ),
        column(6,
            ...)
        )
    )
```
* Note that the first argument in `column()` is the width, and the width of each row must add up to 12. This give you substantial flexibility because you can easily create 2-, 3-, or 4- column layouts (more than that starts to get cramped), or use narrow columns to create spacers.

**Themes**
* Creating a complete theme from scratch is a lot of work, but you can get some easy wins using a shinythemes package. The following code shows four options:
```
theme_demo <- function(theme) {
    fluidPage(
        theme = shinythemes::shinytheme(theme),
        sidebarLayout(
            sidebarPanel(
                textInput('txt', 'TextInput:', 'text here'),
                sliderInput('slider', 'Slider input:', min = 1, max = 100, value = 30)
                ),
            mainPanel(
                h1('Header 1'),
                h2('Header 2'),
                p('Some text')
                )
            )
        )
}

theme_demo('darkly')
theme_demo('flatly')
theme_demo('sandstone')
theme_demo('united')
```
* To theme any app just pass `theme = shinythemes::shinytheme(theme)` to `fluidPage()`
* For a gallery of pre-curated themes visit [here](https://shiny.rstudio.com/gallery/shiny-theme-selector.html)

**Exercise #1**
* Update the Central Limit Theorem app presented in the chapter so that the sidebar is on the right instead of the left.
    - Answer:
    ```
    ui <- fluidPage(
      headerPanel('Central Limit Theorem'),
      sidebarLayout(
        mainPanel(
          plotOutput('hist')
        ),
        sidebarPanel(
          numericInput('m', 'Number of samples:', 2, min = 1, max = 100)),    
      )
    )

    server <- function(input, output, server) {
      output$hist <- renderPlot({
        means <- replicate(1e4, mean(runif(input$m)))
        hist(means, breaks = 20)
      })
    }

    shinyApp(ui, server)
    ```

**Exercise #2**
* Browse the themes available in the shinythemes package, and update the theme of the app from the previous exercise.
    - Answer:
    ```
    ui <- fluidPage(
      theme = shinythemes::shinytheme('spacelab'),
      headerPanel('Central Limit Theorem'),
      sidebarLayout(
        mainPanel(
          plotOutput('hist')
        ),
        sidebarPanel(
          numericInput('m', 'Number of samples:', 2, min = 1, max = 100)),    
      )
    )

    server <- function(input, output, server) {
      output$hist <- renderPlot({
        means <- replicate(1e4, mean(runif(input$m)))
        hist(means, breaks = 20)
      })
    }

    shinyApp(ui, server)
    ```

**Under the Hood**
* All input, output, and layout functions return HTML, the descriptive language that underpins every website.
* You can see that HTML by executing UI functions directly in the console via:
```
fluidPage(
    textInput('name', "What's your name?")
    )
```

#### Chapter 4: Basic Reactivity
**Introduction**
* In Shiny, you express your server logic using reactive programming.
* Reactive programming is an elegant and powerful programming paradigm, but it can be disorienting at first because it's a very different paradigm to writing a script.
* The key idea of reactive programming is to specify a graph of dependencies so that when an input changes, all outputs are automatically updated. This makes the flow of an app considerably simpler.

**The `server` Function**
* As you've seen, the guts of every Shiny app looks like this:
```
library(shiny)

ui <- fluidPage(
    # Front-end interface
    )

server <- function(input, output, session){
    # Back-end logic
}

shinyApp(ui, server)
```
* Instead of a single static object like `ui`, the backend is a function, `server()`
* You'll never call the function yourself; instead, Shiny invokes it whenever a new session begins.
    * A session captures the state of one live instance of a shiny app.
    * A session begins each time the Shiny app is loaded in a browser, either by different people, or by the same person opening multiple tabs.
    * The server function is called once for each session, creating a private scope that holds the unique state for that user, and every variable created inside the server function is only accessible to that session. This is why almost all of the reactive programming you'll do in Shiny will be inside the server function.
* Server functions take three parameters:
    1. `input`
    2. `output`
    3. `session`
    * Because you never call the server function yourself, you'll never create these objects yourself. Instead, they're created by Shiny when the session begins, connecting back to a specific session.

**Input**
* The `input` argument is a list-like object that contains all the input data sent from the browser, named according to the input ID.
* For example, if your UI contains a numeric input control with an input ID of `count`, like so:
```
ui <- fluidPage(
    numericInput('count', label = 'Number of values', value = 100)
    )
```
Then you can access the value of that input with `input$count`. It will initially contain the value `100`, and it will automatically update as the user changes the value in the browser.
* Unlike a typically list, `input` objects are read-only. If you attempt to modify an input inside the server function, like in the example below, you'll get an error:
```
server <- function(input, output, session){
    input$count <- 10 # can't do this!
}

shinyApp(ui, server)
```
* You'll also get an error if you try and read from an input while not in a reactive context created by a function like `renderText()` or `reactive()`. For example, you would get an error while trying to do the following:
```
server <- function(input, output, session) {
    message('The value of input$count is', input$count)
}

shinyApp(ui, server)
```

**Output**
* `output` is very similar to `input` as it's also a list-like object named according to the output ID.
* The main difference is that you use it for sending output not receiving input.
* You always use the `output` object in concert with a `render` function, as in the following example:
    ```
    ui <- fluidPage(
        textOutput('greeting')
        )

    server <- function(input, output, session){
        output$greeting <- renderText({'Hello Human!'})
    }
    ```
* Like the `input` object, `output` is picking about how you use it. If you forget the `render` function, shown below, you will get an error.
```
server <- function(input, output, session){
    output$greeting <- 'Hello human'
}

shinyApp(ui, server)
```
* You'll also get an error if you attempt to read from an output as demonstrated below:
```
server <- function(input, output, session){
    message('The greeting is ', output$greeting)
}

shinyApp(ui, server)
```
* The `render` function does two things:
    1. Sets up a reactive context that automatically tracks what inputs the outputs uses.
    2. Converts the output of your R code into HTML suitable for display on a web page.

**Reactive Programming**
* An app is going to be pretty boring if it only has inputs or only has outputs. The real magic of Shiny happens when you have an app with both.
* A simple example:
```
ui <- fluidPage(
    textInput('name', "What's your name?"),
    textOutput('greeting')
    )

server <- function(input, output, session) {
    output$greeting <- renderText({
        paste0('Hello ', input$name, "!")
        })
}
```
* In the above example, the output message updates automatically as you type.
* This is a big idea in Shiny: you don't need to specify when the output code is run because Shiny automatically figures it out for you.

**Imperative vs. Declarative Programming**
* There are two important styles of programming:
    1. Imperative: You issue a specific command and it's carried out immediately. This is the style of programming you're used to in your analysis scripts: you command R to load your data, transform it, visualize it, and save the results to disk.
    2. Declarative: You express higher-level goals or describe important constraints, and rely on someone else to decide how and/or when to translate that into action.
* Shiny uses declarative programming.
* With imperative code you say 'make me a sandwich'. With declarative code you say 'ensure there is a sandwich in the refrigerator whenever I look inside of it.'
* Imperative code is assertive, declarative code is passive-aggressive.

**Lazyness**
* One of the strengths of declarative programming in Shiny is that it allows apps to be extremely lazy.
* A Shiny app will only ever do the minimal amount of work needed to update the output controls that you can currently see.

**The Reactive Graph**
* Shiny's laziness has another important property. In most R code, you can understand the order of execution by reading the code from top to bottom. That doesn't work in Shiny, because code is only run when needed.
* To understand the order of execution you need to instead look at the reactive graph, which describes how inputs and outputs are connected.
* As your app gets more complicated, it's often useful to make a quick high-level sketch of the reactive graph to remind you how all the pieces fit together.

**Reactive Expressions**
* A reactive expression is a tool that reduces duplication in your reactive code by introducing additional nodes into the reactive graph.
* Example:
    ```
    server <- function(input, output, session) {
        text <- reactive(paste0("Hello ", input$name, "!"))
        output$greeting <- renderText(text())
    }
    ```

**Execution Order**
* The order in which reactive code is run is determined only by the reactive graph, not by its layout in the server function.

**Reactive Expressions**
* Reactive expressions are important for two reasons:
    1. They help create efficient apps by giving Shiny more information so that it can do less recomputation when inputs change.
    2. They make it easier for humans to understand the app by simplifying the reactive graph.
* Reactive expressions have a flavor of both inputs and outputs:
    - Like inputs, you can use the results of a reactive expression in an output.
    - Like outputs, reactive expressions depend on inputs and automatically know when they need updating.
* Because of this duality, some functions work with either reactive inputs or expressions, and some functions work with either reactive expressions or reactive outputs. We'll use **producers** to refer either reactive inputs or expressions, and **consumers** to refer to either reactive expressions or outputs.

**Example App**
* Motivation: Imagine I want to compare two simulated datasets with a plot and a hypothesis test. I’ve done a little experimentation and come up with the functions below: `histogram()` visualizes the two distributions with a histogram, and `t_test()` compares their means with with a t-test.
```
library(ggplot2)

histogram <- function(x1, x2, binwidth = 0.1, xlim = c(-3, 3)) {
  df <- data.frame(
    x = c(x1, x2),
    g = c(rep("x1", length(x1)), rep("x2", length(x2)))
  )

  ggplot(df, aes(x, fill = g)) +
    geom_histogram(binwidth = binwidth) +
    coord_cartesian(xlim = xlim)
}

t_test <- function(x1, x2) {
  test <- t.test(x1, x2)

  sprintf(
    "p value: %0.3f\n[%0.2f, %0.2f]",
    test$p.value, test$conf.int[1], test$conf.int[2]
  )
}

x1 <- rnorm(100, mean = 0, sd = 0.5)
x2 <- rnorm(200, mean = 0.15, sd = 0.9)

histogram(x1, x2)
t_test(x1, x2)
```
* You can move this same code into a Shiny app. Starting with the UI we'll use the following layout: The first row has three columns for input controls (distribution 1, distribution 2, and plot controls). The second row has a wide column for the plot, and a narrow column for the hypothesis test.
```
ui <- fluidPage(
  fluidRow(
    column(4,
      "Distribution 1",
      numericInput("n1", label = "n", value = 1000, min = 1),
      numericInput("mean1", label = "µ", value = 0, step = 0.1),
      numericInput("sd1", label = "σ", value = 0.5, min = 0.1, step = 0.1)
    ),
    column(4,
      "Distribution 2",
      numericInput("n2", label = "n", value = 1000, min = 1),
      numericInput("mean2", label = "µ", value = 0, step = 0.1),
      numericInput("sd2", label = "σ", value = 0.5, min = 0.1, step = 0.1)
    ),
    column(4,
      "Histogram",
      numericInput("binwidth", label = "Bin width", value = 0.1, step = 0.1),
      sliderInput("range", label = "range", value = c(-3, 3), min = -5, max = 5)
    )
  ),
  fluidRow(
    column(9, plotOutput("hist")),
    column(3, verbatimTextOutput("ttest"))
  )
)
```
* The server then calls the `histogram()` and `t_test()` functions:
```
server <- function(input, output, session) {
  output$hist <- renderPlot({
    x1 <- rnorm(input$n1, input$mean1, input$sd1)
    x2 <- rnorm(input$n2, input$mean2, input$sd2)

    histogram(x1, x2, binwidth = input$binwidth, xlim = input$range)
  })

  output$ttest <- renderText({
    x1 <- rnorm(input$n1, input$mean1, input$sd1)
    x2 <- rnorm(input$n2, input$mean2, input$sd2)

    t_test(x1, x2)
  })
}
```
* We can improve upon the server function code above by creating reactive expressions for `x1` and `x2` to avoid repeating code.
```
server <- function(input, output, session) {
  x1 <- reactive(rnorm(input$n_1, input$mean_1, input$sd_1))
  x2 <- reactive(rnorm(input$n_2, input$mean_2, input$sd_2))

  output$hist <- renderPlot({
    histogram(x1(), x2(), binwidth = input$binwidth, xlim = input$range)
  })

  output$ttest <- renderText({
    t_test(x1(), x2())
  })
}
```

**Controlling Timing of Evaluation**
* To create another example app that involves a simulation that updates on a regular cadence we can take advantage of `reactiveTimer()`
* In the following example, the app updates every 500 ms with a new simulated distribution.
```
ui <- fluidPage(
  fluidRow(
    column(3,
      numericInput("lambda1", label = "lambda1", value = 1),
      numericInput("lambda2", label = "lambda1", value = 1),
      numericInput("n", label = "n", value = 1e4, min = 0)
    ),
    column(9, plotOutput("hist"))
  )
)

server <- function(input, output, session) {
  timer <- reactiveTimer(500)

  x1 <- reactive({
    timer()
    rpois(input$n, input$lambda1)
  })
  x2 <- reactive({
    timer()
    rpois(input$n, input$lambda2)
  })

  output$hist <- renderPlot({
    histogram(x1(), x2(), binwidth = 1, xlim = c(0, 40))
  })
}
```

**On Click**
* You can also update an app on a user click via `actionButton()`
* In the following example, we again simulate a distribution but this time we wait until the user clicks the button `Simulate!`
* To do so we need to introduce a new tool called `eventReactive()`, which waits for an event such as a button click before starting the chain of events.
```
ui <- fluidPage(
  fluidRow(
    column(3,
      numericInput("lambda1", label = "lambda1", value = 1),
      numericInput("lambda2", label = "lambda1", value = 1),
      numericInput("n", label = "n", value = 1e4, min = 0),
      actionButton("simulate", "Simulate!")
    ),
    column(9, plotOutput("hist"))
  )
)

server <- function(input, output, session) {
  x1 <- eventReactive(input$simulate, {
    rpois(input$n, input$lambda1)
  })
  x2 <- eventReactive(input$simulate, {
    rpois(input$n, input$lambda2)
  })

  output$hist <- renderPlot({
    histogram(x1(), x2(), binwidth = 1, xlim = c(0, 40))
  })
}
```

**Observers**
* So far, we’ve focussed on what’s happening inside the app. But sometimes you need to reach outside of the app and cause side-effects to happen elsewhere in the world. This might be saving a file to a shared network drive, sending data to a web API, updating a database, or (most commonly) printing a debugging message to the console. These actions don’t affect how your app looks, so you can’t use an output and a render function. Instead you need to use an observer.
* There are multiple ways to create an observer but the easiest is via `observeEvent()`.
* `observeEvent()` is similar to `eventReactive()`
* It has two important arguments `eventExpr` and `handlerExpr`
    - The first argument is the input or expression to take a dependency on; the second argument is the code that will run.
* For example, the following modification to server() means that every time that text is updated, a message will be sent to the console:
```
server <- function(input, output, session) {
  text <- reactive(paste0("Hello ", input$name, "!"))

  output$greeting <- renderText(text())
  observeEvent(input$name, {
    message("Greeting performed")
  })
}
```
* There are two important difference between `observeEvent()` and `eventReactive()`:
    1. You don't assign the result of `observeEvent()` to a variable
    2. Thus, you can't refer to it from other reactive consumers.

#### Chapter 5: Case Study: Emergency Room Injuries    
**Introduction**
* As a short example of a real-life Shiny App, we'll build an app that showcases some data analysis.
* Libraries we'll use:
```
library(shiny)
library(vroom) # fast file reading
library(tidyverse)


```

**The Data**
* We'll use data from the National Electronic Injury Surveillance System (NEISS) that's collected by the Consumer Product Safety Commission.
* The data includes 250,000 observations and 10 variables on injuries sustained in the US.
* There are two additional dataframes we will join in, which have product and population information.
* Load Data:
```
# Set Working Directory
setwd('/USers/chrisfeller/Desktop/Mastering_Shiny/')

injuries <- vroom::vroom('data/injuries.tsv.gz')
products <- vroom::vroom('data/products.tsv')
population <- vroom::vroom('data/population.tsv')
```

**Exploration**
* Before we create the app, lets explore the data a little. We'll start by looking at the product associated with the most injuries: 1842 'stairs or steps'. First we'll pull out the injuries associated with this product.
```
selected <- injuries %>% filter(prod_code == 1842)
nrow(selected)
```
* Next, let's look at some summaries of the diagnosis, body part, and location where the injury occurred. Note that we are weighting by the  `weight` variable so that the counts can be interpreted as estimated total injuries across the whole US.
```
selected %>% count(diag, wt = weight, sort = TRUE)

selected %>% count(body_part, wt = weight, sort = TRUE)

selected %>% count(location, wt = weight, sort = TRUE)
```
* Next, we'll make a plot to explore patterns across age and sex.
```
summary <- selected %>%
            count(age, sex, wt = weight)

summary %>%
    ggplot(aes(age, n, colour = sex)) +
    geom_line() +
    labs(y = 'Estimated number of injuries')
```
* Next, we'll recreate the same plot but normalizing for population size.
```
summary <- selected %>%
    count(age, sex, wt = weight) %>%
    left_join(population, by = c('age', 'sex')) %>%
    mutate(rate = n / population * 1e4)

summary %>%
    ggplot(aes(age, rate, colour = sex)) +
    geom_line(na.rm = TRUE) +
    labs(y = 'Injuries per 10,000 people')
```
* Lastly, we can look at some of the narratives of injuries.
```
selected %>%
    sample_n(10) %>%
    pull(narrative)
```
* Having done this exploration for one product, it would be very nice if we could easily do it for other products, without having to retype the code. A Shiny App is perfect for this.

**Prototype**
* When building a complex app, start as simple as possible, so you can confirm the basic mechanics work before you start doing something more complicated.
* We will start with one input (the product code), three tables, and one plot.
* It may be helpful to do a few pencil-and-paper sketches to explore the UI and reactive graph before committing to code.
* We'll start with one row for the inputs, one row for all three tables (giving each table 4 columns, 1/3 of the 12 column width, and then one row for the plot).
```
ui <- fluidPage(
    fluidRow(
        column(6,
            selectInput('code', 'Product', setNames(products$prod_code, products$title))
            )
        ),
    fluidRow(
        column(4, tableOutput('diag')),
        column(4, tableOutput('body_part')),
        column(4, tableOutput('location'))
        ),
    fluidRow(
        column(12, plotOutput('age_sex'))
        )
    )
```
* For the server function, we'll convert the `selected` and `summary` variables to reactive expressions.
    - This is a reasonably general pattern: you typically create create variables in your data analysis as a way of decomposing the analysis into steps, and avoiding having to recompute things multiple times, and reactive expressions play the same role in Shiny apps.
    - Often it's a good idea to spend a little time cleaning up your analysis code before you start your Shiny app, so you can think about these problems in regular R code, before you add the additional complexity of reactivity.
```
server <- function(input, output, session) {
      selected <- reactive(injuries %>% filter(prod_code == input$code))

      output$diag <- renderTable(
        selected() %>% count(diag, wt = weight, sort = TRUE)
      )
      output$body_part <- renderTable(
        selected() %>% count(body_part, wt = weight, sort = TRUE)
      )
      output$location <- renderTable(
        selected() %>% count(location, wt = weight, sort = TRUE)
      )

      summary <- reactive({
        selected() %>%
          count(age, sex, wt = weight) %>%
          left_join(population, by = c("age", "sex")) %>%
          mutate(rate = n / population * 1e4)
      })

      output$age_sex <- renderPlot({
        summary() %>%
          ggplot(aes(age, n, colour = sex)) +
          geom_line() +
          labs(y = "Estimated number of injuries") +
          theme_grey(15)
      })
    }
```

**Rate vs. Count**
* So far, we're displaying only a single plot, but we'd like to give the user the choice between visualizing the number of injuries or the population-standardize rate. To do so we add a `selectInput()` to the UI to make that decision.
```
fluidRow(
    column(4,
        selectInput('code', 'Product', setNames(products$prod_code, products$title))
        ),
    column(2, selectInput('y', 'Y axis', c('rate', 'count')))
    ),
```
* The change in the server will look like the following conditional statement:
```
output$age_sex <- renderPlot({
    if (input$y == "count") {
      summary() %>%
        ggplot(aes(age, n, colour = sex)) +
        geom_line() +
        labs(y = "Estimated number of injuries") +
        theme_grey(15)
    } else {
      summary() %>%
        ggplot(aes(age, rate, colour = sex)) +
        geom_line(na.rm = TRUE) +
        labs(y = "Injuries per 10,000 people") +
        theme_grey(15)
    }
  })
 ```

**Narrative**
* Lastly, we want to provide some way to access the narratives.
* There are two parts to this solution. First we add a new row to the bottom of the UI. We'll use an action button to trigger a new story, and put the narrative in a `textOutput()`
```
fluidRow(
   column(2, actionButton("story", "Tell me a story")),
   column(10, textOutput("narrative"))
 )
```
* This action button is an integer that increments each time it's clicked, which will trigger a re-execution of the random selection. /
```
output$narrative <- renderText({
    input$story
    selected() %>% pull(narrative) %>% sample(1)
  })
```

#### Chapter 6: Advanced UI
**Introduction**
* The native languages of the web at HTML (for content), CSS (for styling), and JavaScript (for behavior). Shiny is designed to be accessible for R users who aren't familiar with any of those languages.
* However, you can still take advantage of these languages to customize your apps and extend the Shiny framework.

**HTML 101**
* HTML is a markup language for describing web pages.
* A markup language is just a document format that contains plain text content, plus embedded instructions for annotating, or 'marking up', specific sections of that content.
* These instructions can control the appearance, layout, and behavior of the text they mark up, and also provide structure to the document.
* A simple snippet of HTML:
```
This time I <em>really</em> mean it!
```
    - The `<em>` and `</em>` markup instructions indicate that the word `really` should be displayed with specific emphasis (italics).
    - `<em>` is an example of a *start* tag, and `<em/>` is an example of an *end* tag.

**Inline Formatting Tags**
* `em` is just one of many HTML tags that are used to format text.
* Other tags:
    - `<strong>...</strong>` makes bold text
    - `<u>...</u>` makes text underlined
    - `<s>...</s>` makes text strikeout
    - `<code>...</code>` makes text monospaced

**Block Tags**
* Another class of tags is used to wrap entire blocks of text.
* You can use `<p>...</p>` to break text into distinct paragraphs, or `<h3>...</h3>` to turn a line into a subheading.
* Example:
    ```
    <h3>Chapter I. Down the Rabbit-Hole</h3>

    <p>Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, ‘and what is the use of a book,’ thought Alice ‘without pictures or conversations?’</p>

    <p>So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.</p>
    ```

**Tags with Attributes**
* Some tags need to do more than just demarcate some text. An `<a>` (for 'anchor') tag is used to create a hyperlink.
    - Although it's not enough to just wrap `<a>...</a>` around the link's text, as you also need to specify where the hyperlink points to.
* Start tags let you include *attributes* that customize the appearance or behavior of the tag. In the case above, we'll add an `href` attribute to our `<a>` start tag:
    ```
    <p>Learn more about <strong>Shiny</strong> at <a href='https://shiny.rstudio.com'> this website</a>.</p>
    ```
* There are dozens of attributes that all tags accept and hundreds of attributes that are specific to particular tags. However, there are two attributes that are used constantly:
    1. The `id` attributes uniquely identifies a tag in a document. That is, no two tags in a single document should share the same `id` value, and each tag can have zero or one `id` value. As far as a web browser is concerned, the `id` attribute is completely optional and has no intrinsic effect on the appearance or behavior of the rendered tag. However, it's incredibly useful for identifying a tag for special treatment by CSS or JavaScript, and as such, plays a crucial role for Shiny apps.
    2. The `class` attribute provides a way of classifying tags in a document. Unlike `id`, any number of tags can have the same class, and each tag can have multiple classes (space separated). Again, classes don't  have an intrinsic effect, but are hugely helpful for using CSS or JavaScript to target groups of tags.
* Example:
    ```
    <p id='storage-low-message' class='message warning'>Storage space is running low!</p>
    ```
    - Here, the `id` and `class` values have had no discernible effect. But we could, for example, write CSS that any elements with the `message` class should appear at the top of the page, and that any elements with the `warning` class should have a yellow background and bold text; and we could write JavaScript that automatically dismisses the message if the storage situation improves.

**Parents and Children**
* In the example above, we have a `<p>` tag that contains some text that contains `<strong>` and `<a>` tags. We can refer to `<p>` as the parent of `<strong>`/`<a>`, and `<strong>`/`<a>` as the children of `<p>`. And naturally, `<strong>` and `<a>` are called siblings.
* It's often helpful to think of tags and text as forming a tree structure:
```
<p>
├── "Learn more about"
├── <strong>
│   └── "Shiny"
├── "at"
├── <a href="...">
│   └── "this website"
└── "."
```

**Comments**
* Just as you can use the `#` character to comment out a line of R code, HTML lets you comment out parts of your web page. Use `<!--` to start a comment and `-->` to end one. Anything between these delimiters will be ignored during the rendering of the web page, although it will still be visible to anyone who looks at your raw HTML by using your browser's View Source command.
```
<p>This HTML will be seen.</p>

<!-- <p>This HTML will not.</p> -->

<!--
<p>
Nor will this.
</p>
-->
```

**Escaping**
* Any markup language like HTML, where there are characters that have special meaning, needs to provide a way to “escape” those special characters–that is, to insert a special character into the document without invoking its special meaning.
* For example, the `<` character in HTML has a special meaning, as it indicates the start of a tag. What if you actually want to insert a `<` character into the rendered document–or, let’s say, an entire `<p>` tag?
* The escaped version of `<` is `&lt;`, and > is `&gt;`
    - Example:
        ```
        <p>In HTML, you start paragraphs with "&lt;p&gt;" and end them with "&lt;/p&gt;".</p>
        ```
* Each escaped character in HTML starts with `&` and ends with `;`.
* There are lots of valid sequences of characters that go between, but besides `lt` (less than) and `gt` (greater than), the only one you’re likely to need to know is `amp`; `&amp;` is how you insert a & character into HTML.
* Escaping `<`, `>`, and & is mandatory if you don’t want them interpreted as special characters; other characters can be expressed as escape sequences, but it’s generally not necessary.
* Escaping `<`, `>`, and & is so common and crucial that every web framework contains a function for doing it (in our case it’s `htmltools::htmlEscape`), but as we’ll see in a moment, Shiny will usually do this for you automatically.

**Generating HTML with Tag Objects**
* To write HTML in R we will use the `htmltools` package.
* In htmltools, we create the same trees of parent tags and child tags/text as in raw HTML, but we do so using R function calls instead of angle brackets. For example, this HTML from an earlier example:
```
<p id="storage-low-message" class="message warning">Storage space is running low!</p>
```
Would look like this in R:
```
library(htmltools)

p(id='storage-low-message', class='message warning', 'Storage space is running low!')
```
* When included in Shiny UI, it's HTML becomes part of the user interface.
* What are the main differences:
    - The `<p>` tag has become `p()` function call, and the end tag is gone. Instead, the end of the `<p>` tag is indicated by the function call's closing parenthesis.
    - The `id` and `class` attributes have become named arguments to `p()`.
    - The text contained within `<p>...</p>` has become a string that is passed as an unnamed argument to `p()`.

**Using Function to Create Tags**
* Only the most common HTML tags have a function directly exposed in the htmltools namespace: `<p>`, `<h1>` through `<h6>`, `<a>`, `<br>`, `<div>`, `<span>`, `<pre>`, `<code>`, `<img>`, `<strong>`, `<em>`, and `<hr>`.
* When writing these tags, you can simply use the tag name as the function name, e.g. `div()` or `pre()`
* To write all other tags, prefix the tag name with `tags$`. For example, to create a `<ul>` tag, there’s no dedicated `ul()` function, but you can call `tags$ul()`.
    - The `tags` object is a named list that `htmltools` provides, and it comes preloaded with almost all of the valid tags in the HTML5 standard.
* When writing a lot of HTML from R, you may find it tiresome to keep writing `tags$`. If so, you can use the `withTags` function to wrap an R expression, wherein you can omit the `tags$` prefix. In the following code, we call `ul()` and `li()`, whereas these would normally be `tags$ul()` and `tags$li()`.
```
withTags(
  ul(
    li("Item one"),
    li("Item two")
  )
)
```
* Finally, in some relatively obscure cases, you may find that not even tags supports the tag you have in mind; this may be because the tag is newly added to HTML and has not been incorporated into `htmltools` yet, or because it’s a tag that isn’t defined in HTML per se but is still understood by browsers (e.g. the `<circle>` tag from SVG). In these cases, you can fall back to the `tag()` (singular) function and pass it any tag name.
    - Example:
    ```
    tag("circle", list(cx="10", cy="10", r="20", stroke="blue", fill="white"))
    ```
    - Notice that the `tag()` function alone needs its attribute and children wrapped in a separate `list()` object.

**Using Named Arguments to Create Attributes**
* When calling a tag function, any named arguments become HTML attributes.
```
a(class="btn btn-primary", `data-toggle`="collapse", href="#collapseExample",
  "Link with href"
)
```
    * In raw HTML the above would have been:
    ```
    ## <a class="btn btn-primary" data-toggle="collapse" href="#collapseExample">Link with href</a>
    ```
* The preceding example includes some attributes with hyphens in their names. Be sure to quote such names using backticks, or single or double quotes. Quoting is also permitted, but not required, for simple alphanumeric names.
* Generally, HTML attribute values should be single-element character vectors, as in the above example. Other simple vector types like integers and logicals will be passed to `as.character()`.
* Another valid attribute value is `NA`. This means that the attribute should be included, but without an attribute value at all:
```
tags$input(type = "checkbox", checked = NA)
```
    * In raw HTML the above would have been:
    ```
    ## <input type="checkbox" checked/>
    ```
* You can also use `NULL` as an attribute value, which means the attribute should be ignored (as if the attribute wasn’t included at all). This is helpful for conditionally including attributes.
```
is_checked <- FALSE
tags$input(type = "checkbox", checked = if (is_checked) NA)
```
    * In raw HTML the above would have been:
    ```
    ## <input type="checkbox"/>
    ```

**Using Unnamed Arguments to Create Children**
* Tag functions interpret unnamed arguments as children. Like regular HTML tags, each tag object can have zero, one, or more children; and each child can be one of several types of objects.

**Tag Objects**
* Tag objects can contain other tag objects. Trees of tag objects can be nested as deeply as you like.
    * Example:
    ```
    div(
      p(
        strong(
          a(href="https://example.com", "A link")
        )
      )
    )
    ```

**Plain Text**
* Tag objects can contain plain text, in the form of single-element character vectors.
    * Example:
    ```
    p("I like turtles.")
    ```
* One important characteristic of plain text is that `htmltools` assumes you want to treat all characters as plain text, including characters that have special meaning in HTML like < and >. As such, any special characters will be automatically escaped.

**Verbatim HTML**
* Sometimes, you may have a string that should be interpreted as HTML; similar to plain text, except that special characters like `<` and `>` should retain their special meaning and be treated as markup. You can tell `htmltools` that these strings should be used verbatim (not escaped) by wrapping them with `HTML()`:
    * Example:
    ```
    html_string <- "I just <em>love</em> writing HTML!"
    div(HTML(html_string))
    ```

**Lists**
* While each call to a tag function can have as many unnamed arguments as you want, you can also pack multiple children into a single argument by using a `list()`. The following two code snippets will generate identical results:
```
tags$ul(
  tags$li("A"),
  tags$li("B"),
  tags$li("C")
)

# OR

tags$ul(
  list(
    tags$li("A"),
    tags$li("B"),
    tags$li("C")
  )
)
```
* It can sometimes be handy to use a list when generating tag function calls programmatically. The snippet below uses `lapply` to simplify the previous example.
```
tags$ul(
  lapply(LETTERS[1:3], tags$li)
)
```

**NULL**
* You can use `NULL` as a tag child. `NULL` children are similar to `NULL` attributes; they’re simply ignored, and are only supported to make conditional child items easier to express.
* In this example, we use `show_beta_warning` to decide whether or not to show a warning; if not, the result of the if clause will be `NULL`.
```
show_beta_warning <- FALSE

div(
  h3("Welcome to my Shiny app!"),
  if (show_beta_warning) {
    div(class = "alert alert-warning", role = "alert",
      "Warning: This app is in beta; some features may not work!"
    )
  }
)
```

**Mix and Match**
* Tag functions can be called with any number of unnamed arguments, and different types of children can be used within a single call.
* Example:
    ```
    div(
      "Text!",
      strong("Tags!"),
      HTML("Verbatim <span>HTML!</span>"),
      NULL,
      list(
        "Lists!"
      )
    )
    ```

**Customizing with CSS**
* We'll now switch to Cascading Style Sheets (CSS), the language that specifies the visual style and layout of the page.

**Introduction to CSS**
* CSS lets us specify directive that control how the HTML tree of tags and text is rendered; each directive is called a *rule.*
* Here’s some example CSS that includes two rules: one that causes all `<h3>` tags (level 3 headings) to turn red and italic, and one that hides all `<div class="alert">` tags in the `<footer>`.
```
h3 {
  color: red;
  font-style: italic;
}

footer div.alert {
  display: none;
}
```
* The part of the rule that precedes the opening curly brace is the selector; in this case, `h3`. The selector indicates which tags this rule applies to.
* The parts of the rule inside the curly braces are properties. This particular rule has two properties, each of which is terminated by a semicolon.

**CSS Selectors**
* You can select tags that match a specific `id` or `class`, select tags based on their parents, select tags based on whether they have sibling tags.
    - You can combine such criteria together using 'and', 'or', and 'not' semantics.
* Here are some extremely common selector patters:
    - `.foo` - All tags whose class attributes include `foo`
    - `div.foo` - All `<div>` tags whose `class` attributes include `foo`
    - `#bar` - The tag whose `id` is `bar`
    - `div#content p:not(#intro)` - All `<p>` tags inside the `<div>` whose `id` is `content`, except the `<p>` tag whose `id` is `intro`

**CSS Properties**
* The syntax of CSS properties is very simple and intuitive. The challenge is the shear number of them.
    - There are dozens upon dozens of properties that control typography, margin and padding, word wrapping and hyphenation, sizing and positioning, borders and shadows, scrolling and overflow, animation and 3D transforms.
* Here are some examples of common properties:
    - `font-family: Open Sans, Helvetica, sans-serif;` Display text using the Open Sans typeface, if it’s available; if not, fall back first to Helvetica, and then to the browser’s default sans serif font.
    - `font-size: 14pt;` set the font size to 14 point.
    - `width: 100%; height: 400px;` Set the width to 100% of the tag’s container, and the height to a fixed 400 pixels.
    - `max-width: 800px;` Don’t let the tag grow beyond 800 pixels wide.
* Most of the effort in mastering CSS is in knowing what properties are available to you, and understanding when and how to use them.

**Including Custom CSS in Shiny**
* Shiny gives you several options for adding custom CSS into your apps. Which method you choose will depend on the amount and complexity of your CSS.
* If you have one or two very simple rules, the easiest way to add CSS is by inserting a `<style>` tag, using `tags$style()`. This can go almost anywhere in your UI.
    * Example:
    ```
    library(shiny)

    ui <- fluidPage(
      tags$style(HTML("
        body, pre { font-size: 18pt; }
      "))
    )
    ```

**Standalone CSS file with `includeCSS`**
* The second method is to write a standalone `.css` file, and use the `includeCSS` function to add it to your UI.
    * Example:
    ```
    ui <- fluidPage(
      includeCSS("custom.css"),
      ... # the rest of your UI
    )
    ```
* The `includeCSS` call will return a `<style>` tag, whose body is the content of `custom.css`.

**Standalone CSS file with `<link>` Tag**
* You can also choose to serve up the `.css` file at a separate URL, and link to it from your UI.
To do this, create a `www` subdirectory in your application directory (the same directory that contains `app.R`) and put your CSS file there—for example, `www/custom.css`. Then add the following line to your UI:
    ```
    tags$head(tags$link(rel="stylesheet", type="text/css", href="custom.css"))
    ```
    - Note that the `href` attribute should not include `www`; Shiny makes the contents of the `www` directory available at the root URL path.

#### Chapter 7: Why Reactivity?
**Why Reactive Programming?**
* Reactive programming is a style of programming that emphasizes values that change over time, and calculating and actions that depend on those values.
* For Shiny apps to be useful, we need two things:
    1. Expressions and outputs should update whenever one of there input values changes. This ensures that input and output stay in sync.
    2. Expressions and outputs should update only when one of their inputs changes. This ensures that apps respond quickly to user input, doing the minimal amount.
* It's relatively easy to satisfy one of the two conditions, but much harder to satisfy both.

**Reactive Programming**
* To enable reactivity in the console we use a special Shiny mode `consoleReactive(TRUE)` to exemplify reactivity in action.
* This mode isn’t enabled by default because it makes a certain class of bug harder to spot in an app, and it’s primary benefit is to help you understand reactivity.
```
library(shiny)
consoleReactive(TRUE)
```
* To create a variable that is reactive we will use:
```
temp_c <- reactiveVal(10)
```
    - This creates a single reactive value that has a special syntax for getting and setting its value. To get the value you call it like a function; to set the value, you call it with a value:
    ```
    temp_c(20) # set
    temp_c() # get
    ```
* Now we can create a reactive expression that depends on this value:
```
temp_f <- reactive({
    message('Converting')
    (temp_c() + 32) * 9 / 5
    })

temp_f()
```
    - Then if `temp_c()` changes, `temp_f()` will also be up to date.
    ```
    temp_c(-3)
    temp_f()
    ```
* Note that the conversion only happens if we request the value of `temp_f()` (unlike the event-driven approach), and the computation happens only once (unlike the functional approach). A reactive expression caches the result of the last call, and will only recompute if one of the inputs changes.
* Together these properties ensure that Shiny does as little work as possible, making your app as efficient as possible.

**A Brief History of Reactive Programming**
* Spreadsheets are closely related to reactive programming: you declare the relationship between cells (using formulas), and when one cell changes, all of its dependencies automatically update.
* Now reactive programming has come to dominate UI programming on the web, with hugely popular frameworks like React, Vue.js, and Angular which are either inherently reactive or designed to work hand-in-hand with reactive backends.

#### Chapter 9: Dependency Tracking
**How Dependency Tracking Works**
* The most striking aspect of reactive programming in Shiny is that a reactive expression, observer, or output “knows” which reactive values/inputs and reactive expressions it depends on.
    - Example:
    ```
    ouput$plot <- renderPlot({
        plot(head(cars, input$rows))
        })
    ```
* This is because Shiny uses dynamic instrumentation, where as the code is run it collects additional information about what's going on.

**Reactive Context**
* In the example above, before the plot output begins executing, it creates an object that's internal to Shiny called a reactive context.
* You will never actually see reactive contexts.
* The reactive context doesn’t represent the plot output as a whole, but just a single execution of the output.
* If, over the life of a Shiny session, the plot is (re)rendered a dozen times, then a dozen reactive contexts will have been created.
* The Shiny package has a top-level variable (like a global variable, but one only visible to code inside the Shiny package) that is always pointing to the “current” or “active” reactive context.
* The plot output assigns its new context to this variable, then executes its code block, then restores the previous value of the variable.
    * Example (not real code just an illustration of how this works):
    ```
    # Create the new context
    ctx <- ReactiveContext$new()

    # Set as the current context (but save the prev context)
    prev_ctx <- shiny:::currentContext
    shiny:::currentContext <- ctx

    # Actually run user code here
    renderPlot({ ... })

    # Restore the prev context
    shiny:::currentContext <- prev_ctx
    ```
* The purpose of the context object is to provide a rendezvous point between the reactive consumer that is executing, and the reactive producers that it’s reading from
* There are two important methods on context objects:
    1. `invalidate()`: Informs the context that a producter that it read from is now potentially out of date (invalidated); and so whatever reactive consumer owns the context should also be considered out of date.
    2. `onInvalidated(func)`: Asks the context to invoke the given callback function in the future, if and when `invalidate()` is called.

**Conditional Dependency**    
* Example App:
```
library(shiny)

ui <- fluidPage(
  selectInput("choice", "A or B?", c("a", "b")),
  numericInput("a", "a", 0),
  numericInput("b", "b", 10),
  textOutput("out")
)

server <- function(input, output, session) {
  output$out <- renderText({
    if (input$choice == "a") {
      input$a
    } else {
      input$b
    }
  })
}
```

#### Chapter 11: Reactive Components
**Building Blocks**
* There are three objects almost all reactive programming related functions in Shiny are built on.
    - They are called reactive primitives because they are a fundamental part of the reactive framework and can not be implemented from simpler components.
    1. Reactive values (used to implement reactive inputs)
    2. Expressions
    3. Observers (used to implement reactive outputs)

**Reactive Values**
* `inputs` are a read-only example of a reactive value that Shiny uses to communicate user actions in the browser to your R code.
* A reactive value is a special type of function that returns its current value when called without arguments, and updates its value when called with a single argument.
* The big difference between reactive values and ordinary R values is that reactive values track who accesses them. And then when the value changes, it automatically lets everyone know that there's been a change.
* A regular variable asks "What's that value of `x`?", while a reactive value asks "What's the value of `input$x`? And please notify me the next time `input$x` changes!"
* There are two fundamental types of reactive consumers in Shiny.
    1. Actions (with side effects)
        - Change teh world in some way: e.g. `print()`, `plot()`, `write.csv()`
    2. Calculations (no side effects)
        - return a value: e.g. `sum()`, `mean()`, or `read.csv()`
* Almost all R functions are either calculations or actions.
* In programming terminology, changing the world is called a side-effect and by this we mean any effects apart from a function's return value.
    - Changing a file on a desk is a side effect.
    - Printing words to the console is a side effect.
    - Sending a message to another computer is a side effect.

**Observers: Automatic Actions**
* Observers are reactive consumers that take a code block that performs an action of some kind.
* Observers are reactive consumers because they know how to respond to one of their dependencies changes: they re-run their code block.
* This observer does two things. It prints out a message giving the current value of `x`, and it subscribes to be notified of the next change to `x()`. When `x` changes, and this observer is notified, it requests that the Shiny runtime run its code block again, and two steps repeat.


**Reactive Expressions: Smart Calculations**
* Reactive expressions are the other fundamental type of reactive consumer.
* While observers model actions that have side effects, reactive expressions model calculations that return values.
* Creating a reactive expression is like declaring an R function: nothing actually happens until you call it.
* Reactive expressions are reactive: they know when the reactive values they’ve read have changed, and they alert their readers when their own value may have changed. They’re also lazy: they contain code, but that code doesn’t execute unless/until someone tries to actually retrieve the value of the reactive expression (by calling it like a function).
* Most importantly, reactive expressions cache their most recent value.
* These particular properties – laziness, caching, reactivity, and lack of side effects – combine to give us an elegant and versatile building block for reactive programming.

**Outputs**
* Outputs are reactive consumers.
* Output code is allowed to read reactive values like `input$x` and then know when those reactive dependencies change.
* Whereas observers execute eagerly and reactive expressions execute lazily, outputs are somewhere in between.
* When an output’s corresponding UI element is visible in the browser, outputs execute eagerly; that is, once at startup, and once anytime their relevant inputs or reactive expressions change.
    * However, if their UI element becomes hidden (e.g. it is located on a `tabPanel` that is not active, or `removeUI` is called to actively remove it from the page) then Shiny will automatically suspend (pause) that output from reactively executing.
