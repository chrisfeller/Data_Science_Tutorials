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
