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
