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
