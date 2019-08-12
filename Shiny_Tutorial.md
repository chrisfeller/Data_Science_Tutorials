### R Shiny Tutorial
#### August 2019

---
#### How to Start Shiny Video Series
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
