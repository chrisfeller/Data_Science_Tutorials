### Pandas Styling
#### July 2019

#### Pandas Styling Example
```
styled_data = (data
                 .style
                 .set_table_styles(
                 [{'selector': 'tr:nth-of-type(odd)',
                   'props': [('background', '#eee')]},
                  {'selector': 'tr:nth-of-type(even)',
                   'props': [('background', 'white')]},
                  {'selector':'th, td', 'props':[('text-align', 'center')]}])
                 .set_properties(subset=['COLUMN_1'], **{'text-align': 'left'})
                 .hide_index()
                 .background_gradient(subset=['COLUMN_1'], cmap='Reds'))
html = styled_data.render()
imgkit.from_string(html, 'plots/data.png', {'width': 1})
```

#### Pandas Styling CSS Example
```
def dataFrame_to_image(data, css, outputfile="df.png", format="png"):
    '''
    Render a Pandas DataFrame as an image. Adopted from :
    https://medium.com/@andy.lane/convert-pandas-dataframes-to-images-using-imgkit-5da7e5108d55

    Args:
        data: a pandas DataFrame
        css: a string containing rules for styling the output table. This must
             contain both the opening an closing <style> tags.
    Return:
        *outputimage: filename for saving of generated image
        *format: output format, as supported by IMGKit. Default is "png"
    '''
    fn = str(random.random()*100000000).split(".")[0] + ".html"

    try:
        os.remove(fn)
    except:
        None
    text_file = open(fn, "a")

     write the CSS
    text_file.write(css)
     write the HTML-ized Pandas DataFrame
    text_file.write(data.to_html(index=False))
    text_file.close()

     See IMGKit options for full configuration,
     e.g. cropping of final image
    imgkitoptions = {"format": format}

    imgkit.from_file(fn, outputfile, options=imgkitoptions)
    os.remove(fn)



 # Save .png of correlations w/o background gradient
 css = """
     <style type=\"text/css\">
     table {
     color: 333;
     font-family: Helvetica, Arial, sans-serif;
     width: 640px;
     border-collapse:
     collapse;
     border-spacing: 0;
     }
     td, th {
     border: 1px solid transparent; /* No more visible border */
     height: 30px;
     }
     th {
     background: DFDFDF; /* Darken header a bit */
     font-weight: bold;
     }
     td {
     background: FAFAFA;
     text-align: center;
     }
     table tr:nth-child(odd) td{
     background-color: white;
     }
     </style>
     """
dataFrame_to_image(corr_df, css, outputfile="plots/correlation_table.png", format="png")
```

####
* [Tutorial 1](https://pbpython.com/styling-pandas.html)
* [Tutorial 2](https://medium.com/@andy.lane/convert-pandas-dataframes-to-images-using-imgkit-5da7e5108d55)
