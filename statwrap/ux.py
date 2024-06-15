import ipywidgets as widgets
import io
import pandas as pd
from IPython.display import display

class DataUploadWidget:
    def __init__(self, accept: str = '.csv,.xls,.xlsx,.xlsm,.xlsb,.odf,.ods,.odt'):
        self.accept = accept
        self.supported = {'.csv', '.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'}
        self.uploader = widgets.FileUpload(accept=accept, multiple=False)
        self.submit_button = widgets.Button(description="Submit")
        self.df = None

    def createDF(self):
        file_dict = self.uploader.value[0]
        file_content = file_dict['content']
        content_stream = io.BytesIO(file_content.tobytes())
        file_name = file_dict['name']
        file_extension = file_name.split('.')[-1]

        if file_extension == 'csv':
            df = pd.read_csv(content_stream)
        elif file_extension in {'xls', 'xlsx', 'xlsm', 'xlsb'}:
            df = pd.read_excel(content_stream)
        elif file_extension in {'odf', 'ods', 'odt'}:
            df = pd.read_excel(content_stream, engine='odf')
        else:
            raise ValueError("Unsupported file extension")

        return df

    def uploadData(self):
        def on_submit(button):
            self.df = self.createDF()
            display(self.df.head())
        
        self.submit_button.on_click(on_submit)
        
        display(self.uploader, self.submit_button)
