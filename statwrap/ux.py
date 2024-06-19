'''
Stats functions adapted to the conventions of Freedman, Pisani, and Purves 2007.
'''

import ipywidgets as widgets
import io
import pandas as pd
from IPython.display import display
from IPython import get_ipython


class DataUploadWidget:
    """
    A widget for uploading data files and creating a pandas DataFrame.

    Parameters
    ----------
    variable_name : str
        The name of the variable to store the DataFrame in the IPython environment.
    accept : str, optional
        A comma-separated string of accepted file extensions. Default is '.csv,.xls,.xlsx,.xlsm,.xlsb,.odf,.ods,.odt'.
    auto_display : bool, optional
        If True, the widget is displayed immediately upon creation. Default is True.

    Attributes
    ----------
    accept : str
        Accepted file extensions.
    supported : set
        Set of supported file extensions.
    uploader : ipywidgets.FileUpload
        File upload widget.
    submit_button : ipywidgets.Button
        Button to create the DataFrame from the uploaded file.
    variable_name : str
        Name of the variable to store the DataFrame.
    auto_display : bool
        If True, the widget is displayed immediately upon creation.

    Methods
    -------
    createDF()
        Creates a pandas DataFrame from the uploaded file.
    uploadData()
        Displays the file upload widget and button, and handles the file upload event.

    Examples
    --------
    Single DataFrame Usage:

    >>> uploader = DataUploadWidget("df")

    Multiple DataFrame Usage:
    
    >>> uploader1 = DataUploadWidget("df1")
    >>> uploader2 = DataUploadWidget("df2")

    Delayed Output Usage:

    >>> output = widgets.Output()
    >>> with output:
    >>>     uploader = DataUploadWidget('df3', auto_display=False)
    
    >>> display(output)
    >>> display(uploader.uploader, uploader.submit_button)
    """

    def __init__(self, variable_name, accept: str = '.csv,.xls,.xlsx,.xlsm,.xlsb,.odf,.ods,.odt', auto_display: bool = True):
        self.accept = accept
        self.supported = {'.csv', '.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'}
        self.uploader = widgets.FileUpload(accept=accept, multiple=False)
        self.submit_button = widgets.Button(description=f"Create DF as \"{variable_name}\"")
        self.variable_name = variable_name
        self.auto_display = auto_display

        self.uploadData()

    def createDF(self):
        """
        Create a pandas DataFrame from the uploaded file.

        Returns
        -------
        pd.DataFrame
            DataFrame created from the uploaded file.

        Raises
        ------
        ValueError
            If the file extension is unsupported.
        """
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
        """
        Display the file upload widget and button, and handle the file upload event.
        """
        def on_submit(button):
            df = self.createDF()
            ipython = get_ipython()
            ipython.user_global_ns[self.variable_name] = df
            display(df.head())
        
        self.submit_button.on_click(on_submit)
        
        if self.auto_display:
            display(self.uploader, self.submit_button)