import ipywidgets as widgets
import io
import pandas as pd

class BaseUpload(widgets.FileUpload):
    """
    Base class for uploading files. Simplifies behavior of widgets.FileUpload and makes accessing file content easier.
    """

    def __init__(self, accept: str, supported: set, **kwargs: str) -> None:
        '''
        Initialization function for BaseUpload class. It is best not to modify the accept attribute after initialization.

        :param accept: Comma-separated file types to accept, e.g. '.csv,.xlsx'. Behaves like for widgets.FileUpload.
        :type accept: str
        :param supported: Set of supported file extensions for accept parameter, {'.csv', '.xlsx'} for example.
        :type supported: set
        :return: None
        '''
        kwargs.update({'accept': accept})
        super().__init__(**kwargs)
        self.multiple = False
        self.layout = widgets.Layout(width='auto')
        self.supported = supported
        self.description = f"Upload {self.accept}"
        self.validate_accept()

    def update_description(self) -> None:
        '''
        Update description to specify accepted file types. Use if accept parameter is modified after initialization.

        :return: None
        '''
        self.description = f"Upload {self.accept}"
        self.validate_accept()

    def validate_accept(self) -> None:
        '''
        Validate only accepted file types are included.

        :return: None
        :raises ValueError: If accepted extensions are not a subset of supported extensions.
        '''
        extensions = [ext.strip() for ext in self.accept.split(',')]
        if self.supported and (set(extensions) - self.supported):
            raise ValueError(f"Unsupported file types in accept parameter. Supported types: {', '.join(self.supported)}")


class DataUpload(BaseUpload):
    """
    Widget for uploading data files and retrieving a pd.DataFrame.

    .. code-block:: python

        excel_uploader = DataUpload(accept = '.csv,.xlsx')
        display(excel_uploader)
        df = excel_uploader.content()  # if file uploaded
    """

    def __init__(self, accept: str = '.csv,.xlsx', **kwargs: str) -> None:
        '''
        Initialization function for DataUpload class.
        Only reads the first sheet of a multi-sheet excel file.

        :param accept: Comma-separated file type to accept including '.csv', '.xlsx', or '.csv,.xlsx'.
        :type accept: str
        :return: None
        '''
        self.supported = {'.csv', '.xlsx'}
        super().__init__(accept=accept, supported=self.supported, **kwargs)
        
    def createDF(self):
        file_dict = self.value[0]

        # Access the content field and convert to a stream
        file_content = file_dict['content']
        content_stream = io.BytesIO(file_content.tobytes())

        # Check the file extension to determine how to read it
        file_name = file_dict['name']
        file_extension = file_name.split('.')[-1]

        # Read the content into a DataFrame based on the file extension
        if file_extension == 'csv':
            df = pd.read_csv(content_stream)
        elif file_extension == 'xlsx':
            df = pd.read_excel(content_stream)
        else:
            raise ValueError("Unsupported file extension")

        # Print the first few rows of the DataFrame
        return df
    
    def content(self) -> pd.DataFrame:
        """
        Obtains content of uploaded file.

        :return: DataFrame.
        :rtype: pd.DataFrame
        """
        if not self.value:
            return pd.DataFrame()  # return an empty DataFrame if no file is uploaded

        # Extract the file metadata and content correctly
        for _, uploaded_file in self.value.items():
            filename = uploaded_file['name']
            content = uploaded_file['content']
            file_extension = f'.{filename.split(".")[-1]}'
            content = io.BytesIO(content)
        
            if file_extension == '.csv':
                df = pd.read_csv(content)
            elif file_extension == '.xlsx':
                df = pd.read_excel(content)
            else:
                raise ValueError("Unsupported file extension")

            return df
        return pd.DataFrame()