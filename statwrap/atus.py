'''
Functions specific to the American Time Use Survey
'''
import pandas as pd
from zipfile import ZipFile

def get_microdata_link(file, year, multi_year=True):
    """
    Returns link to American Time Use Survey (ATUS) file data for given parameters.

    Parameters
    ----------
    file : str
        The name of the ATUS file ["resp", "rost", "sum", "act", "cps", "who"].
    year : int
        The survey year of interest. Use 22 for 2022 (%y).
    multi_year : bool, optional
        Returns either single or multi-year data, default is True.

    Returns
    -------
    str
        Link to the zip file.

    Examples
    --------
    >>> get_microdata_link('resp', 2023)
    'https://www.bls.gov/tus/datafiles/atusresp-0323.zip'
    """

    base_url = "https://www.bls.gov/tus/datafiles/atus"

    # Convert the year to a string and get the last two digits
    year_suffix = str(year)[-2:]

    # Multi-year handling
    if multi_year:
        link = f"{base_url}{file}-03{year_suffix}.zip"
    else:
        link = f"{base_url}{file}-20{year_suffix}.zip"

    return link

def read_zip(filepath):
    """
    Extracts a ZIP file containing ATUS data and reads the extracted .dat file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        The file path to the ZIP file containing the ATUS data. The .dat file within the ZIP
        is expected to have the same base name as the ZIP file (with dashes replaced by underscores)
        and a `.dat` extension.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the data from the extracted .dat file.

    Notes
    -----
    This function assumes that the .dat file within the ZIP archive has the same name as the
    ZIP file, with the extension changed to `.dat` and any dashes replaced with underscores.

    Examples
    --------
    >>> df = read_zip('atus_data.zip')
    >>> df.head()
    """
    with ZipFile(filepath, 'r') as z:
        z.extractall()
        dat = filepath.replace(".zip",'.dat').replace("-",'_')
        df = pd.read_csv(z.open(dat))
    return df

