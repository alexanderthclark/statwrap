import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='StatWrap',
    version='0.1.4',
    description='A package for people new to statistics and Python.',
    long_description=long_description,  # This is the new line
    long_description_content_type="text/markdown",  # This is the new line
    url='https://github.com/alexanderthclark/statwrap',
    project_urls={
        'Documentation': 'https://statwrap.readthedocs.io/'
    },
    author='Alexander Clark',
    install_requires=['pandas','numpy','scipy', 'IPython', 'matplotlib'],
    author_email='',  # consider adding your email
    packages=setuptools.find_packages(),
    zip_safe=False,
)
