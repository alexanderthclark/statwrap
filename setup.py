import setuptools
setuptools.setup(name='StatWrap',
version='0.1.0',
description='A package for people new to statistics and Python.',
url='https://github.com/alexanderthclark/statwrap',
author='Alexander Clark',
install_requires=['pandas','numpy','scipy', 'IPython'],
author_email='',
packages=setuptools.find_packages(),
zip_safe=False)