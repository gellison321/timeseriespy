from distutils.core import setup
from setuptools import find_packages

setup(
  name = 'timeseriespy',
  packages = find_packages(),
  version = '1.0.1',
  license='',
  description = 'A clustering tool for timeseries data with temporal distortions.',
  author = 'Grant Ellison',
  author_email = 'gellison321@gmail.com',
  url = 'https://github.com/gellison321/tsclustering',
  download_url = 'https://github.com/gellison321/tsclustering/archive/refs/tags/1.0.1.tar.gz',
  keywords = ['timeseries', 'barycenter', 'clustering', 'data science','data analysis', 'kmeans', 'time series clustering',
              'time series comparison', 'classificaiton', 'machine learning', 'barycenter'],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
  ],
  install_requires=['numpy','scipy'],
)