import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='pandas_little_helper',
     version='0.1a1',
     author="Yair Eshel Cahansky",
     author_email="guruyaya@gmail.com",
     description="List of helpful data science stuff",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/guruyaya/pandas_little_helper",
     packages=['pandas_little_helper'],
     classifiers=[
         "Programming Language :: Python :: 3",
         'Programming Language :: Python :: 3.8',
         'Programming Language :: Python :: 3.9',
         'Intended Audience :: Developers',      # Define that your audience are developers
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent",
     ],
 )