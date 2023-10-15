import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoviz",
    version="0.1.730",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Automatically Visualize any dataset, any size with a single line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/AutoViz",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "ipython",
        "jupyter",
        "xlrd",
        "wordcloud",
        "emoji",
        "numpy",
        "pandas",
        "pyamg",
        "matplotlib>=3.3.3",
        "seaborn>=0.11.1",
        "scikit-learn",
        "statsmodels",
        "nltk",
        "textblob",
        "holoviews~=1.17.1",
        "bokeh~=3.2.2",
        "hvplot~=0.8.4",
        "panel>=1.2.3",
        "xgboost>=0.82",
        "fsspec>=0.8.3",
        "typing-extensions>=4.1.1",
        "pandas-dq==1.28"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
