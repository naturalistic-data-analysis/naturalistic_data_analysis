# Software Installation
*Written by Luke Chang*

For this course, we will be providing hands on tutorials working with real data using open source software written in the Python programming language. We have included some introductory tutorials in the **Background Resources** section to help you get up to speed on using [Python](https://dartbrains.org/features/notebooks/1_Introduction_to_Programming.html) and in particular how to manipulate data with [pandas](https://dartbrains.org/features/notebooks/2_Introduction_to_Pandas.html), [plot](https://dartbrains.org/features/notebooks/3_Introduction_to_Plotting.html) data, and work with [neuroimaging data](https://dartbrains.org/features/notebooks/5_Introduction_to_Neuroimaging_Data.html). All of our tutorials will be provided in the form of a [Jupyter Notebook](https://jupyter.org/). These notebooks are distributed via a [jupyter-book](https://jupyterbook.org/intro.html) and can be downloaded to run on your personal computer.

Throughout this course various tutorials will use the following libraries:

- [datalad](https://www.datalad.org/) - data versioning software (see Download Data tutorial for installation instructions and overview)
- [nibabel](https://nipy.org/nibabel/) - a library to read and write common neuroimaging file formats.
- [nilearn](https://brainiak.org/) - a neuroimaging toolbox for performing statistical learning
- [brainiak](https://brainiak.org/) - a neuroimaging toolbox for performing advanced fMRI analyses
- [nltools](https://neurolearn.readthedocs.io/en/latest/) - a neuroimaging toolbox for performing multivariate analyses
- [hypertools](https://hypertools.readthedocs.io/en/latest/index.html) - a toolbox for visualizing high dimensional data
- [timecorr](https://timecorr.readthedocs.io/en/latest/) - a toolbox for calculating dynamic correlations
- [pliers](https://github.com/tyarkoni/pliers) - a toolbox for extracting features from multimodal data

## Installing Software
First, if you have never worked with Python before, we recommend installing Python via the [Anaconda distribution](https://www.anaconda.com/products/individual). This includes many popular packages used in scientific computing. Be sure to install Python 3.7. 

Package management in Python can be a little tricky. Anaconda has provided their own system called `conda`, which includes precompiled packages that are theoretically tested to reduce conflicts. Conda is usually a good place to start when installing packages. If a package isn't included in conda, you will next try installing the package using the Python packaging system called `pip`. This tends to be our main workhorse for installing packages. If a package isn't in PyPI, then you may have to install from a github repository. This can be done using `pip.

Package installation is probably easiest through the command line in a terminal, but you can also run commands in the shell by using the `!` cell magic.

If you use Python for many different projects, you may find it helpful to create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html), which provides a way to install a specific collection of packages. This can be useful if you use different versions of the same library for different projects, or if you want to avoid conflicts across packages.

### Conda
Some packages can be installed with `conda`. The more established scientific packages are already included in the anaconda distribution and usually can be installed via `conda`. The advantages of installing via `conda` is that distributions usually undergo an extra round of testing to make sure they are compatible and are precompiled. Smaller scale projects can often be installed via `conda` using a specific channel such as *conda-forge*. We have found some of the [brainiak](https://github.com/brainiak/brainiak) dependencies to be finicky to install with `pip` and instead recommend installing it with `conda`.

If you would like to create a conda environment for this course, run the following command in the terminal or with a `!` in front of it within a jupyter notebook.
```
conda create -n naturalistic python=3.7 anaconda
```
To activate your environment:
```
conda activate naturalistic
```
If you would like would like to exit your environment:
```
conda deactivate
```

One neat thing about `conda` environments is that you can select a specific one to serve as your kernel for a jupyter notebook. To get this to work, run:
```
conda install -c anaconda ipykernel
```
and then
```
python -m ipykernel install --user --name=naturalistic
```

Ok, let's install the packages we need from conda first.

!conda install numpy scipy scikit-learn pandas matplotlib seaborn

!conda install -c brainiak -c defaults -c conda-forge brainiak

### Github
Some packages are not released on [Anaconda](https://www.anaconda.com/products/individual) or [PyPI](https://pypi.org/) and must be installed directly from their github repository. For example, one of the dependencies in the [timecorr](https://timecorr.readthedocs.io/en/latest/) toolbox requires a package called `brainconn`, which must be installed from their github repository.

!pip install git+https://github.com/FIU-Neuro/brainconn#egg=brainconn

### Pip
Otherwise, `pip` will be the main way you install packages. Let's now install the rest of the packages you will need for this course.

!pip install nibabel datalad nilearn nltools hypertools timecorr pliers statesegmentation networkx nltk requests urllib3

## Jupyter Notebooks 
[Jupyter notebooks](https://jupyter.org/) are a great way to have your code, comments and results show up inline in a web browser. Work for this course will be done in Jupyter notebooks so you can reference what you have done, see the results and someone else could redo it in the future, similar to a typical lab notebook.

Rather than writing and re-writing an entire program, you can write lines of code and run them one at a time. Then, if you need to make a change, you can go back and make your edit and rerun the program again, all in the same window.

Finally, you can view examples and share your work with the world very easily through [nbviewer](https://nbviewer.jupyter.org/). One easy trick if you use a cloud storage service like dropbox is to paste a link to the dropbox file in nbviewer. These links will persist as long as the file remains being shared via dropbox.

If you would like to get a quick overview of Jupyter Notebooks check out [talk1](https://www.youtube.com/watch?v=CSkTJRNBTME&index=3&t=0s&list=PLEE6ggCEJ0H0KOlMKx_PUVB_16VoCfGj9) and [talk2](https://youtu.be/OKPSDQIT-ns) from the [MIND summer school](http://mindsummerschool.org/). 

## R
You can also connect notebooks not just to Python kernels, but also to other languages such as [R](https://www.r-project.org/). 

The easiest way to get started is to install `R` and the `r-irkernel` package using `conda`. 

Alternatively, if this doesn't work you can also install R directly from their [website](https://cran.r-project.org/mirrors.html). To connect this version of R to jupyter, we need to manually install the `IRkernel` from within the version of `R` that you downloaded.

Start R, then run the following commands:

```R
install.packages('IRkernel')
IRkernel::installspec()  # to register the kernel in the current R installation
```

If you end up using R in your work, you will likely also find [RStudio](https://rstudio.com/) as well as packages from the [tidyverse](https://www.tidyverse.org/) to be helpful.

!conda install -c r r-irkernel