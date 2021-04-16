# Download Data

*Written by Luke Chang*

Throughout this course we will be using two open source naturalistic datasets to demonstrate how to use different analytic techniques. If you end up using any of this data in a paper, be sure to cite the original papers.

The **Sherlock** dataset contains 16 participants that watched 50 minutes of Sherlock across 2 scanning run (i.e., Part1 & Part2) and then verbally recalled the narrative in the scanner using a noise cancelling microphone. The TR was 1.5. If you would like to access the stimuli, we have included the video from Part1 (1st 25min) and an audio recording of the show can be downloaded in the stimuli folder of this [openneuro](https://openneuro.org/datasets/ds002345/versions/1.0.1) page. We have preprocessed the data using fmriprep and performed denoising. See the [Preprocessing](http://naturalistic-data.org/features/notebooks/Preprocessing.html) tutorial for more details. Note that we have also *cropped* the viewing files so that each subject has the same number of TRs and are aligned in time at the start of the movie. The Recall data has not been cropped, but we have included the details of when the subjects recall specific scenes in the `Sherlock_Recall_Scene_n50_Onsets.csv` in the onset folder. Finally, we have also included some scene annotations shared by the authors in the `Sherlock_Segments_1000_NN_2017.xlsx` file in the onsets folder.

    Chen, J., Leong, Y., Honey, C. et al. Shared memories reveal shared structure in neural activity across individuals. Nat Neurosci 20, 115–125 (2017). https://doi.org/10.1038/nn.4450

The **Paranoia** dataset contains 23 participants who listened to 22 minute original narrative that describes an ambiguous social scenario. It was written such that some individuals might find it highly suspicious. The transcript and audio recording can be downloaded in the stimuli folder on [openneuro](https://openneuro.org/datasets/ds001338/versions/1.0.0). The TR was 1s and there was a 3 sec fixation before the beginning of each run.

    Finn, E.S., Corlett, P.R., Chen, G. et al. Trait paranoia shapes inter-subject synchrony in brain activity during an ambiguous social narrative. Nat Commun 9, 2043 (2018). https://doi.org/10.1038/s41467-018-04387-2

The datasets are being shared using [DataLad](https://www.datalad.org/) on the [German Neuroinformatics Node](http://www.g-node.org/), which is an international forum for sharing experimental data and analysis tools.

In this notebook, we will walk through how to access the datset using DataLad. 

## DataLad

The easist way to access the data is using [DataLad](https://www.datalad.org/), which is an open source version control system for data built on top of [git-annex](https://git-annex.branchable.com/). Think of it like git for data. It provides a handy command line interface for downloading data, tracking changes, and sharing it with others.

While DataLad offers a number of useful features for working with datasets, there are three in particular that we think make it worth the effort to install for this course.

1) Cloning a DataLad Repository can be completed with a single line of code `datalad clone <repository>` and provides the full directory structure in the form of symbolic links. This allows you to explore all of the files in the dataset, without having to download the entire dataset at once.

2) Specific files can be easily downloaded using `datalad get <filename>`, and files can be removed from your computer at any time using `datalad drop <filename>`. As these datasets are large, this will allow you to only work with the data that you need for a specific tutorial and you can drop the rest when you are done with it.

3) All of the DataLad commands can be run within Python using the datalad [python api](http://docs.datalad.org/en/latest/modref.html).

We will only be covering a few basic DataLad functions to get and drop data. We encourage the interested reader to read the very comprehensive DataLad [User Handbook](http://handbook.datalad.org/en/latest/) for more details and troubleshooting.

### Installing Datalad

DataLad can be easily installed using [pip](https://pip.pypa.io/en/stable/).

`pip install datalad`

Unfortunately, it currently requires manually installing the [git-annex](https://git-annex.branchable.com/) dependency, which is not automatically installed using pip.

If you are using OSX, we recommend installing git-annex using [homebrew](https://brew.sh/) package manager.

`brew install git-annex`

If you are on Debian/Ubuntu we recommend enabling the [NeuroDebian](http://neuro.debian.net/) repository and installing with apt-get.

`sudo apt-get install datalad`

For more installation options, we recommend reading the DataLad [installation instructions](https://git-annex.branchable.com/).


!pip install datalad

### Download Data with DataLad

#### Download Sherlock
The Sherlock dataset can be accessed at the following location https://gin.g-node.org/ljchang/Sherlock. To download the Sherlock dataset run `datalad install https://gin.g-node.org/ljchang/Sherlock` in a terminal in the location where you would like to install the dataset. The full dataset is approximately 109gb.

You can run this from the notebook using the `!` cell magic.

!datalad install https://gin.g-node.org/ljchang/Sherlock

#### Download Paranoia
The Paranoia dataset can be accessed at the following location https://gin.g-node.org/ljchang/Paranoia. To download the Paranoia dataset run `datalad clone https://gin.g-node.org/ljchang/Paranoia`. The full dataset is approximately 100gb.

!datalad install https://gin.g-node.org/ljchang/Paranoia

## Datalad Basics

You might be surprised to find that after cloning the dataset that it barely takes up any space `du -sh`. This is because cloning only downloads the metadata of the dataset to see what files are included.

You can check to see how big the entire dataset would be if you downloaded everything using `datalad status`.

!datalad status --annex

### Getting Data
One of the really nice features of datalad is that you can see all of the data without actually storing it on your computer. When you want a specific file you use `datalad get <filename>` to download that specific file. Importantly, you do not need to download all of the dat at once, only when you need it.

Now that we have cloned the repository we can grab individual files. For example, suppose we wanted to grab the first subject's confound regressors generated by fmriprep.

!datalad get fmriprep/sub-01/func/sub-01_task-sherlockPart1_desc-confounds_regressors.tsv

Now we can check and see how much of the total dataset we have downloaded using `datalad status`

!datalad status --annex all

If you would like to download all of the files you can use `datalad get .`. Depending on the size of the dataset and the speed of your internet connection, this might take awhile. One really nice thing about datalad is that if your connection is interrupted you can simply run `datalad get .` again, and it will resume where it left off.

You can also install the dataset and download all of the files with a single command `datalad install -g https://gin.g-node.org/ljchang/Sherlock`. You may want to do this if you have a lot of storage available and a fast internet connection. For most people, we recommend only downloading the files you need for a specific tutorial.

### Dropping Data
Most people do not have unlimited space on their hard drives and are constantly looking for ways to free up space when they are no longer actively working with files. Any file in a dataset can be removed using `datalad drop`. Importantly, this does not delete the file, but rather removes it from your computer. You will still be able to see file metadata after it has been dropped in case you want to download it again in the future.

As an example, let's drop the Sherlock confound regressor .tsv file.

!datalad drop fmriprep/sub-01/func/sub-01_task-sherlockPart1_desc-confounds_regressors.tsv

### Datalad has a Python API!
One particularly nice aspect of datalad is that it has a Python API, which means that anything you would like to do with datalad in the commandline, can also be run in Python. See the details of the datalad [Python API](http://docs.datalad.org/en/latest/modref.html).

For example, suppose you would like to clone a data repository, such as the Sherlock dataset. You can run `dl.clone(source=url, path=location)`. Make sure you set `sherlock_path` to the location where you would like the Sherlock repository installed.

import os
import glob
import datalad.api as dl
import pandas as pd

sherlock_path = '/Users/lukechang/Downloads/Sherlock'

dl.clone(source='https://gin.g-node.org/ljchang/Sherlock', path=sherlock_path)


We can now create a dataset instance using `dl.Dataset(path_to_data)`.

ds = dl.Dataset(sherlock_path)

How much of the dataset have we downloaded?  We can check the status of the annex using `ds.status(annex='all')`.

results = ds.status(annex='all')

Looks like it's empty, which makes sense since we only cloned the dataset. 

Now we need to get some data. Let's start with something small to play with first.

Let's use `glob` to find all of the tab-delimited confound data generated by fmriprep. 

file_list = glob.glob(os.path.join(sherlock_path, 'fmriprep', '*', 'func', '*tsv'))
file_list.sort()
file_list

glob can search the filetree and see all of the relevant data even though none of it has been downloaded yet.

Let's now download the first subjects confound regressor file and load it using pandas.

result = ds.get(file_list[0])

confounds = pd.read_csv(file_list[0], sep='\t')
confounds.head()

What if we wanted to drop that file? Just like the CLI, we can use `ds.drop(file_name)`.

result = ds.drop(file_list[0])

To confirm that it is actually removed, let's try to load it again with pandas.

confounds = pd.read_csv(file_list[0], sep='\t')


Looks like it was successfully removed.

We can also load the entire dataset in one command if want using `ds.get(dataset='.', recursive=True)`. We are not going to do it right now as this will take awhile and require lots of free hard disk space.

Let's actually download one of the files we will be using in the tutorial. First, let's use glob to get a list of all of the functional data that has been preprocessed by fmriprep, denoised, and smoothed.

file_list = glob.glob(os.path.join(sherlock_path, 'fmriprep', '*', 'func', '*crop*nii.gz'))
file_list.sort()
file_list

Now let's download the first subject's file using `ds.get()`. This file is 825mb, so this might take a few minutes depending on your internet speed.

result = ds.get(file_list[0])

How much of the dataset have we downloaded?  We can check the status of the annex using `ds.status(annex='all')`.

result = ds.status(annex='all')

Ok, that concludes our tutorial for how to download data for this course with datalad using both the command line interface and also the Python API.