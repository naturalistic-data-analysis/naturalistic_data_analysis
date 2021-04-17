# Preprocessing
*Written by Luke Chang*

There are currently no agreed upon conventions for preprocessing naturalistic neuroimaging data. In this tutorial, we show how we preprocessed the datasets used in all of our tutorials. This tutorial assumes you have some basic knowledge of preprocessing. If you have questions about the specific details, we enourage you to read other tutorials, such as the preprocessing [overview](https://dartbrains.org/content/Preprocessing.html) from the Dartbrains course.

In our [lab](http://cosanlab.com/), we perform standard realignment and spatial normalization using a custom [nipype](https://nipype.readthedocs.io/en/latest/) [workflow](https://github.com/cosanlab/cosanlab_preproc) or [fmriprep](https://fmriprep.org/en/stable/). We also perform basic denoising by removing global and multivariate spikes, average CSF activity, linear/quadratic trends, and 24 motion covariates (e.g., 6 centered realignment, their squares, derivative, and squared derivatives). We are very cautious about performing high-pass filtering as many of the effects we are interested in occur in [slower frequencies](https://www.biorxiv.org/content/early/2018/12/16/487892). We find that including average activity from a CSF mask helps a lot in reducing different types of physiological and motion related artifacts. We typically apply spatial smoothing, but depending on the question we don't always perform this step. For spatial feature selection, we rarely use searchlights and instead tend to use parcellations. This allows us to quickly prototype analysis ideas using smaller numbers of parcels (e.g., n=50) and then increase the number if we want greater spatial specificity. Starting with K=50 parcellation speeds up our computation by several orders of magnitude compared to voxelwise or searchlight approaches. In addition, the bonferroni correction for multiple comparisons is p < 0.001, which is a reasonable threshold to observe statistically significant results with our research questions. We usually use a [parcellation](https://neurovault.org/collections/2099/) scheme that we developed with [Tal Yarkoni](https://talyarkoni.org/) based on meta-analytic coactivation using the neurosynth database. Selecting the right spatial features for a particular question is a surprisingly under-appreciated topic. See [Jolly & Chang, 2021](https://academic.oup.com/scan/advance-article/doi/10.1093/scan/nsab010/6121195?login=true) discussing the costs and benefits of different spatial feature selection strategies. Head motion is also an important consideration with naturalistic designs particularly with long scanning sessions or when participants actively speak in the scanner. We have recently evaluated the efficacy of using [caseforge headcases](https://caseforge.co/?gclid=CjwKCAjw8df2BRA3EiwAvfZWaGA5Jz_RABlW6vKdvdjJCULwaeFW3BHMI-FkSLX27DbS4B7LlUHOrhoCYKIQAvD_BwE) in reducing head motion in naturalistic viewing studies ([Jolly et al., 2020](https://www.sciencedirect.com/science/article/pii/S1053811920306935?via%3Dihub)). 


## Naturalistic Data Preprocessing
For the two datasets we are using in this course (Sherlock & Paranoia), we performed very minimal preprocessing. First, we used [fmriprep](https://fmriprep.readthedocs.io/en/stable/) to realign and spatially normalize the data. If you don't have strong opinions about the details of preprocessing, we highly recommend using fmriprep, which is developed and maintained by a team at the [Center for Reproducible Research](http://reproducibility.stanford.edu/) led by Russ Poldrack and Chris Gorgolewski. Fmriprep was designed to provide an easily accessible, state-of-the-art interface that is robust to variations in scan acquisition protocols, requires minimal user input, and provides easily interpretable and comprehensive error and output reporting. We like that they share a docker container with all of the relevant software packages, it is very simple to run, and that there is a large user base that actively report bugs so that it is constantly improving.

After preprocessing with fmriprep, we smoothed the data (fwhm=6mm) and performed basic voxelwise denoising using a GLM. This entails including the 6 realignment parameters, their squares, their derivatives, and squared derivatives. We also include dummy codes for spikes identified from global signal outliers and outliers identified from frame differencing (i.e., temporal derivative). We chose to not perform high-pass filtering and instead include linear & quadratic trends, and average CSF activity to remove additional physiological and scanner artifacts. Finally, to save space, we downsampled to Float32 precision.

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltools.stats import regress, zscore
from nltools.data import Brain_Data, Design_Matrix
from nltools.stats import find_spikes 
from nltools.mask import expand_mask

def make_motion_covariates(mc, tr):
    z_mc = zscore(mc)
    all_mc = pd.concat([z_mc, z_mc**2, z_mc.diff(), z_mc.diff()**2], axis=1)
    all_mc.fillna(value=0, inplace=True)
    return Design_Matrix(all_mc, sampling_freq=1/tr)

base_dir = '/Volumes/Engram/Data/Sherlock/fmriprep'

fwhm=6
tr = 1.5
outlier_cutoff = 3

file_list = [x for x in glob.glob(os.path.join(base_dir, '*/func/*preproc*gz')) if 'denoised' not in x] 
for f in file_list:
    sub = os.path.basename(f).split('_')[0]

    data = Brain_Data(f)
    smoothed = data.smooth(fwhm=fwhm)

    spikes = smoothed.find_spikes(global_spike_cutoff=outlier_cutoff, diff_spike_cutoff=outlier_cutoff)
    covariates = pd.read_csv(glob.glob(os.path.join(base_dir, sub, 'func', '*tsv'))[0], sep='\t')
    mc = covariates[['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z']]
    mc_cov = make_motion_covariates(mc, tr)
    csf = covariates['csf'] # Use CSF from fmriprep output
    dm = Design_Matrix(pd.concat([csf, mc_cov, spikes.drop(labels='TR', axis=1)], axis=1), sampling_freq=1/tr)
    dm = dm.add_poly(order=2, include_lower=True) # Add Intercept, Linear and Quadratic Trends

    smoothed.X = dm
    stats = smoothed.regress()
    stats['residual'].data = np.float32(stats['residual'].data) # cast as float32 to reduce storage space
    stats['residual'].write(os.path.join(base_dir, sub, 'func', f'{sub}_denoise_smooth{fwhm}mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))


We also saved the cropped denoised viewing data as an hdf5 file to speed up loading times when using nltools.

data_dir = '/Volumes/Engram/Data/Sherlock/fmriprep'

for scan in ['Part1', 'Part2']:
    file_list = glob.glob(os.path.join(data_dir, '*', 'func', f'*crop*{scan}*nii.gz'))
    for f in file_list:
        data = Brain_Data(f)
        data.write(f"{f.split('.nii.gz')[0]}.hdf5")

Finally, we have also precomputed average activations within a whole brain parcellation (n=50) for some of the tutorials.

data_dir = '/Volumes/Engram/Data/Sherlock/fmriprep'

mask = Brain_Data('http://neurovault.org/media/images/2099/Neurosynth%20Parcellation_0.nii.gz')

for scan in ['Part1', 'Part2']:
    file_list = glob.glob(os.path.join(data_dir, '*', 'func', f'*crop*{scan}*hdf5'))
    for f in file_list:
        sub = os.path.basename(f).split('_')[0]
        print(sub)
        data = Brain_Data(f)
        roi = data.extract_roi(mask)
        pd.DataFrame(roi.T).to_csv(os.path.join(os.path.dirname(f), f"{sub}_{scan}_Average_ROI_n50.csv" ), index=False)

## Recommended reading

* Jolly, E., Sadhukha, S., & Chang, L. J. (2020). Custom-molded headcases have limited efficacy in reducing head motion for fMRI. BioRxiv.

* Jolly, E., & Chang, L.J. (2020). Multivariate spatial feature selection in fMRI. OSF.

* Chang, L. J., Jolly, E., Cheong, J. H., Rapuano, K., Greenstein, N., Chen, P. H. A., & Manning, J. R. (2018). Endogenous variation in ventromedial prefrontal cortex state dynamics during naturalistic viewing reflects affective experience. BioRxiv, 487892.

Have thoughts on preprocessing?  Please share them as a github [issue](https://github.com/naturalistic-data-analysis/naturalistic_data_analysis/issues) on our jupyter-book repository and we can incorporate them into the notebook.