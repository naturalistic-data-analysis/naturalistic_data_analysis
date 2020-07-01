# Neuroimaging Analysis Methods For Naturalistic Data
Naturalistic stimuli, such as films or stories, are growing in popularity for human neuroimaging experiments. More controlled than resting-state data, and more engaging and ecologically valid than traditional tasks, these rich, dynamic paradigms yield data that can be analyzed in many different ways to answer a variety of scientific questions. However, unlike resting-state and task-based designs, there are currently no standard methods to analyze data from naturalistic experiments. While traditional methods (e.g., univariate GLM and functional connectivity) can be used, naturalistic designs open up the possibility for new opportunities for methodological innovation that can take advantage of the unique features of these data. Unfortunately, most of the most widely used neuroimaging analysis toolboxes do not currently include any of the analysis techniques that are being used to analyze naturalistic data. 

To meet this need, we have created [naturalistic-data.org](http://naturalistic-data.org/) an online book that includes a collection of the state of the art techniques used in analyzing naturalistic data. Most of the techniques were developed in individual labs (including some of the contributors to this project). We have prepared a collection of interactive tutorials that provide background information and walkthroughs of how to perform the analytic technique on two different open datasets (*Sherlock* from [Chen et al., 2017](https://www.nature.com/articles/nn.4450) and *Paranoia* from [Finn et al., 2018](https://www.nature.com/articles/s41467-018-04387-2)) using open science tools developed within the Python and R programming languages. Most of the tutorials also include videos explaining the theory or applications of the technique that complement the more technical hands on tutorials. All of the videos can also be viewed separately on our Naturalistic Data Analysis [youtube channel](https://www.youtube.com/playlist?list=PLbaGqHoEYoN8Le4l2-hx5zYA94UZ6A7kF). This resource was developed as a new format for a full day Educational Course for OHBM 2020 by [Luke Chang](http://cosanlab.com/), [Emily Finn](https://esfinn.github.io/), [Jeremy Manning](http://www.context-lab.com/), and [Tor Wager](https://sites.dartmouth.edu/canlab/) who are all professors at Dartmouth College in the [Department of Psychological and Brain Sciences](https://pbs.dartmouth.edu/). We have an additional 10 contributors from all over the world that are emerging as leaders in this nascent field. We hope to continue adding content as new analysis methods are developed in the field and welcome [contributions](http://naturalistic-data.org/features/markdown/Contributing.html) from anyone in the form of videos, or analysis tutorials.

![ohbm](../../images/logo/Montreal_Conf_Banner.jpg)

# Overview
The course is divided into three different sections.

## Getting Started
In the **Getting Started** section, we provide tutorials for how to download the data we will be using for the tutorials with [datalad](https://www.datalad.org/). Currently, the [server](https://gin.g-node.org/ljchang) we are hosting the data on can be slow, so we encourage you to start this as soon as possible if you would like to download all of the data (+300 gb). However, different tutorials rely on different data, so you can also wait to download only the data you need for the methods you are interested in working with. We also provide instructions for how to download and install the software that will be required for various tutorials. For the most part, almost all of the tools should be platform independent (mac, pc, linux). Finally, we explain how we preprocessed the data and provide our current recommendations for preprocessing naturalistic data that might be slightly different than analyzing task-based or resting state data.

## Background Resources
As all of our tutorials introduce fairly advanced techniques using primarily Python, we have provided additional tutorials that we hope can quickly get most people up to speed to be able to follow along. These include basic introductions to Python, working with behavioral and neuroimaging data, and plotting. In addition, if you would like an even more in depth background to the basics of neuroimaging using Python, we encourage you to check out the [DartBrains.org](https://dartbrains.org) online course, which has a similar format to this one. If you already feel comfortable programming in Python, we encourage you to just dive right into the tutorials.

## Analysis Tutorials
The analysis tutorials have been designed to provide a general overview of the main issues and cutting edge techniques for analyzing naturalistic data. However, we note that most of the methods could also be used with task-based or resting-state data as well. 

### How do we build models using naturalistic designs?
One of the core issues in analyzing naturalistic data is building a model to predict brain responses. Unlike traditional task-based experimental designs it is not always known when participants on average are engaged in a specific cognitive process. We have included a tutorial on how to use the [Pliers](https://github.com/tyarkoni/pliers) toolbox ([]) from the Tal Yarkoni’s [Psychoinformatics lab](http://pilab.psy.utexas.edu/) to obtain **automated annotations** using many types of models trained to recognize objects, speech, music, and people. software as a service. 

An early and very popular technique to get around the issue of knowing what types of cognitive processes were being engaged at specific points in time was to use other people as a model. **Intersubject Correlation** (ISC) methods were developed by [Uri Hasson’s lab](https://www.hassonlab.com/) to identify neural activity that was shared across individuals ([Hasson et al., 2004](https://science.sciencemag.org/content/303/5664/1634), [Nummenmaa et al., 2018](https://linkinghub.elsevier.com/retrieve/pii/S2352-250X(18)30023-X), [Nastase et al., 2019](https://academic.oup.com/scan/article/14/6/667/5489905) ). This is currently one of the most popular analytic techniques, which has lead to many extensions including **Intersubject Functional Connectivity** (ISFC; [Simony et al., 2016](https://www.nature.com/articles/ncomms12141)), **Intersubject Phase Synchrony** (ISPS; [Glerean et al., 2012](https://www.liebertpub.com/doi/full/10.1089/brain.2011.0068?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed)) for looking at the dynamics of ISC, and **Intersubject Representational Analysis** (ISRSA; [Chen et al., 2020](http://cosanlab.com/static/papers/Chen_2020_Neuroimage.pdf), [Finn et al., 2020](https://www.sciencedirect.com/science/article/pii/S1053811920303153), [van Baar et al., 2019](https://www.nature.com/articles/s41467-019-09161-6)) for looking at individual variation. 

Most of these methods with the exception of ISRSA assume that individuals have common neural responses. However, in practice we know that there is enormous heterogeneity across individuals. Functional Alignment techniques such as **hyperalignment**([Haxby et al., 2011](https://www.sciencedirect.com/science/article/pii/S0896627311007811?via%3Dihub), [Haxby et al., 2020](https://elifesciences.org/articles/56601)) or the **shared response model** ([Chen et al., 2015](https://papers.nips.cc/paper/5855-a-reduced-dimension-fmri-shared-response-model)) aim to minimize individual variations by realigning brains into a common model based on shared neural responses.

Finally, we acknowledge the importance of models in analyzing naturalistic data, particularly in univariate encoding models ([Huth et al., 2016](https://www.nature.com/articles/nature17637?version=meter+at+3&module=meter-Links&pgtype=article&contentId=&mediaId=&referrer=&priority=true&action=click&contentCollection=meter-links-click), [Nishimoto et al., 2011](https://www.sciencedirect.com/science/article/pii/S0960982211009377), [Naselaris, et al., 2011](https://www.sciencedirect.com/science/article/pii/S1053811910010657) ), but unfortunately do not currently include any tutorials using these techniques at this time. We hope to add these in the near future and welcome contributions from others.

### How does the brain segment events?
- HMM
- GSBS

### How can we study the dynamics of brain activity?
- TimeCorr
- Hidden Semi-Markov Models
- ISFC

### What is the timescale of neural information?
- Temporal receptive windows
- ISPS

### How can we visualize high dimensional data?
- hypertools

# License for this book
All content is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0) license.

