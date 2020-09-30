PRIMO Similarity Search
=======================

This document and repository are a work in progress. Talk to callie if you have
questions.

Setup
-----
This repository comes with a Dockerfile which allows you to reproduce our
development environment. To use it, you must have a GPU-equipped server or
workstation, and the ability to download and install
[docker](https://www.docker.com/) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Note that the docker environment does not include the image datasets used for our
experiments. These are publicly available, but they require about 700 gigabytes
of storage, and take a significant amount of time to download. We have included
scripts to manage the download (see [Downloading Datasets](#downloading-datasets)
below), but you will need to ensure you have the space available.

Once you have installed docker and nvidia-docker, run the following command in
this directory to build the docker image:

```
docker build -t primo .
```

Then run the following command to start the container, which will launch a
jupyter notebook server on port 8888:

```
sudo bash docker.sh /path/to/dataset-dir
```

Replace `/path/to/dataset-dir` with a path where you have set aside the space
required to download the image datasets.



Downloading Datasets
--------------------


Model Training
--------------



Simulation
----------



Experimental Setup & Analysis
-----------------------------



Benchmarking in-silico Algorithms
---------------------------------



Plots & Figures
---------------



