PRIMO Similarity Search
=======================


Setup
-----
This repository comes with a Dockerfile which allows you to reproduce our
development environment. To use it, you must have a GPU-equipped server or
workstation, and the ability to download and install
[docker](https://www.docker.com/) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Note that this environment does not include the image dataset 
([OpenImages V4](https://storage.googleapis.com/openimages/web/download_v4.html)) used for
training and experiments. This dataset is publicly available, but requires over 1 terabyte of storage space, and a
significant amount of time to download. Scripts to manage the download are included in this
repository (see [Downloading Datasets](#downloading-datasets) below).

For convenience, we have pre-processed the images with VGG16-FC2 to extract feature vectors.
These feature vectors take up about 60 gigabytes and are available for download from the 
[primo-openimages](https://github.com/uwmisl/primo-openimages) repository. 

Once you have installed docker and nvidia-docker, run the following command in
this directory to build the docker image:

```
docker build -t primo .
```

Then run the following command to start the container, which will launch a
jupyter notebook server on port 8888 (use `-p PORT` to specify a different one):

```
sudo bash docker.sh -d /path/to/primo-openimages
```

Replace `/path/to/primo-openimages` with the path to the `primo-openimages` repository.


Downloading Datasets
--------------------
The [primo-openimages](https://github.com/uwmisl/primo-openimages) repository contains
the VGG16-FC2 feature vectors for the images used in our experiments. These are sufficient
to train the encoder and perform wetlab experiments, but if you wish to view the images themselves
you will need to download the original files.

If you just want to download or view a single image,
you can use its unique identifier to look up its URL, using
[this index](https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv).

If you want to download all of the images and organize them into the same sets that we used
for our experiments, open and run [this notebook](notebooks/01_datasets/01_download.ipynb).

The code used to extract the feature vectors can be found in
[this notebook](notebooks/01_datasets/02_extract_features.ipynb).


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



