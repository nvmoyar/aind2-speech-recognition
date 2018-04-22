# Artificial Intelligence Engineer Nanodegree

[//]: # (Image References)

[image1]: ./images/ASR_screenshot.png "ASR Screenshot"


## Different Deep Learning approaches to an Acoustic model for an ASR pipeline

### Project overview

In this notebook, some different approaches are used to build the acoustic model for an end-to-end automatic speech recognition (ASR) pipeline:

* Model 0: RNN
* Model 1: RNN + TimeDistributed Dense
* Model 2: CNN + RNN + TimeDistributed Dense
* Model 3: Deeper RNN + TimeDistributed Dense
* Model 4: Bidirectional RNN + TimeDistributed Dense
* Model 5: CNN + Deep RNN + TimeDistributed Dense 
* Models Comparison and discussion of the models 1, 2, 3, 4 and 5
* Final Model: Dilated Convolution + Deep RNN + TimeDistributed Dense 
* Discussion of final model architecture 

The first part of this notebook investigates the [LibriSpeech](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf), the dataset that will be used to train and evaluate this pipeline. The wav signal is preprocessed in order to obtain frequencies and MFCC features. The final discussion includes observations about using either tensor as input features for the pipeline. 

```sample_models.py``` module, includes code for all the models. Once this module is imported in the notebook, the different architectures are trained within the notebook. 

The second and longest part of the notebook includes discussion about the performance of all the models and compares results on using spectrogram or MFCC features.

The third part shows a predicted transcription based on the probability distribution of the chosen acoustic models, the output on the second part of the notebook. 

![ASR Screenshot][image1]

### Install environment, Test

* [Install instructions](https://github.com/udacity/AIND-VUI-Capstone)
* [Test](http://localhost:8888/notebooks/AIND-VUI-Capstone/vui_notebook.ipynb)
* [Demo](https://www.floydhub.com/nvmoyar/projects/speech-recognition)

#### Requirements

FloydHub is a platform for training and deploying deep learning models in the cloud. It removes the hassle of launching your own cloud instances and configuring the environment. For example, FloydHub will automatically set up an AWS instance with TensorFlow, the entire Python data science toolkit, and a GPU. Then you can run your scripts or Jupyter notebooks on the instance. 
For this project: 

> floyd run --mode jupyter --gpu --env tensorflow-1.0

You can see your instance on the list is running and has ID XXXXXXXXXXXXXXXXXXXXXX. So you can stop this instance with Floyd stop XXXXXXXXXXXXXXXXXXXXXX. Also, if you want more information about that instance, use Floyd info XXXXXXXXXXXXXXXXXXXXXX

#### Environments

FloydHub comes with a bunch of popular deep learning frameworks such as TensorFlow, Keras, Caffe, Torch, etc. You can specify which framework you want to use by setting the environment. Here's the list of environments FloydHub has available, with more to come!

#### Datasets 

With FloydHub, you are uploading data from your machine to their remote instance. It's a really bad idea to upload large datasets like CIFAR along with your scripts. Instead, you should always download the data on the FloydHub instance instead of uploading it from your machine.

Further Reading: [How and Why mount data to your job](https://docs.floydhub.com/guides/data/mounting_data/)

### Usage 

floyd run --gpu --env tensorflow-1.1 --data tontoninten/datasets/librispeech/1:LibriSpeech --mode jupyter

[**You only need to mount the data to your job, since the dataset may have already been uploaded by other user**](https://www.floydhub.com/search/datasets?query=LibriSpeech)

#### Output

Often you'll be writing data out, things like TensorFlow checkpoints, updated notebooks, trained models and HDF5 files. You will find all these files, you can get links to the data with:

> floyd output run_ID

### Special Thanks

We(Udacity) have borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.
