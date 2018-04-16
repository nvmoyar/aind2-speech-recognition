[//]: # (Image References)

[image1]: ./images/ASR_screenshot.png "ASR Screenshot"


## Project Overview

In this notebook, several approaches are used to build the acoustic model for an end-to-end automatic speech recognition (ASR) pipeline:

* Model 0: RNN
* Model 1: RNN + TimeDistributed Dense
* Model 2: CNN + RNN + TimeDistributed Dense
* Model 3: Deeper RNN + TimeDistributed Dense
* Model 4: Bidirectional RNN + TimeDistributed Dense
* Model 5: CNN + Deep RNN + TimeDistributed Dense 
* Models Comparison and discussion of the models 1, 2, 3, 4 and 5
* Final Model: Dilated Convolution + Deep RNN + TimeDistributed Dense 
* Discussion of final model architecture 

![ASR Screenshot][image1]

### Install environment, Test

[Install instructions](https://github.com/udacity/AIND-VUI-Capstone)
[Test](http://localhost:4000/jekyll-uno/AIND-VUI-Capstone)
[Demo](https://www.floydhub.com/nvmoyar/projects/speech-recognition)


## Special Thanks

We have borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.
