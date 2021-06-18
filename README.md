# LF-DAAE : A Disparity-Aware AutoEncoder for Light Field image compression

This is the source code to our compression model in "Learning-Based Practical Light Field Image Compression Using A Disparity-Aware Model", M. Singh, R. M. Rameshan, PCS'21.

![architecture](https://github.com/moha23/LF-DAAE/blob/main/archi.png)

## Requirements

- tensorflow-gpu 2.4 
- tensorflow-probability 0.12.1 
- tensorflow-compression 2.0b2
- tensorflow-addons 0.12

A [Singularity container](https://sylabs.io/singularity/) with all required packages is available, contact at s18002@students.iitmandi.ac.in. 

## Training and test data 

Training data should be placed in a folder, defaults to `'./train'` and can be changed via command line arguments. For training, we use 64x64 patches of the data. Each folder in `'./train'` contains 8 views belonging to the same row, each having spatial dimension 64x64x3.

```
train
  - folder1
    - 1_1.png
    - 1_2.png
      .
      .
    - 1_7.png
    - 1_8.png
  - folder2
    - 2_1.png
    - 2_2.png
      .
      .
    - 2_7.png
    - 2_8.png
```

The test images can be full sized, default directory is `'./test'`. Each folder in `'./test'` contains 64 views, or the center 8x8 views of the entire 4D light field. The views should be named as `i_j.png`, where `i` is given by the row and `j` by the column of the view with respect to entire 4D light field. 

## Usage

For training:

```
python lfdaae.py train
```

Additional arguments can be added to change default batch size, checkpoint directory, etc. To see list of available commands:

```
python lfdaae.py -h
python lfdaae.py train -h
```

To compress a light field image:

```
python lfdaae.py compress './test/examplefolder' './output'
```

This will read all 64 views in `'examplefolder'` and compress each row and save the bitstreams as a `.tfci` file in the `'./output'` folder. Reconstructions from the bistream will also be saved. 

To decompress a bitstream:
```
python lfdaae.py decompress './output/ex.tfci' './output'
```

## To do

- Update code for compatibility with TF2.5 and TFC2.2
- Add paper/video links

## Notes

Parts of the code here are borrowed from the example codes in the [Tensorflow Compression repository](https://github.com/tensorflow/compression). 

If you find it useful in your research, kindly cite our PCS'21 paper.

