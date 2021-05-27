# BarDensr

## What is it for?

This package is a collection of tools for dealing with spatial multiplexed data.  Specifically, we assume the following setup.

1. The samples.  There is a two- or three-dimensional object being examined.  In deference to the spatial transcriptomics applications, this object will hereafter be called the **"samples."** (*this could be tissues, or slide from the cell culture, as in Jellerts lab applications*) 
2. The barcodes.  There are several different kinds of objects which manifest in the tissue (e.g. different types of rolonies in a spatial transcriptomics experiment).  Hereafter, each kind of object will be referred to as a **"barcodes."**  We will let J denote the total number of different barcodes.*note that barcodes can correspond to genes, but not necessary, for instance in the cellular barcoding applications barcodes are nucleotide sequence but not genes.*
    2.  Second, apply a linear transformation to the blurred density, indepently for each voxel.  This linear transformation can be understood as an NxJ matrix.  It will hereafter be referred to as the **"codebook"** and the result will be called the **"noiseless imagestack"**.
    3.  Finally, add noise to the noiseless imagestack.

Put another way, slightly more concisely:
- There are J different "barcodes" (e.g. 300 different barcodes labeling 300 RNA transcripts)
- There is a grid of M0 x M1 x M2 "voxels" (e.g. 2048 x 2048 x 150 voxels)
- There are N different "frames" (e.g. 7 imaging rounds and 4 channels = 28 frames)
- There is an unobservable M0 x M1 x M2 x J "density" giving a nonnegative value for each barcodes at each voxel
- There is a NxJ "codebook" matrix full of nonnegative numbers, indicating how much we expect a given barcode (j) to contribute to the observations at given frame (f).
- We observe a N x M0 x M1 x M2 "imagestack" giving a nonnegative value for each frame at each voxel
- Given the density, we assume the imagestack can be modelled by the following process: blur along the spatial dimensions, apply the codebook along the barcodes dimensions, and add noise.

## What can BarDensr do?

Currently:

- Spot-calling
    - Given an imagestack, the codebook, and the point-spread function, attempt to guess the density.
    - Given a density, attempt to identify bumps (e.g. individual rolonies).
- Registration
    - Generate movies which can help identify if the imagestack has registration issues.
    - Find translations of an imagestack so that for each frame the same voxel corresponds to the same physical location in the tissue.
- Preprocessing
    - GPU-accelerated background subtraction via Lucy-Richardson
    - Generate figures which can help identify colorbleed in the imagestack (and suggest a correction)

We are working on a few additional algorithms for the following tasks.  These can be found if you dig into this code, but they are not really ready for public use.

- Given an imagestack, try to guess the codebook.
- Correct vignetting artifacts in an imagestack.
- Stitch several imagestacks together from different fields of view.
- Find affine transformation of an imagestack so that for each frame the same voxel corresponds to the same physical location in the tissue.
- Attempt to reconstruct cell morphology from a density.

## How do I use it?

### Installation

```
pip install --upgrade git+https://github.com/jacksonloper/bardensr.git
```

### Data structures

To use this python package, you will need to store your data with the following conventions.
- An imagestack should be a numpy array of shape N x M0 x M1 x M2.  Here M0, M1, M2 refer to the spatial dimensions of the tissue.  If the tissue is measured only in two-dimensions, one can set one of these values to unity (i.e. a numpy array of shape N x M0 x M1 x 1).
- A codebook should be a numpy array of dimension N x J.

### Documentation of functionality

The public API (at [readthedocs](http://bardensr.readthedocs.io)) and the [example notebook](https://github.com/jacksonloper/bardensr/blob/master/examples/basics.ipynb) should be enough to get started.   We welcome any requests or suggestions for improved documentations; submit an issue to this github repo.

## FAQ

### How do I make bardensr use GPUs?  How do I  make it use CPUs?

The heavy lifting of this package is all performed by tensorflow.  As such, if you want to insist that the lifting is run on a GPU or CPU, you can wrap function calls with `tf.device`.  The simplest version of this pattern is as follows:

```
devicename = 'GPU' # <-- or 'CPU'
with tf.device(devicename):
    interesting_result=bardensr.foo(my_cool_data)
```

### I ran out of RAM :(

Several options for dealing with memory limitations --

1. Create minitiles from your imagestack, and process each separately.
2. Use CPU (GPUs almost always have less RAM).
3. Use lower precision (e.g. convert numpy array to float16)
4. Use a bigger machine!

