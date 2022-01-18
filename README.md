# BarDensr

## What is it for?

This package is a collection of tools for dealing with spatial multiplexed data.  Specifically, we assume the following setup.

- There are `J` different "barcodes" (e.g. 300 different barcodes labeling 300 RNA transcripts)
- There is a grid of `M0 x M1 x M2` "voxels" (e.g. 2048 x 2048 x 150 voxels)
- There are `N` different "frames" (e.g. 7 imaging rounds and 4 channels = 28 frames)
- There is an unobservable `M0 x M1 x M2 x J` "density" giving a nonnegative value for each barcodes (`j`) at each voxel (`m0,m1,m2`), indicating where the "**rolonies**" are. 
- There is a `N x J` "codebook" matrix full of nonnegative numbers, indicating how much we expect a given barcode (`j`) to contribute to the observations at given frame (`n`).
- We observe a `N x M0 x M1 x M2` "imagestack" giving a nonnegative value for each frame (`n`) at each voxel (`m0,m1,m2`).
- Given the density, we assume the imagestack can be modelled by the following process: blur along the spatial dimensions, apply the codebook along the barcodes dimensions, and add noise.

## What can BarDensr do?

Currently:

- Spot-calling
    - Given an imagestack, the codebook, and the point-spread function, attempt to guess the density.
    - Given a density, attempt to identify bumps (e.g. individual rolonies).
- Registration
    - Generate movies which can help identify if the imagestack has registration issues.
    - Find transformation of an imagestack so that for each frame the same voxel corresponds to the same physical location on the slices.
- Preprocessing
    - GPU-accelerated background subtraction via Lucy-Richardson
    - Generate figures which can help identify colorbleed in the imagestack (and suggest a correction)

We are working on a few additional algorithms for the following tasks.  These can be found if you dig into this code, but they are not really ready for public use.

- Given an imagestack, try to guess the codebook.
- Correct vignetting artifacts in an imagestack.
- Stitch several imagestacks together from different fields of view.
<!-- - Find affine transformation of an imagestack so that for each frame the same voxel corresponds to the same physical location on the slices. -->
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

