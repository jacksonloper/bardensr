# BARDENSR

## What is it for?

This package is a collection of tools for dealing with spatial multiplexed data.  Specifically, we assume the following setup.

1. The tissue.  There is a two- or three-dimensional object being examined.  In deference to the spatial transcriptomics applications, this object will hereafter be called the **"tissue."**
2. The genes.  There are several different kinds of objects which manifest in the tissue (e.g. different types of rolonies in a spatial transcriptomics experiment).  Hereafter, each kind of object will be referred to as a **"gene."**  We will let J denote the total number of different genes.
3. The voxels.  The tissue is measured on a two- or three- dimensional grid.  Hereafter, each measured location will be referred to as a **"voxel."**. We will let M denote the total number of voxels.
4. The gene density.  For each voxel (m) and each gene (j), we may imagine that there is a positive number indicating something like the density of that gene within that voxel.   We will herafter refer to this as the **gene density**.
5. The frames and imagestack.  The tissue is measured several times under different conditions (e.g. with different lasers, in different rounds of a spatial transcriptomics experiment).  Hereafter, each condition will be referred to as a **"frame."**. Collectively, the measurements for all voxels under all conditions will be called the **"imagestack."**. We will let N denote the total number of frames.
6. The model.  Given the density, we assume the imagestack may be approximately modeled using the following observation model:
    1. First, apply a linear transformation to the density, independently for each genes.  This linear transformation will typically be something like a blur kernel. It will hereafter be referred to as the **"point-spread function"** and the result will be called the **"blurred density"**.
    2.  Second, apply a linear transformation to the blurred density, indepently for each voxel.  This linear transformation can be understood as an NxJ matrix.  It will hereafter be referred to as the **"codebook"** and the result will be called the **"noiseless imagestack"**.
    3.  Finally, add noise to the noiseless imagestack.

## What does it do?

Currently:

- Spot-calling
    - Given an imagestack, the codebook, and the point-spread function, attempt to guess the density.
    - Given a density, attempt to identify bumps (e.g. individual rolonies).
- Registration
    - Generate movies which can help identify if the imagestack has registration issues.
    - Find translations of an imagestack so that for each frame the same voxel corresponds to the same physical location in the tissue.
- Preprocess an imagestack (background subtraction, normalization)
- Generate figures which can help identify colorbleed in the imagestack.

We are working on a few additional algorithms for the following tasks.  These can be found if you dig into this code, but they are not really ready for public use.

- Given an imagestack, try to guess the codebook.
- Correct vignetting artifacts in an imagestack.
- Stitch several imagestacks together from different fields of view.
- Find affine transformation of an imagestack so that for each frame the same voxel corresponds to the same physical location in the tissue.

## How do I use it?

### Installation

```
pip install --upgrade git+https://github.com/jacksonloper/bardensr.git
```

### Data structures

To use this python package, you will need to store your data with the following conventions.
- An imagestack should be a numpy array of shape N x M0 x M1 x M2.  Here M0, M1, M2 refer to the spatial dimensions of the tissue.  If the tissue is measured only in two-dimensions, one can set one of these values to unity (i.e. a numpy array of shape N x M0 x M1 x 1).
- A codebook should be a numpy array of dimension N x J.

### GPUs and CPUs

The heavy lifting of this package is all performed by tensorflow.  As such, if you want to insist that the lifting is run on the GPU or CPU, you can wrap function calls with `tf.device`, using the following pattern.

```
devicename = 'GPU' # <-- or 'CPU'
with tf.device(devicename):
    foo=bardensr.bar()
```

GPUs will be faster, but if you try to process very large imagestacks you may run out of RAM.  In this case you have two options.

1. Create minitiles from your imagestack, and process each separately.  Some examples of how this can be done are given in the [example notebook](examples/basics.ipynb).
2. Use CPUs.

### How do I use it?

The public API (below) and the [example notebook](examples/basics.ipynb) should be enough to get started.   We welcome any requests or suggestions for improved documentations; submit an issue to this github repo.

## Public API

### bardensr.build_density_singleshot(imagestack,codebook,...)

A correlation-based approach to get a crude estimate of the density.  Fast and adequate for many purposes.  Does not account for a point-spread function.

```
Input:
- imagestack (N x M0 x M1 x M2 numpy array)
- codebook (N x J numpy array)
- noisefloor (a floating point number indicating your estimate for what level of signal is "noise")

Output:
- evidence_tensor (M0 x M1 x M2 x J), a crude estimate for the density
```

### bardensr.build_density_iterative(imagestack,codebook,...)

An optimization-based approach estimate the density.

```
Input:
- imagestack (N x M0 x M1 x M2 numpy array)
- codebook (N x J numpy array)
- [optional] l1_penalty (a penalty which sends spurious noise signals to zero; this should be higher if the noise-level is higher)
- [optional] psf_radius (a tuple of 3 numbers; default (0,0,0); your estimate of the psf magnitude in units of voxels for each spatial axis of the imagestack (shape of psf is assumed to be Gaussian))
- [optional] iterations (number of iterations to train; default is 100)
- [optional] estimate_codebook_gain (boolean; default False; if True, we will attempt to correct the codebook for any per-frame gains, e.g. if frame 4 of the imagestack is 10 times brighter than all other frames)
- [optional] rounds (integer default None; if provided, must divide evenly into N, and it is then assumed that the frames can be understood as R rounds of imaging with N/C channels per round)
- [optional] estimate_colormixing (boolean; default False; if True, we will attempt to correct the codebook for color bleed between channels; only works if "rounds" is provided)
- [optional] estimate_phasing (boolean; default False; if True, we will attempt to correct the codebook for incomplete clearing of signal between rounds; only works if "rounds" is provided)

Output:
- evidence_tensor (M0 x M1 x M2 x J), an estimate for the density giving rise to this imagestack
```

### bardensr.find_peaks(evidence_tensor)

```
Input:
- evidence tensor (M0 x M1 x M2 x J)
- threshold
- [optional] radius (tuple of 3 numbers; default (1,1,1); indicating the minimum possible size of a bump)

Output: bumps, a pandas dataframe with the following columns
- m0 -- where the bumps were found along the first spatial dimension
- m1 -- where the bumps were found along the second spatial dimension
- m2 -- where the bumps were found along the third spatial dimension
- j -- where the bumps were found along the gene dimension
- magnitude -- value of evidence_tensor in the middle of the bump
```

### bardensr.register_translations_lowrank(imagestack,codebook,...)

A method that uses the codebook and the model to find a translation of the imagestack which is more consistent with the observation model.

```
Input
- imagestack (N x M0 x M1 x M2 numpy array)
- codebook (N x J numpy array)
- [optional] maximum_wiggle (tuple of 3 integers; default (10,10,10); maximum possible wiggle permitted along each spatial dimension)
- [optional] niter (integer; default 50; number of rounds of gradient descent to run in estimating the registration)
```