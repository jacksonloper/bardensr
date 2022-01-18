__all__=[
    'minmax',
    'background_subtraction',
    'colorbleed_plot',
]


import numpy as np
from . import preprocessing_tf

def colorbleed_plot(framea,frameb):
    '''
    Generates a plot which may help determine if there is colorbleed between two frames.

    Input:

    - framea (M0 x M1 x M2)
    - frameb (M0 x M1 x M2)

    '''

    import matplotlib.pylab as plt


    plt.hexbin(framea.ravel(),frameb.ravel(),np.ones(np.prod(framea.shape)),
            reduce_C_function = lambda x: np.log(np.sum(x)),
            gridsize=50)

def minmax(imagestack):
    '''
    Performs a simple per-frame normalization on the imagestack
    (subtract min, then divide by mean).

    Input should be an imagestack of shape N x M0 x M1 x M2
    '''

    assert len(imagestack.shape)==4

    return preprocessing_tf.minmax(imagestack,[1,2,3]).numpy()

def background_subtraction(imagestack,sigmas):
    '''
    Perform a dead-basic background subtraction
    (subtracts a blurred version of the imagestack, and clips to stay positive).

    Input:

    - imagestack, N x M0 x M1 x M2
    - sigmas, 3 float-ish numbers indicating how much to blur along
      last 3 axes of the imagestack
    '''

    assert len(imagestack.shape)==4


    return preprocessing_tf.background_subtraction(imagestack,[1,2,3],sigmas).numpy()

def mnmx(X,axes):
    '''
    Input
    - X -- M0 x M1 x M2 x ... x M(n-1)
    - axes -- set of integers in {0,1,...n-1}

    normalize by min and max along axes
    '''

    return preprocessing_tf.mnmx(X,axes).numpy()

def mnmx_background_subtraction(X,axes,sigmas):
    '''
    Input
    - X -- M0 x M1 x M2 ... x M(n-1)
    - axes -- set of integers in {0,1,...n-1}
    - blurs -- corresponding floating points

    this
    1. runs gaussian background subtraction along axes sigmas
    2. normalizes by min and max along axes
    '''

    return preprocessing_tf.mnmx_background_subtraction(X,axes).numpy()