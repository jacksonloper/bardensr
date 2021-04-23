from . import preprocessing_tf

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