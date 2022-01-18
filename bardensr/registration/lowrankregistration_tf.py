from . import translations_tf
import tensorflow as tf

def _calc_loss(X,code,t,sz):
    '''
    Proportion of the variance unexplained by using
    a single code to explain each pixel.
    '''
    tf.debugging.assert_all_finite(t,'translations have become nan; check validity of codebook')
    newX = translations_tf.floating_slices(X,t,sz,'hermite')
    dots=tf.einsum('fj,f...->...j',code,newX)

    mx = tf.reduce_max(dots,axis=-1)
    # weights = tf.nn.softmax(dots,axis=-1)
    # mx = dots*weights

    loss=-mx**2
    return loss

def _calc_scaled_loss(X,code,t,sz):
    '''
    Proportion of the variance unexplained by using
    a single code to explain each pixel.
    '''
    newX = translations_tf.floating_slices(X,t,sz,'hermite')
    dots=tf.einsum('fj,f...->...j',code,newX)

    mx = tf.reduce_max(dots,axis=-1)
    # weights = tf.nn.softmax(dots,axis=-1)
    # mx = dots*weights

    loss=-mx**2

    #######################
    # we do things as above to keep things
    # in numerical precision, but here is what we actually want...

    # full_loss = 1+tf.reduce_sum(loss)/tf.reduce_sum(X**2)
    #           = 1+tf.reduce_mean(loss)*nitems_loss / (tf.reduce_mean(X**2)*nitems_X)
    nitems_loss=tf.cast(tf.reduce_prod(tf.shape(loss)),t.dtype)
    nitems_X=tf.cast(tf.reduce_prod(tf.shape(X)),t.dtype)
    mean_x2=tf.cast(tf.reduce_mean(X**2),dtype=t.dtype)
    mean_loss=tf.cast(tf.reduce_mean(loss),dtype=t.dtype)
    full_loss = 1+ (mean_loss / mean_x2) * (nitems_loss / nitems_X)

    return full_loss

@tf.function
def _calc_loss_and_grad(X,code,t,sz):
    '''
    X -- F x M0 x M1 x ... Mn
    code -- F x J
    t -- F x n
    sz -- n
    '''

    with tf.GradientTape() as tape:
        tape.watch(t)
        loss=_calc_loss(X,code,t,sz)
    grad=tape.gradient(loss,t)

    #######################
    # we do things as above to keep things
    # in numerical precision, but here is what we actually want...

    # full_loss = 1+tf.reduce_sum(loss)/tf.reduce_sum(X**2)
    #           = 1+tf.reduce_mean(loss)*nitems_loss / (tf.reduce_mean(X**2)*nitems_X)
    nitems_loss=tf.cast(tf.reduce_prod(tf.shape(loss)),grad.dtype)
    nitems_X=tf.cast(tf.reduce_prod(tf.shape(X)),grad.dtype)
    mean_x2=tf.cast(tf.reduce_mean(X**2),dtype=grad.dtype)
    mean_loss=tf.cast(tf.reduce_mean(loss),dtype=grad.dtype)
    full_loss = 1+ (mean_loss / mean_x2) * (nitems_loss / nitems_X)
    full_grad = grad / (mean_x2 * nitems_X)

    return full_loss,full_grad