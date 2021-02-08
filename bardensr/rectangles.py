import numpy as np

def rectangle_intersection(st1,en1,st2,en2):
    st=np.array([st1,st2]).max(axis=0)
    en=np.array([en1,en2]).min(axis=0)

    assert en.shape==st.shape

    if (en<=st).any():
        return False,None,None
    else:
        return True,st,en


def rect2slice(st,en,integerify=True):
    if integerify:
        return tuple([slice(int(np.ceil(a)),int(np.floor(b))) for (a,b) in zip(st,en)])
    else:
        return tuple([slice(a,b) for (a,b) in zip(st,en)])

def slice2rect(*slices):
    st=[]
    en=[]

    for x in slices:
        st.append(x.start)
        en.append(x.stop)

    return st,en

def sliceit0(F,st2,en2,fill_value=0.0):
    '''
    Given a tensor F indicating the value of a function from (0,0,...) to F.shape

    find it's slice from st2 to en2, filling with "fill_value" if necessary.
    '''

    st1=np.zeros(len(F.shape),dtype=np.int)
    return sliceit(F,st1,st2,en2,fill_value=fill_value)

def scatter_ignoreoob(base,indices,values):
    '''
    Input:
    - base, an array (M0 x M1 x M2...Mn)
    - indices, an array (S x n)
    - vals, an array (S)

    Sets base[indices[i]]=vals[i], but if indices[i] is oob it is ignored
    '''

    shp=np.array(base.shape)
    good = (indices>=0).all(axis=1) & (indices<shp[None,:]).all(axis=1)
    base[tuple(indices[good].T)]=values[good]

def sliceit(F,st1,st2,en2,fill_value=0.0):
    '''
    Given a tensor F indicating the value of a function from st1 to st1+F.shape

    find it's slice from st2 to en2, filling with "fill_value" if necessary.
    '''

    st1=np.require(st1)
    st2=np.require(st2)
    en1=st1+np.r_[F.shape]
    en2=np.require(en2)

    gsize=en2-st2
    assert gsize.dtype==int
    assert F.shape==tuple(en1-st1)

    qualia,st,en = rectangle_intersection(st1,en1,st2,en2)

    if qualia:
        if fill_value is None:
            if (en-st!=st2-en2).any():
                raise Exception("Need a fill value for %s %s %s %s"%(st1,en1,st2,en2))
            rez = np.empty(gsize)
            rez[rect2slice(st-st2,en-st2)] = F[rect2slice(st-st1,en-st1)]
            return rez
        else:
            rez = np.ones(gsize)*fill_value
            rez[rect2slice(st-st2,en-st2)] = F[rect2slice(st-st1,en-st1)]
            return rez

    else:
        if fill_value is None:
            raise Exception("Need a fill value for %s %s %s %s"%(st1,en1,st2,en2))
        else:
            return np.ones(gsize)*fill_value
