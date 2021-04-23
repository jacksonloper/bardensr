import numpy as np
import operator
import dataclasses
import math


@dataclasses.dataclass
class Rectangle:
    start: np.ndarray
    stop: np.ndarray

    def __post_init__(self):
        self.start=np.require(self.start)
        self.stop=np.require(self.stop)
        assert len(self.start.shape)==1
        assert len(self.stop.shape)==1
        assert len(self.start)==len(self.stop)
        assert (self.start<=self.stop).all()

    @property
    def as_slices(self):
        assert self.start.dtype==int
        assert self.stop.dtype==int
        return tuple([slice(st,en) for (st,en) in zip(self.start,self.stop)])

    @property
    def size(self):
        return self.stop-self.start

    @property
    def center(self):
        return .5*(self.start+self.stop-1)

    @property
    def empty(self):
        return (self.size==0).all()

    def __mul__(self,other):
        if other is None:
            return self
        return Rectangle(
            np.r_[self.start,other.start],
            np.r_[self.stop,other.stop]
        )

    def __rmul__(self,other):
        if other is None:
            return self
        else:
            return other*self

    def __and__(self,other):
        good,st,en=rectangle_intersection(self.start,self.stop,other.start,other.stop)
        if good:
            return Rectangle(st,en)
        else:
            return Rectangle(self.start,self.start)

def rectangle_intersection(st1,en1,st2,en2):
    st=np.array([st1,st2]).max(axis=0)
    en=np.array([en1,en2]).min(axis=0)

    assert en.shape==st.shape

    if (en<=st).any():
        return False,None,None
    else:
        return True,st,en

def pointcloud_in_box_test(pts,st,en,left_closed=True,right_closed=False):
    '''
    Input
    - st n-tensor
    - en n-tensor
    - pts (batch x n)
    - left_closed = True
    - right_closed = False

    Output: boolean indicating which pts fall inside box defined by

        prod_i [st[i],en[i])

    left_closed and right_closed control the kinds of intervals considered, i.e.

    True,False  --> prod_i [st[i],en[i])
    True,True   --> prod_i [st[i],en[i]]
    False,False --> prod_i (st[i],en[i])
    False,True  --> prod_i (st[i],en[i]]
    '''

    ndim=len(st)

    c1=operator.le if left_closed else operator.lt
    c2=operator.le if right_closed else operator.lt

    pts=np.require(pts)

    good = True
    for i,(st,en) in enumerate(zip(st,en)):
        good &= c1(st,pts[:, i]) & c2(pts[:,i],en)
    return good

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

    return np.array(st),np.array(en)

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

def fill_ignoreoob(base, indices, value):
    '''
    Input:
    - base, an array (M0 x M1 x M2...Mn)
    - indices, an array (S x n)
    - value

    Sets base[indices[i]]=value, but if indices[i] is oob it is ignored
    '''

    shp = np.array(base.shape)
    good = (indices >= 0).all(axis=1) & (indices < shp[None, :]).all(axis=1)
    base[tuple(indices[good].T)] = value

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
