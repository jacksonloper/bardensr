import numpy as np
import operator
import dataclasses
import math

@dataclasses.dataclass
class Rectangle:
    '''
    An n-dimensional box, beginning at start (inclusive)
    and ending at stop (exclusive).
    '''
    start: np.ndarray
    stop: np.ndarray

    def __post_init__(self):
        self.start=np.require(self.start)
        self.stop=np.require(self.stop)
        assert len(self.start.shape)==1
        assert len(self.stop.shape)==1
        assert len(self.start)==len(self.stop)
        assert (self.start<=self.stop).all()
        self._n=len(self.start)

    @property
    def is_integral(self):
        '''
        boolean -- whether this rectangle's boundaries
        are integers
        '''
        return self.start.dtype==int and self.stop.dtype==int

    @property
    def as_slices(self):
        '''
        turn this rectangle into a tuple of slices
        that could be used in indexing into a numpy array
        (as long as self.is_integral)
        '''
        assert self.start.dtype==int
        assert self.stop.dtype==int
        return tuple([slice(st,en) for (st,en) in zip(self.start,self.stop)])

    @property
    def size(self):
        '''
        returns n-vector representing size of box
        in each dimension
        '''
        return self.stop-self.start

    @property
    def center(self):
        '''
        returns middlepoint of the box
        '''
        return .5*(self.start+self.stop)

    @property
    def empty(self):
        '''
        boolean -- is the box empty?
        '''
        return (self.size==0).all()

    def filter_pointcloud(self,pc):
        '''
        returns boolean array indicating which points in pc
        are in this box.  input should be number of points x dimension
        '''

        pc=np.require(pc)
        if len(pc.shape)<2:
            raise ValueError("pc should be matrix")
        if pc.shape[1]!=self._n:
            raise ValueError("last dim of pc should be same as box's dim")

        good=np.ones(pc.shape[:-1],dtype=np.bool)
        for i in range(self._n):
            good=good&(pc[...,i]>=self.start[i])
            good=good&(pc[...,i]<self.stop[i])
        return good

    def slice_into(self,F,constant_value=0):
        '''
        returns the slice of F which lies inside this box
        '''

        if not self.is_integral:
            raise ValueError("this is not an integer box, can't slice")
        if self._n!=len(F.shape):
            raise ValueError("F and this box don't have same dimension")

        V=np.zeros(self.size,dtype=F.dtype)
        Fbox=Rectangle(np.zeros(self._n,dtype=int),np.array(F.shape,dtype=int))
        intersection=(self&Fbox)
        if intersection.empty:
            return V
        else:
            rect_in_my_coords=Rectangle(
                intersection.start-self.start,
                intersection.stop-self.start
            )
            rect_in_F_coords=intersection
            V[rect_in_my_coords.as_slices]=F[rect_in_F_coords.as_slices]
            return V

    def __mul__(self,other):
        '''
        cartesian product of this box with another
        '''
        if other is None:
            return self
        return Rectangle(
            np.r_[self.start,other.start],
            np.r_[self.stop,other.stop]
        )

    def __rmul__(self,other):
        '''
        cartesian product of this box with another
        '''
        if other is None:
            return self
        else:
            return other*self

    def __and__(self,other):
        '''
        intersection of this box with another
        '''
        st=np.array([self.start,other.start]).max(axis=0)
        en=np.array([self.stop,other.stop]).min(axis=0)

        if en.shape!=st.shape:
            raise ValueError("boxes are on different dimensions")

        if (en<=st).any():
            # no intersection
            return Rectangle(self.start,self.start)
        else:
            return Rectangle(st,en)
