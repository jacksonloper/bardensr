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
