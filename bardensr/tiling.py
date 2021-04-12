import numpy as np
import scipy as sp
import itertools
from . import rectangles
import dataclasses
import math

import collections

Tile=collections.namedtuple('Tile',['look','grab','put'])

@dataclasses.dataclass
class LookGrabPut:
    look: rectangles.Rectangle
    grab: rectangles.Rectangle
    put: rectangles.Rectangle

    def __mul__(self,other):
        if other is None:
            return self
        return LookGrabPut(
            self.look*other.look,
            self.grab*other.grab,
            self.put*other.put,
        )

    def __rmul__(self,other):
        if other is None:
            return self
        else:
            return other*self



def tile_up_simple(start,stop,sz):
    '''
    batches a stretch of indices up into equally-sized
    nonoverlapping tiles which avoid the edge by "border"
    '''

    bins=np.r_[start:stop:sz]
    assert len(bins)>0

    return [rectangles.Rectangle([bins[i]],[bins[i+1]]) for i in range(len(bins)-1)]

def tile_up_simple_nd(starts,stops,szs):
    ls=[tile_up_simple(l,s,b) for (l,s,b) in zip(starts,stops,szs)]
    return [math.prod(x,start=None) for x in itertools.product(*ls)]



def tile_up(length,inner_sz,border_sz,last_edge_behavior='short'):
    '''
    batches a stretch of indices up into overlapping tiles.

    Input:
    - length
    - inner_sz
    - border_sz

    Output is a list of Tile objects

    For example, with
    - length=24
    - inner_sz=5
    - border_sz=2, the output should be 5 tiles with:

    lookblocks       grabblocks   putblocks
    [0,7]            [0,5]        [0,5]
    [3,12]           [2,7]        [5,10]
    [8,17]           [2,7]        [10,15]
    [13,22]          [2,7]        [15,20]
    [18,24]          [2,6]        [20,24]

                        1 1 1 1 1 1 1 1 1 1 2 2 2 2
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
    . . . . . . . . . . . . . . . . . . . . . . . .|
    0 1 2 3 4 5 6                                  |   look
          0 1 2 3 4 5 6 7 8                        |   blocks
                    0 1 2 3 4 5 6 7 8              |   shown
                              0 1 2 3 4 5 6 7 8    |   here
                                        0 1 2 3 4 5|


                        1 1 1 1 1 1 1 1 1 1 2 2 2 2
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
    . . . . . . . . . . . . . . . . . . . . . . . .|
    0 1 2 3 4                                      |   grab/put
              2 3 4 5 6                            |   blocks
                        2 3 4 5 6                  |   shown
                                  2 3 4 5 6        |   here
                                            2 3 4 5|



    '''

    ib = inner_sz+border_sz
    ib2 = inner_sz + border_sz*2

    if ib >= length:
        # only one tile!
        lookblocks=[(0,length)]
        grabblocks=[(0,length)]
        putblocks=[(0,length)]
    else:
        lookblocks=[]
        grabblocks=[]
        putblocks=[]

        lookblocks.append((0,ib))
        grabblocks.append((0,inner_sz))
        putblocks.append((0,inner_sz))

        def get_next_block(st):
            '''
            creates another block, with lookblock starting
            at st
            '''
            en = st+ib2
            if en>length:
                # uh oh.  this is our last tile!
                if last_edge_behavior=='reduplicate':
                    st = np.min([st,length-ib])
                    lookblocks.append((st,length))
                    grabblocks.append((border_sz,(length-st)))
                    putblocks.append((st+border_sz,length))
                elif last_edge_behavior=='short':
                    lookblocks.append((st,length))
                    grabblocks.append((border_sz,length-st))
                    putblocks.append((st+border_sz,length))
                elif last_edge_behavior=='drop':
                    pass
                else:
                    raise NotImplementedError()
                return False
            else:
                # regular old tile
                lookblocks.append((st,en))
                grabblocks.append((border_sz,ib))
                putblocks.append((st+border_sz,en-border_sz))
                return True

        while get_next_block(putblocks[-1][1]-border_sz):
            pass

    rez=[]
    for ((l0,l1),(g0,g1),(p0,p1)) in zip(lookblocks,grabblocks,putblocks):
        rez.append(LookGrabPut(
            rectangles.Rectangle([l0],[l1]),
            rectangles.Rectangle([g0],[g1]),
            rectangles.Rectangle([p0],[p1]),
        ))
    return rez

def tile_up_noborder(length,inner_sz,border_sz,last_edge_behavior='short'):
    '''
    batches a stretch of indices up into overlapping tiles,
    and refuses to consider regions that don't have suitable
    borders

    Input:
    - length
    - inner_sz
    - border_sz

    Output is a list of Tile objects

    For example, with
    - length=24
    - inner_sz=5
    - border_sz=2, the output should be 5 tiles with:

    lookblocks       grabblocks   putblocks
    [0,7]            [2,5]        [2,5]
    [3,12]           [2,7]        [5,10]
    [8,17]           [2,7]        [10,15]
    [13,22]          [2,7]        [15,20]
    [18,24]          [2,4]        [20,22]

                        1 1 1 1 1 1 1 1 1 1 2 2 2 2
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
    . . . . . . . . . . . . . . . . . . . . . . . .|
    0 1 2 3 4 5 6                                  |   look
          0 1 2 3 4 5 6 7 8                        |   blocks
                    0 1 2 3 4 5 6 7 8              |   shown
                              0 1 2 3 4 5 6 7 8    |   here
                                        0 1 2 3 4 5|


                        1 1 1 1 1 1 1 1 1 1 2 2 2 2
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
    . . . . . . . . . . . . . . . . . . . . . . . .|
        2 3 4                                      |   grab/put
              2 3 4 5 6                            |   blocks
                        2 3 4 5 6                  |   shown
                                  2 3 4 5 6        |   here
                                            2 3    |



    '''

    ib = inner_sz+border_sz
    ib2 = inner_sz + border_sz*2

    if ib >= length:
        # only one tile!
        assert (length-border_sz*2)>0,'with that much border, theres nothing left here...'
        lookblocks=[(0,length)]
        grabblocks=[(0,length-border_sz*2)]
        putblocks=[(border_sz,length-border_sz)]
    else:
        lookblocks=[]
        grabblocks=[]
        putblocks=[]

        lookblocks.append((0,ib2))
        grabblocks.append((border_sz,border_sz+inner_sz))
        putblocks.append((border_sz,border_sz+inner_sz))

        def get_next_block(st):
            '''
            creates another block, with
            - lookblock starting at st
            - putblock starting at st+border_sz
            - probably ending at st+ib2 (unless it is the last tile)
            '''
            en = st+ib2
            if en>length:
                # uh oh.  this is our last tile!
                if last_edge_behavior=='drop':
                    return False
                elif last_edge_behavior=='reduplicate':
                    st = np.min([st,length-ib2])
                    lookblocks.append((st,length))
                    grabblocks.append((border_sz,(length-st-border_sz)))
                    putblocks.append((st+border_sz,length-border_sz))
                    return False
                elif last_edge_behavior=='short':
                    lookblocks.append((st,length))
                    grabblocks.append((border_sz,length-st-border_sz))
                    putblocks.append((st+border_sz,length-border_sz))
                    return False
                else:
                    return NotImplementedError(last_edge_behavior)
            else:
                # regular old tile
                lookblocks.append((st,en))
                grabblocks.append((border_sz,ib))
                putblocks.append((st+border_sz,en-border_sz))
                return True

        while get_next_block(putblocks[-1][1]-border_sz):
            pass

    rez=[]
    for ((l0,l1),(g0,g1),(p0,p1)) in zip(lookblocks,grabblocks,putblocks):
        rez.append(LookGrabPut(
            rectangles.Rectangle([l0],[l1]),
            rectangles.Rectangle([g0],[g1]),
            rectangles.Rectangle([p0],[p1]),
        ))
    return rez


def calc_neighborhood_structure(tiles):
    n=len(tiles)
    X=np.zeros((n,n),dtype=np.bool)
    for i in range(n):
        for j in range(n):
            if not (tiles[i]&tiles[j]).empty:
                X[i,j]=True
    return X

def tile_up_nd(shp,inner_szs,border_szs=None,outer_border=True,last_edge_behavior='short'):
    '''
    Input:
    - shape
    - inner_szs (one for each dim in shape)
    - border_szs(one for each dim in shape)

    Output
    - a list of MultiTile objects
    '''

    shp=np.require(shp,dtype=int)
    inner_szs=np.require(inner_szs,dtype=int)
    if border_szs is None:
        border_szs=np.zeros(len(inner_szs),dtype=int)

    if outer_border:
        lgps = [tile_up(sh,i,b,last_edge_behavior) for (sh,i,b) in zip(shp,inner_szs,border_szs)]
    else:
        lgps = [tile_up_noborder(sh,i,b,last_edge_behavior) for (sh,i,b) in zip(shp,inner_szs,border_szs)]


    lgps=list(itertools.product(*lgps))

    return [math.prod(x,start=None) for x in lgps]
