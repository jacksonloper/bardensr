import numpy as np
import scipy as sp
import itertools
from . import rectangles
import dataclasses

import collections

Tile=collections.namedtuple('Tile',['look','grab','put'])

@dataclasses.dataclass
class MultiTile:
    look: tuple
    grab: tuple
    put: tuple

def downsample_multitile(mt,ds):
    return MultiTile(
        tuple([slice(x.start//d,x.stop//d) for x,d in zip(mt.look,ds)]),
        tuple([slice(x.start//d,x.stop//d) for x,d in zip(mt.grab,ds)]),
        tuple([slice(x.start//d,x.stop//d) for x,d in zip(mt.put,ds)]),
    )

def tiles2multitiles(*tiles):
    return MultiTile(
        tuple([x.look for x in tiles]),
        tuple([x.grab for x in tiles]),
        tuple([x.put for x in tiles])
    )

def tile_up(length,inner_sz,border_sz):
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
        return [Tile(slice(0,length),slice(0,length),slice(0,length))]
    else:
        lookblocks=[]
        grabblocks=[]
        putblocks=[]

        lookblocks.append(slice(0,ib))
        grabblocks.append(slice(0,inner_sz))
        putblocks.append(slice(0,inner_sz))

        def get_next_block(st):
            '''
            creates another block, with lookblock starting
            at st
            '''
            en = st+ib2
            if en>length:
                # uh oh.  this is our last tile!
                lookblocks.append(slice(st,length))
                grabblocks.append(slice(border_sz,length-st))
                putblocks.append(slice(st+border_sz,length))
                return False
            else:
                # regular old tile
                lookblocks.append(slice(st,en))
                grabblocks.append(slice(border_sz,ib))
                putblocks.append(slice(st+border_sz,en-border_sz))
                return True

        while get_next_block(putblocks[-1].stop-border_sz):
            pass

        return [Tile(*x) for x in zip(lookblocks,grabblocks,putblocks)]

def calc_neighborhood_structure(tiles):
    n=len(tiles)
    X=np.zeros((n,n),dtype=np.bool)
    for i in range(n):
        for j in range(n):
            st1,en1=rectangles.slice2rect(*tiles[i].look)
            st2,en2=rectangles.slice2rect(*tiles[j].look)
            if rectangles.rectangle_intersection(st1,en1,st2,en2)[0]:
                X[i,j]=True
    return X


def tile_up_nd(shp,inner_szs,border_szs):
    '''
    Input:
    - shape
    - inner_szs (one for each dim in shape)
    - border_szs(one for each dim in shape)

    Output
    - a list of MultiTile objects
    '''
    lgps = [tile_up(sh,i,b) for (sh,i,b) in zip(shp,inner_szs,border_szs)]
    tiles=list(itertools.product(*lgps))
    mt=[tiles2multitiles(*x) for x in tiles]
    return mt
