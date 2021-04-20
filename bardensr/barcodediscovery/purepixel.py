import numpy as np
from .. import misc

def seek_barcodes(X,thre_onehot,proportion_good_rounds_required):
    R,C,M0,M1,M2=X.shape
    channelchoices=np.argmax(X,axis=1)
    meanC=np.mean(X,axis=1)
    mxC=np.max(X,axis=1)

    # get rounds which are deemed adequate.
    # channelchoice for inadequate rounds are marked with -1
    satisfactory_round=mxC>thre_onehot*meanC
    channelchoices[~satisfactory_round]=-1

    # count how many good rounds we have for each pixel location
    scores=np.sum(satisfactory_round,axis=0)

    # collect the good ones
    good=(scores>=proportion_good_rounds_required*R).ravel()
    if np.sum(good)==0:
        return np.zeros((R,C,0),dtype=np.float)
    else:
        # deduplicat
        channelchoicesfl=channelchoices.reshape((R,-1))
        barcodes_found=np.unique(channelchoicesfl[:,good],axis=-1).T
        # convert to a normal codebook form
        barcodes_found=misc.convert_codebook_to_onehot_form(barcodes_found, C=C)

        return barcodes_found

def calc_signalprops(X,thre_onehot=1.0,use_l2=False,lowg=.01,normX=True):
    if normX:
        X=X/X.max()
    R,C,M0,M1,M2=X.shape
    channelchoices=np.argmax(X,axis=1) # R x M0 x M1 x M2
    meanC=np.mean(X,axis=1)
    mxC=np.max(X,axis=1) # R x M0 x M1 x M2

    # get rounds which are deemed adequate.
    # channelchoice for inadequate rounds are marked with -1
    satisfactory_round=mxC>thre_onehot*meanC  # R x M0 x M1 x M2
    channelchoices[~satisfactory_round]=-1 # inadequate rounds get -1s

    # zero out action deemed unsatisfactory
    mxC[~satisfactory_round]=0

    # sum all action and compare it to total action
    if use_l2:
        ratios=np.sum(mxC**2,axis=0) / (lowg+np.sum(X**2,axis=(0,1)))
    else:
        ratios=np.sum(mxC,axis=0) / (lowg+np.sum(X,axis=(0,1)))

    return ratios,channelchoices


def seek_barcodes_by_signalprop(X,r2_thre,thre_onehot=1.0,use_l2=True,lowg=.001,maxnorm_signalprops=True):
    R,C,M0,M1,M2=X.shape
    ratios,channelchoices=calc_signalprops(X,thre_onehot=thre_onehot,use_l2=use_l2,lowg=lowg)

    if maxnorm_signalprops:
        ratios=ratios/ratios.max()

    # collect the good ones
    good=(ratios>=r2_thre).ravel() # M0*M1*M2 vector
    if np.sum(good)==0:
        return np.zeros((R,C,0),dtype=np.float)
    else:
        # trivial deduplication
        channelchoicesfl=channelchoices.reshape((R,-1))
        barcodes_found=np.unique(channelchoicesfl[:,good],axis=-1).T

        # convert to a normal codebook form
        barcodes_found=misc.convert_codebook_to_onehot_form(barcodes_found)

        return barcodes_found
