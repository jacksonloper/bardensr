import numpy as np
from .. import misc

def seek_barcodes(X,thre_onehot,proportion_good_rounds_required):
    R,C,M0,M1,M2=X.shape
    channelchoices=np.argmax(X,axis=1)
    mnC=np.mean(X,axis=1)  # (R, M0, M1, M2)
    mxC=np.max(X,axis=1)

    # get rounds which are deemed adequate.
    # channelchoice for inadequate rounds are marked with -1
    satisfactory_round=mxC>thre_onehot*mnC  # (R, M0, M1, M2)
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
        barcodes_found=misc.convert_codebook_to_onehot_form(barcodes_found)

        return barcodes_found
