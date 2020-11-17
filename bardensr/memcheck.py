import tensorflow as tf
import time
import threading
import collections
import gc
import numpy as np

def memcheck_tensors():
    from tensorflow.python.framework.ops import EagerTensor

    eagers=[]
    graphy=[]
    nalloc=collections.defaultdict(lambda: 0.0)
    for ob in gc.get_objects():
        try:
            if isinstance(ob,EagerTensor):
                eagers.append(('eg',str(ob.dtype),str(ob.device),[int(x) for x in ob.shape]))
                nalloc[ob.device]+=np.prod(ob.shape)*ob.dtype.size
            elif isinstance(ob,tf.Variable):
                eagers.append(('vr',str(ob.dtype),str(ob.device),[int(x) for x in ob.shape]))
                nalloc[ob.device]+=np.prod(ob.shape)*ob.dtype.size
            elif isinstance(ob,tf.Tensor):
                graphy.append((str(ob.dtype),str(ob.device),[int(x) for x in ob.shape]))
        except Exception as e:
            pass


    return eagers,graphy,nalloc

class MemDaemon:
    def __init__(self,checkevery=10):
        self.checkevery=checkevery
        self._KEEPGOING=False
        self._thread=None

        self.errlog=[]
        self.starttime=time.time()
        self.history=collections.defaultdict(lambda: dict(times=[],allocations=[]))

    def plot(self,minutes_ago=None):
        import matplotlib.pylab as plt
        now=time.time()
        for nm in self.history:
            plt.plot(
                (np.array(self.history[nm]['times'])-now)/60,
                np.array(self.history[nm]['allocations'])/1e9,
                'o-',
                label=nm.split('/')[-1])

        plt.axvline(0,label='now',color='red')
        plt.legend()
        plt.ylabel("gigs")
        plt.ylim(0,None)
        plt.xlabel("time (minutes)")



        if minutes_ago is not None:
            plt.xlim(-minutes_ago,None)

    def start_checking(self):
        assert self._thread is None

        def go():
            while self._KEEPGOING:
                try:
                    T=time.time()
                    e,g,nalloc=memcheck_tensors()
                    for nm in nalloc:
                        self.history[nm]['times'].append(T)
                        self.history[nm]['allocations'].append(nalloc[nm])
                except Exception as e:
                    self.errlog.append(str(e))
                time.sleep(self.checkevery)

        self._thread = threading.Thread(target=go)
        self._KEEPGOING=True
        self._thread.start()

    def stop_checking(self):
        if self._thread is not None:
            self._KEEPGOING=False
            self._thread.join()
            self._thread = None
