import numpy as np
import tensorflow as tf

class Trainer:
    def __init__(self,X,model):
        X=tf.convert_to_tensor(X,dtype=tf.float64)
        self.X=X
        self.model=model

        self.losses=[model.loss(self.X)]
        self.losses[-1]['improvement']=0
        self.losses[-1]['action']='initialization'

    def record_loss(self,nm):
        self.losses.append(self.model.loss(self.X))
        self.losses[-1]['action']=nm
        self.losses[-1]['improvement']=self.losses[-2]['loss'] - self.losses[-1]['loss']


    def update(self,nms,record_loss_every_change=False):
        for nm in nms:
            getattr(self.model,'update_'+nm)(self.X)
            if record_loss_every_change:
                self.record_loss(nm)
        if not record_loss_every_change:
            self.record_loss('endsweep: ' + '_'.join(nms))

    def train(self,nms,iters,record_loss_every_change=False,tqdm_notebook=False):
        import tqdm.notebook

        trange=range(iters)
        if tqdm_notebook:
            trange=tqdm.notebook.tqdm(trange)

        for i in trange:
            self.update(nms=nms,record_loss_every_change=record_loss_every_change)
            if tqdm_notebook:
                trange.set_description(f"{self.losses[-1]['loss']:.2e}")

    def status(self,print_bads=True):
        import matplotlib.pylab as plt
        overall_losses=[x['loss'] for x in self.losses]
        worst=np.diff(overall_losses).max()
        if worst<=0:
            print("we never went the wrong way!")
        else:
            print("we went wrong way",worst)
        plt.plot(overall_losses,'-o')
        plt.gca().set_yscale('log')

        lossinfo=self.model.loss(self.X)
        for nm in lossinfo:
            print(nm.rjust(20),f'{lossinfo[nm]:.2e}')

        if print_bads:
            bads=[x for x in self.losses if x['improvement']<0]
            for b in bads:
                print(b['action'],b['reconstruction'],b['improvement'])
