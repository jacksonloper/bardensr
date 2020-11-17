import matplotlib.pylab as plt
import numpy as np

def plot_rbc(R,C,callback,sideways=False,notick=False,**kwargs):
    if sideways:
        with AnimAcross(columns=R,**kwargs) as a:
            for c in range(C):
                for r in range(R):
                    ~a 
                    if r==0:
                        plt.ylabel(f"Ch:{c}",fontsize=30)
                    if c==0:
                        plt.title(f"R:{r}",fontsize=30)
                    if notick:
                        plt.xticks([]); plt.yticks([]);

                    callback(r,c,a)
    else:
        with AnimAcross(columns=C,**kwargs) as a:
            for r in range(R):
                for c in range(C):
                    ~a
                    if r==0:
                        plt.title(f"Ch:{c}",fontsize=30)
                    if c==0:
                        plt.ylabel(f"R:{r}",fontsize=30)
                    if notick:
                        plt.xticks([]); plt.yticks([]);

                    callback(r,c,a)


def hexlbin(a,b,c=None,**kwargs):
    if c is None:
        c=np.ones(len(a))
    plt.hexbin(a,b,c,reduce_C_function=lambda x:np.log(1+np.sum(x)),**kwargs)

class AnimAcross:
    def __init__(self,ratio=.8,sz=4,columns=None,aa=None,asp=1.0):
        self.aa=aa
        self.axes_list=[]
        self.cbs={}
        self.ratio=ratio
        self.sz=sz
        self.columns=columns
        self.asp=asp

    def __enter__(self):
        if self.aa is not None:
            return self.aa
        else:
            return self

    def __invert__(self):
        self.axes_list.append(plt.gcf().add_axes([0,0,self.ratio,self.ratio],label="axis%d"%len(self.axes_list)))

    def __neg__(self):
        self.axes_list.append(plt.gcf().add_axes([0,0,self.ratio,self.ratio],label="axis%d"%len(self.axes_list)))
        plt.axis('off')

    def __call__(self,s,*args,**kwargs):
        ~self;
        plt.title(s,*args,**kwargs)

    def cb(self,mappable,idx=None):
        if idx is None:
            idx = len(self.axes_list)-1
        self.cbs[idx] = mappable

    def __exit__(self,exc_type,exc_val,exc_tb):
        if self.aa is not None:
            return

        if self.columns is None:
            dims=[
                (1,1), # no plots
                (1,1), # 1 plot
                (1,2), # 2 plots
                (1,3), # 3 plots
                (2,2), # 4 plots
                (2,3), # 5 plots
                (2,3), # 6 plots
                (3,3), # 7 plots
                (3,3), # 8 plots
                (3,3), # 9 plots
                (4,4)
            ]
            if len(self.axes_list)<len(dims):
                dims=dims[len(self.axes_list)]
            else:
                cols=int(np.sqrt(len(self.axes_list)))+1
                rows = len(self.axes_list)//cols + 1
                dims=(rows,cols)
        else:
            cols=self.columns
            if len(self.axes_list)%cols==0:
                rows=len(self.axes_list)//cols
            else:
                rows=len(self.axes_list)//cols + 1
            dims=(rows,cols)

        plt.gcf().set_size_inches(self.sz,self.sz*self.asp)
        k=0

        for j in range(dims[0]):
            for i in range(dims[1]):
                if k<len(self.axes_list):
                    self.axes_list[k].set_position((i,dims[0]-j,self.ratio,self.ratio))
                k=k+1

        for i in range(len(self.axes_list)):
            if i in self.cbs:
                plt.colorbar(mappable=self.cbs[i],ax=self.axes_list[i])

        if exc_type is not None:
            print(exc_type,exc_val,exc_tb)
