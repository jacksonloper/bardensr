import matplotlib.pylab as plt
import numpy as np
from . import rectangles
import io

from mpl_toolkits.mplot3d import Axes3D

def savefig_PIL(format='png',bbox_inches='tight',**kwargs):
    with io.BytesIO() as f:
        plt.savefig(f,format=format,bbox_inches=bbox_inches,**kwargs)
        f.seek(0)
        s=f.read()
    return s

def plotmesh(mesh,**kwargs):
    fig=plt.gcf()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(*mesh.vertices.T,triangles=mesh.faces,linewidth=0.2, antialiased=True,
                    **kwargs)


def meanmin_plot(E1,E2,unmatched):
    srt=np.argsort(E1)

    nothingfound=(E1==np.inf)
    nothingcouldbefound=unmatched

    J=len(E1)

    # plot mmds errors
    plt.plot(range(J),E1[srt],'.',label='failure to cover')
    plt.plot(range(J),-E2[srt],'.',label='failure to be contained')

    # plot total detection failures
    bad=np.where(unmatched[srt])[0]
    plt.plot(bad,np.zeros(len(bad)),'C2x',label=f'this barcode was not identified ({len(bad)} examples)')

    # plot spot detection failure
    bad=np.where((~unmatched[srt])*(E1[srt]==np.inf))[0]
    plt.plot(bad,np.zeros(len(bad)),'rx',label=f'barcode identified, but no spots found ({len(bad)} barcodes)')

    plt.legend()
    plt.ylabel("meannmin divergences")
    plt.axhline(0)

def eqaq3(ax=None):
    if ax is None:
        ax=plt.gca()
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def bytes2PIL(s,**kwargs):
    import PIL
    with io.BytesIO() as f:
        f.write(s)
        f.seek(0)
        img=PIL.Image.open(f,**kwargs).copy()
    return img

def gif_from_pngs(pngs,duration=250):
    import PIL
    imgs=[bytes2PIL(x) for x in pngs]
    with io.BytesIO() as f:
        imgs[0].save(f,save_all=True,append_images=imgs[1:],
                duration=duration,loop=0,format='gif')
        f.seek(0)
        s=f.read()
    return s

def focpt(m1,m2,bc,radius=10,j=None,X=None,**kwargs):
    if X is None:
        X=bc.X
    R,C=X.shape[:2]

    if j is not None:
        code=bc.codebook[:,:,j]
    else:
        code=None

    def go(r,c,a):

        m1s=bc.rolonies['m1']-(m1-radius)
        m2s=bc.rolonies['m2']-(m2-radius)
        goodies = (m1s>=0)&(m2s>=0)&(m1s<radius*2)&(m2s<radius*2)
        m1s=m1s[goodies]
        m2s=m2s[goodies]
        js=bc.rolonies['j'][goodies]
        for (mm1,mm2,j) in zip(m1s,m2s,js):
            if bc.codebook[r,c,j]:
                plt.annotate(str(j),[mm2,mm1],color='white',
                            bbox=dict(fc="k", ec="b", lw=2,alpha=.4))
                plt.scatter([mm2],[mm1],color='red',marker='x')

        # if r==0 and c==0:
        # print(bc.rolonies[goodies].to_markdown())

        sub=rectangles.sliceit0(X[r,c,0],[m1-radius,m2-radius],[m1+radius+1,m2+radius+1])
        plt.imshow(sub)
        plt.title(f'mx={sub.max():.2f}')
        if r==R-1:
            plt.yticks([0,radius,radius*2],[str(m1-radius),str(m1),str(m1+radius)])
            plt.gca().yaxis.tick_right()
        else:
            plt.yticks([])

        if c==C-1:
            plt.xticks([0,radius,radius*2],[str(m2-radius),str(m2),str(m2+radius)])
        else:
            plt.xticks([])


        plt.axhline(radius,color='green')
        plt.axvline(radius,color='green')

        if (code is not None) and code[r,c]:
            plt.axhline(radius,color='red')
            plt.axvline(radius,color='red')

    plot_rbc(R,C,go,**kwargs)

def lutup(A,B,C,D,sc=.5,normeach=False):
    A=A.copy()
    B=B.copy()
    C=C.copy()
    D=D.copy()
    if normeach:
        for x in [A,B,C,D]:
            x[:]=x[:]-np.min(x[:])
            x[:]=x[:]/np.max(x[:])
    else:
        mn=np.inf
        for x in [A,B,C,D]:
            mn=np.min([mn,np.min(x)])
        for x in [A,B,C,D]:
            x[:]=x-mn
        mx=-np.inf
        for x in [A,B,C,D]:
            mx=np.max([mx,np.max(x)])
        for x in [A,B,C,D]:
            x[:]=x/mx
    colors=np.array([
        [1,2,4],  # BLUE!
        [1,4,2],  # GREEN!
        [3,3,1],  # YELLOWY!
        [4,2,1],  # RED!
    ])*sc
    rez=np.zeros(A.shape+(3,))
    rez=rez+A[:,:,None]*colors[0][None,None,:]
    rez=rez+B[:,:,None]*colors[1][None,None,:]
    rez=rez+C[:,:,None]*colors[2][None,None,:]
    rez=rez+D[:,:,None]*colors[3][None,None,:]
    rez=np.clip(rez,0,1)
    rez=(rez*255).astype(np.uint8)
    return rez


def gify(X,sc=.5,normeach=False):
    import PIL
    import io
    imgs=[lutup(*x,sc=sc,normeach=normeach) for x in X]
    imgs=[PIL.Image.fromarray(x) for x in imgs]
    with io.BytesIO() as f:
        imgs[0].save(f,save_all=True,append_images=imgs[1:],
                duration=250,loop=0,format='gif')
        f.seek(0)
        s=f.read()
    return s

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

    def __pos__(self):
        self.axes_list.append(
        plt.gcf().add_axes(
            [0,0,self.ratio,self.ratio],
            label="axis%d"%len(self.axes_list),
            projection='3d',
        ))

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
