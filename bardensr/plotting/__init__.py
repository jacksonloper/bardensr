import matplotlib.pylab as plt
import numpy as np
from .. import rectangles
import io
from .. import misc
import subprocess
from contextlib import contextmanager
import shnauzer
import IPython.display
import collections
import contextlib
import tempfile
import PIL

from mpl_toolkits.mplot3d import Axes3D

def savefig_PIL(format='png',bbox_inches='tight',**kwargs):
    with io.BytesIO() as f:
        plt.gcf().savefig(f,format=format,bbox_inches=bbox_inches,**kwargs)
        f.seek(0)
        s=f.read()
    return s

class AnimGif:
    def __init__(self,format='png',bbox_inches='tight',duration=250):
        self.imgs=[]
        self.format=format
        self.bbox_inches=bbox_inches
        self.duration=duration

    def __call__(self,**kwargs):
        self.imgs.append(
            savefig_PIL(format=self.format,bbox_inches=self.bbox_inches,**kwargs)
        )
        plt.clf()

    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        plt.clf()
        self.gif=gif_from_pngs(self.imgs,duration=self.duration)

    def __invert__(self):
        return IPython.display.Image(self.gif)

def labelcolor(X,max):
    '''
    Input
    - X, integer values, -1 means off
    - max, maximum value

    Output
    - Y

    for each i:
        if X[i]==-1:
            Y[i]=0
        else
            Y[i]=X[i]%(max-1)+1
    '''

    neg=(X==-1)
    X=X%(max-1)+1
    X[neg]=0
    return X

@contextmanager
def pngs_to_mp4_async(fn,fps=12):
    import ffmpeg
    args = (
        ffmpeg
        .input('pipe:', vcodec='png',r=fps)
        .output(fn, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
        yield proc

        proc.stdin.close()
        proc.wait()

@contextlib.contextmanager
def AnimMp4(fps=6):
    with tempfile.TemporaryDirectory() as dir_name:
        fn=dir_name+'movie.mp4'
        with pngs_to_mp4_async(dir_name+'movie.mp4',fps=fps) as p2ma:
            am=_AnimMp4(p2ma)
            yield am
        am._finalize(fn)

class _AnimMp4:
    def __init__(self,p2ma):
        self.p2ma=p2ma
        self.size=None

    def __call__(self,**kwargs):
        with io.BytesIO() as f:
            plt.gcf().savefig(f,format='png',bbox_inches='tight',**kwargs)
            f.seek(0)
            img=PIL.Image.open(f)

            if self.size is None:
                W,H=img.size
                W=2*(W//2)
                H=2*(H//2)
                self.size=(W,H)

            img=img.resize(self.size)

        with io.BytesIO() as f:
            img.save(f,format='png')
            f.seek(0)
            s=f.read()

        self.p2ma.stdin.write(s)
        plt.clf()

    def _finalize(self,fn):
        with open(fn,'rb') as f:
            self._s=f.read()

    def __invert__(self):
        return IPython.display.Video(
            data=self._s,
            embed=True,
            mimetype='video/mp4',
            html_attributes='loop autoplay'
        )

def plot_roc_parametersweep(params,results,force_lim=True):
    import dataclasses
    with AnimAcross() as a:
        for nm in dataclasses.fields(params[0]):
            a(nm.name)

            thisp=[getattr(x,nm.name) for x in params]
            if isinstance(thisp[0],str) or isinstance(thisp[0],bool):
                thisp=np.array(thisp)
                opts=np.unique(thisp)
                for opt in opts:
                    plt.plot(np.array(results)[thisp==opt,0],np.array(results)[thisp==opt,1],'.',label=opt)
                    plt.legend()
            else:
                a.cb(plt.scatter(np.array(results)[:,0],np.array(results)[:,1],c=thisp))
            if force_lim:
                plt.xlim(0,1)
                plt.ylim(0,1)
            plt.xlabel("FDR")
            plt.ylabel("DR")

def plotmesh(mesh,**kwargs):
    fig=plt.gcf()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(*mesh.vertices.T,triangles=mesh.faces,linewidth=0.2, antialiased=True,
                    **kwargs)


def df_to_voxeltensor(shape,df,dilation=0,use_tqdm_notebook=False):
    result=np.full(shape,-1,dtype=np.int)
    for j in misc.maybe_tqdm(np.unique(df['j']),use_tqdm_notebook):
        good=df['j']==j
        sublocs=np.array(df[good][['m0','m1','m2']])
        if dilation>0:
            thison=np.zeros(shape,dtype=np.bool)
            thison[sublocs[:,0],sublocs[:,1],sublocs[:,2]]=True
            thison=sp.ndimage.binary_dilation(thison,iterations=dilation)
            result[thison]=j
        else:
            result[sublocs[:,0],sublocs[:,1],sublocs[:,2]]=j
    return result

def label_density(density,thresh=np.inf):
    result=np.argmax(density,axis=-1)
    result_values=np.max(density,axis=-1)
    bad=result_values<thresh
    result[bad]=-1
    return result

def meanmin_plot(E1,E2,unmatched,legend=True):
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
    if len(bad)>0:
        plt.axvline(bad[0],color='black')

    # plot spot detection failure
    bad=np.where((~unmatched[srt])*(E1[srt]==np.inf))[0]
    plt.plot(bad,np.zeros(len(bad)),'rx',label=f'barcode identified, but no spots found ({len(bad)} barcodes)')
    if len(bad)>0:
        plt.axvline(bad[0],color='black')

    if legend:
        plt.legend()
    plt.ylabel("meannmin divergences")
    plt.axhline(0)

    return srt

def foc3(ctr,radius):
    plt.gca().set_xlim3d(ctr[0]-radius,ctr[0]+radius)
    plt.gca().set_ylim3d(ctr[1]-radius,ctr[1]+radius)
    plt.gca().set_zlim3d(ctr[2]-radius,ctr[2]+radius)

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

def lutup(A,B,C,D,sc=.5,normstyle='none'):
    data=np.stack([A,B,C,D],axis=0).astype(float)

    if normstyle=='each':
        other_axes = tuple(range(1, len(data.shape)))
        data-=np.min(data,axis=other_axes,keepdims=True)
        data/=np.max(data,axis=other_axes,keepdims=True)
    elif normstyle=='all':
        data-=np.min(data)
        data/=np.max(data)
    elif normstyle=='none':
        pass
    else:
        raise NotImplementedError(normstyle)

    colors=np.array([
        [1,2,4],  # BLUE!
        [1,4,2],  # GREEN!
        [3,3,1],  # YELLOWY!
        [4,2,1],  # RED!
    ])*sc

    rez = np.einsum('l...,l...c->...c',data,colors)

    rez=np.clip(rez,0,1)

    rez=(rez*255).astype(np.uint8)

    return rez

def gify(X,sc=.5,normstyle='none',duration=250):
    import PIL
    import io
    imgs=[lutup(*x,sc=sc,normstyle=normstyle) for x in X]
    imgs=[PIL.Image.fromarray(x) for x in imgs]
    with io.BytesIO() as f:
        imgs[0].save(f,save_all=True,append_images=imgs[1:],
                duration=duration,loop=0,format='gif')
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

        k=0

        for j in range(dims[0]):
            for i in range(dims[1]):
                if k<len(self.axes_list):
                    self.axes_list[k].set_position((i,dims[0]-j-1,self.ratio,self.ratio))
                k=k+1

        plt.gcf().set_size_inches(self.sz,self.sz*self.asp)

        for i in range(len(self.axes_list)):
            if i in self.cbs:
                plt.colorbar(mappable=self.cbs[i],ax=self.axes_list[i])

        if exc_type is not None:
            print(exc_type,exc_val,exc_tb)

DirResult=collections.namedtuple('DirResult',['path','subdirs','nodes'])

def get_graph_def_from_tf_concrete_function(cf):
    gd=cf.graph.as_graph_def()

    lk={x.name:i for (i,x) in enumerate(gd.node)}
    node_attributes=[{} for i in range(len(gd.node))]
    E=np.zeros((len(gd.node),len(gd.node)),dtype=np.bool)
    for nd in gd.node:
        node_attributes[lk[nd.name]]['name']=nd.name
        if hasattr(nd,'attr') and 'value' in nd.attr:
            node_attributes[lk[nd.name]]['value']='const'
        for inp in nd.input:
            E[lk[inp.split(":")[0]],lk[nd.name]]=True

    allnames=[x['name'] for x in node_attributes]

    nodes_hit=set()

    grps=collections.defaultdict(lambda: (set(),set()))
    for na in node_attributes:
        nms=na['name'].split('/')
        for i in range(len(nms)):
            a='/'.join(nms[:i])
            b='/'.join(nms[:i+1])
            if b in allnames:
                grps[a][1].add(b)
                nodes_hit.add(b)
        for i in range(len(nms)-1):
            a='/'.join(nms[:i])
            b='/'.join(nms[:i+1])
            grps[a][0].add(b)

    def walker(root=''):
        dr=grps[root]
        yield DirResult(root,*dr)
        for sub in dr[0]:
            yield from walker(sub)

    return node_attributes,E,walker

def tfgraphlook(gd):
    import pygraphviz,collections
    closed_directory_names=[]

    HG_down=collections.defaultdict(set)
    roots=set()
    leaves=set()
    lk={x.name:i for (i,x) in enumerate(gd.node)}

    cdn_lookups={}

    for nd in gd.node:
        nm=nd.name
        for cdn in closed_directory_names:
            if cdn in nd.name:
                cdn=nd.name[:nd.name.find(cdn)+len(cdn)]
                cdn_lookups[nm]=cdn
                nm=cdn

        leaves.add(nm)
        nms=nm.split("/")
        roots.add(nms[0])
        for i in range(1,len(nms)):
            a='/'.join(nms[:i])
            b='/'.join(nms[:i+1])
            HG_down[a].add(b)

    grph=pygraphviz.AGraph(directed=True)

    def addleaf(subg,nm):
        # print(nm)
        if nm in lk:
            attrs=gd.node[lk[nm]].attr
            if 'value' in attrs:
                val=attrs['value']
                if hasattr(val,'tensor'):
                    shp=[str(x).strip() for x in val.tensor.tensor_shape.dim]
                    labloo=','.join([str(x) for x in list(val.tensor.int_val)+list(val.tensor.double_val)])
                    if labloo=='':
                        labloo=f'[constant {str(shp)}]'
                    subg.add_node(nm,label=labloo)
                else:
                    subg.add_node(nm,label=str(val))
            else:
                subg.add_node(nm,label=nm.split('/')[-1])
        else:
            subg.add_node(nm,label=nm.split('/')[-1])

    def crawl(grph,nm,depth):
        if nm in HG_down: # it has descendants!
            subg=grph.add_subgraph(
                name='cluster_'+nm,
                label=nm.split('/')[-1],
                style='filled',
                fillcolor="#ccccccbb"
            )
            for subn in HG_down[nm]:
                crawl(subg,subn,depth+1)

            if nm in leaves:
                addleaf(subg,nm)
        else: # it has no descendents!
            if nm in leaves:
                addleaf(grph,nm)


    for r in roots:
        crawl(grph,r,0)

    for nd in gd.node:
        for inp in nd.input:
            a=inp.split(":")[0]
            b=nd.name
            if a in cdn_lookups:
                a=cdn_lookups[a]
            if b in cdn_lookups:
                b=cdn_lookups[b]
            grph.add_edge(a,b)

    return grph