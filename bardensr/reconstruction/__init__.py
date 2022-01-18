import tensorflow as tf
import tensorflow_felzenszwalb_edt
import numpy as np
import numpy.random as npr



def TV_loss(gt, est):
    '''
    input (est and gt) are both size of (S,M0,M1,M2)
    '''
    out_list = []
    for j in range(len(est)):
        out = .5*(tf.reduce_sum(tf.math.abs(est[j]/tf.reduce_sum(est[j]) - gt[j]/tf.reduce_sum(gt[j]))))
        out_list.append(out)
    return(tf.convert_to_tensor(out_list))    


def morphedt2(f,maxout=100000.0,axes=(0,1)):
    f=tf.convert_to_tensor(f,dtype=tf.float32)
    g=tensorflow_felzenszwalb_edt.edt1d(maxout*(1-f),axes[0])[0]
    g=tensorflow_felzenszwalb_edt.edt1d(g,axes[1])[0]
    return g

def skellypad(f,maxout=100000.0,radius=20.0,smooth=1.0,axes=(0,1)):
    g=morphedt2(f,axes=axes,maxout=maxout)
    g=tf.math.sigmoid((g-radius)/smooth)

    # pad with 1s to avoid weird border effect
    paddings=np.zeros((len(f.shape),2))
    paddings[axes[0]]=(1,1)
    paddings[axes[1]]=(1,1)
    g=tf.pad(g,paddings,constant_values=1)

    # do morph
    g=morphedt2(g,axes=axes,maxout=maxout)

    # unpad
    happyslice=[slice(0,None) for i in range(len(f.shape))]
    happyslice[axes[0]]=slice(1,-1)
    happyslice[axes[1]]=slice(1,-1)
    g=g[tuple(happyslice)]

    # done
    return g


def skelly(f,maxout=100000.0,radius=20.0,smooth=1.0,axes=(0,1)):
    with tf.name_scope("skelly"):
        g=morphedt2(f,axes=axes,maxout=maxout)
        g=tf.math.sigmoid((g-radius)/smooth)
        g=morphedt2(g,axes=axes,maxout=maxout)
        return g

class EDTConvAndTransposeNet(tf.Module):
    def __init__(self,n_rads,*args,name=None,init_rad=20,init_smooth=20,**kwargs):
        super().__init__(name=name)
        self.n_rads=n_rads
        self.rads=tf.Variable(npr.rand(n_rads).astype(np.float32)+init_rad)
        self.smooths=tf.Variable(npr.rand(n_rads).astype(np.float32)+init_smooth)
        self.catn=ConvAndTransposeNet(*args,**kwargs)

    def apply_edt(self,batch): # 
        axes=list(range(1,len(batch.shape)))  # 1,2,3
        lst=[batch]
        for i in range(self.n_rads):
            lst.append(skelly(batch,radius=self.rads[i],smooth=self.smooths[i],axes=axes))
        return tf.stack(lst,axis=-1)  # now the channle is last!

    def __call__(self,x):
        x=self.apply_edt(x)  # (S,channel,M0, M1,M2)
        fui,fuo=self.catn.run_all(x)
        return fui[-1],fuo[-1]

class ConvAndTransposeNet(tf.Module):
    def __init__(self,channels,kernelsizes,strides,name=None,batch=True,final_channel=None,
                                middle_relu=False,middle_skipconnection=False):
        '''
        Two networks.  First network looks like this

            batch x M0 x M1 x M2 x channels[0]
        --> conv(kernelsizes[0],strides[0],channels[1])
        --> relu
        --> conv(kernelsizes[1],strides[1],channels[2])
        --> relu
        ...
        --> conv(kernelsizes[-1],strides[-1],channels[-1])

        Second network applies the same in reverse
            batch x M0 x M1 x M2 x channels[-1]
        --> convT(kernelsizes[-1],strides[-1],channels[-2])
        --> relu
        --> convT(kernelsizes[-2],strides[-2],channels[-3])
        --> relu
        ...
        --> convT(kernelsizes[0],strides[0],channels[0])

        Second network also gets passthroughs from
        the first network

        '''

        super().__init__(name=name)
        self.n_layers=len(kernelsizes)
        self.strides=strides
        self.batch=batch
        self.middle_relu=middle_relu
        self.middle_skipconnection=middle_skipconnection
        self.is_training=True
        if final_channel is None:
            final_channel=channels[0]
        self.final_channel=final_channel

        self.fwd_biases=[]
        self.fwd_layers=[]
        for i,(k,ci,co) in enumerate(zip(kernelsizes,channels[:-1],channels[1:])):
            self.fwd_layers.append(FullSpecConv(ci,co,k))
            self.fwd_biases.append(tf.Variable(tf.zeros(co),name=f'fb{i}'))


        self.matmul_variables=[]
        self.reverse_layers=[]
        self.reverse_biases=[]
        out_channels=list(channels)
        out_channels[0]=final_channel
        for i,(k,ci,co,co2) in enumerate(zip(kernelsizes[::-1],out_channels[1:][::-1],out_channels[:-1][::-1],channels[:-1][::-1])):
            self.reverse_layers.append(FullSpecConv(co,ci,k))
            self.reverse_biases.append(tf.Variable(tf.zeros(co),name=f'rb{i}'))
            if self.middle_skipconnection or i>0:
                self.matmul_variables.append(tf.Variable(tf.zeros((co2,co)),name=f'mm{i}'))


    @tf.Module.with_name_scope
    def run_all(self,x):
        xs=[x]
        shapes=[x.shape]
        for i in range(self.n_layers):
            with tf.name_scope(f'fwd_{i}'):
                x=self.fwd_layers[i].apply(x,self.strides[i],padding='SAME')
                x=x+self.fwd_biases[i]
                shapes.append(x.shape)

                if self.middle_relu or (i!=self.n_layers-1):
                    x=tf.nn.relu(x)

                xs.append(x)

        final_shapes=list(shapes)
        final_shapes[0]=final_shapes[0][:-1] + (self.final_channel,)
        ys=[x]
        y=x

        for i in range(self.n_layers):

            with tf.name_scope(f'rev_{i}'):
                y=self.reverse_layers[i].applyT(y,self.strides[-1-i],final_shapes[-2-i],'SAME')
                y=y+self.reverse_biases[i]

                if self.middle_skipconnection:
                    y=y+tf.linalg.matmul(xs[-2-i],self.matmul_variables[i])
                elif i>0:
                    y=y+tf.linalg.matmul(xs[-2-i],self.matmul_variables[i-1])

                if i!=self.n_layers-1:
                    y=tf.nn.relu(y)

                ys.append(y)
        return xs,ys

    def __call__(self,x):
        fui,fuo=self.run_all(x)
        return fui[-1],fuo[-1]

class FullSpecConv(tf.Module):
    '''
    Similar to tf.keras.layers.Conv3D, except

    - no bias
    - no activation
    - you specify the strides each time you call it
    - you specify the padding strategy each time you call it
    - you specify whether you want the transpose each time you call it
    - if transposed, you specify the output shape each time you call it
    - dilation is forbidden
    '''

    def __init__(self,in_channels,out_channels,kernel_size,name=None):
        super().__init__(name=name)
        self.nd=len(kernel_size)
        assert self.nd in [1,2,3]

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conver=[tf.nn.conv1d,tf.nn.conv2d,tf.nn.conv3d][self.nd-1]
        self.converT=[tf.nn.conv1d_transpose,tf.nn.conv2d_transpose,tf.nn.conv3d_transpose][self.nd-1]

        mult=tf.math.sqrt(tf.cast(tf.reduce_prod(kernel_size)*in_channels,dtype=tf.float32))
        self.kernel=tf.Variable(tf.random.normal(kernel_size+(in_channels,out_channels))/mult)


    def apply(self,x,strides,padding):
        return self.conver(x,self.kernel,(1,)+strides+(1,),padding)

    def applyT(self,x,strides,output_shape,padding):
        '''
        Input:
        - X            -- batch x M0 x M1 x M2 x in_channels
        - strides      -- 3
        - output shape -- 3
        - padding      -- 'same' or 'valid'

        Output: batch x [output_shape] x out_channels
        '''

        return self.converT(x,self.kernel,output_shape,(1,)+strides+(1,),padding)
