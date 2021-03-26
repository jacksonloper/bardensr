import tensorflow as tf

def cnn_and_transpose(kernels,sizes,strides,use_batchnorm=True, final_relu=False):
    lst = []
    for i in range(len(sizes) - 1):
        lst.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if use_batchnorm:
            lst.append(torch.nn.BatchNorm1d(sizes[i + 1]))
        lst.append(torch.nn.ReLU())
    if not final_relu:
        lst = lst[:-1]

    return torch.nn.Sequential(*lst)

class ConvAndTransposeNet(tf.Module):
    def __init__(self,channels,kernelsizes,strides,name=None,batch=True):
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

        '''

        super().__init__(name=name)
        self.n_layers=len(kernelsizes)
        self.strides=strides
        self.batch=batch
        self.is_training=True

        self.fwd_biases=[]
        self.fwd_layers=[]
        for i,(k,ci,co) in enumerate(zip(kernelsizes,channels[:-1],channels[1:])):
            self.fwd_layers.append(FullSpecConv(ci,co,k))
            self.fwd_biases.append(tf.Variable(tf.zeros(co),name=f'fb{i}'))

        self.reverse_layers=[]
        self.reverse_biases=[]
        for i,(k,ci,co) in enumerate(zip(kernelsizes[::-1],channels[1:][::-1],channels[:-1][::-1])):
            self.reverse_layers.append(FullSpecConv(co,ci,k))
            self.reverse_biases.append(tf.Variable(tf.zeros(co),name=f'rb{i}'))

    @tf.Module.with_name_scope
    def run_all(self,x):
        xs=[x]
        shapes=[x.shape]
        for i in range(self.n_layers):
            x=self.fwd_layers[i].apply(x,self.strides[i],padding='SAME')
            x=x+self.fwd_biases[i]
            shapes.append(x.shape)

            if i!=self.n_layers-1:
                x=tf.nn.relu(x)

            xs.append(x)

        ys=[x]
        y=x
        for i in range(self.n_layers):
            y=self.reverse_layers[i].applyT(y,self.strides[-1-i],shapes[-2-i],'SAME')
            y=y+self.reverse_biases[i]
            if i!=self.n_layers-1:
                y=tf.nn.relu(y)

            ys.append(y)
        return xs,ys

    @tf.Module.with_name_scope
    def __call__(self,x):
        shapes=[x.shape]
        for i in range(self.n_layers):
            x=self.fwd_layers[i].apply(x,self.strides[i],padding='SAME')
            x=x+self.fwd_biases[i]
            shapes.append(x.shape)

            if i!=self.n_layers-1:
                x=tf.nn.relu(x)

        y=x
        for i in range(self.n_layers):
            y=self.reverse_layers[i].applyT(y,self.strides[-1-i],shapes[-2-i],'SAME')
            y=y+self.reverse_biases[i]
            if i!=self.n_layers-1:
                y=tf.nn.relu(y)
        return x,y

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
