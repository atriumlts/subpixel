from keras import backend as K
from keras.layers import Conv2D

"""
	Subpixel Layer as a child class of Conv2D. This layer accepts all normal
	arguments, with the exception of dilation_rate(). The argument r indicates
	the upsampling factor, which is applied to the normal output of Conv2D.
	The output of this layer will have the same number of channels as the
	indicated filter field, and thus works for grayscale, color, or as a a
	hidden layer.

	Arguments:
		*see Keras Docs for Conv2D args, noting that dilation_rate() is removed*
		r: upscaling factor, which is applied to the output of normal Conv2D

	A test is included, which performs super-resolution on the Cifar10 dataset.
	Since these images are small, only a scale factor of 2 is used. Test images
	are saved in the directory 'test_output/'. This test runs for 5 epochs,
	which can be altered in line 132. You can run this test by using the
	following commands:

	mkdir test_output
	python keras_subpixel.py	
"""


class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, c/(r*r),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], unshifted[3]/(self.r*self.r))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters']/=self.r*self.r
        config['r'] = self.r
        return config

if __name__ == "__main__":
	import keras
	from keras.models import Model
	from keras.layers import Input, Conv2D, Activation
	from keras.optimizers import Adam
	from keras.losses import mean_squared_error
	from keras.datasets import cifar10
	import skimage.io 	
	from skimage.transform import pyramid_reduce 
	import numpy as np

	#Downloading dataset, downscaling, padding
	(HDimages, ignore), (test_images, ignore) = cifar10.load_data()
	
	downscaled=np.zeros((HDimages.shape[0], 16, 16, 3))
	downscaled_test = np.zeros((test_images.shape[0], 16, 16, 3))
	for i, image in enumerate(HDimages): 
		downscaled[i,:,:,:] = pyramid_reduce(image, 2) 

	for i, image in enumerate(test_images):
		downscaled_test[i,:,:,:] = pyramid_reduce(image, 2)	

	pad = 3	
	padded = np.zeros((downscaled.shape[0], downscaled.shape[1]+2*pad, downscaled.shape[2]+2*pad, downscaled.shape[3]))
	padded_test = np.zeros((downscaled_test.shape[0], downscaled_test.shape[1]+2*pad, downscaled_test.shape[2]+2*pad, downscaled_test.shape[3]))
	for i, image in enumerate(downscaled): 
		padded[i, pad:-1*(pad), pad:-1*(pad), :] = image			
	for i, image in enumerate(downscaled_test): 
		padded_test[i, pad:-1*(pad), pad:-1*(pad), :] = image	

	#Nework architecture, including 2x subpixel layer at end
	shape = padded.shape
	
	inputs=Input(shape[1:4])
	x = Conv2D(32, (3,3), activation='relu')(inputs)
	x = Conv2D(32, (3,3), activation='relu')(x)	
	x = Subpixel(3, (3,3), 2, activation='relu')(x)
	model = Model(inputs=inputs, outputs=x)
	model.compile(optimizer = Adam(), loss = mean_squared_error)
	
	model.fit(padded, HDimages, epochs=5, validation_split=.2)

	result = model.predict(padded_test)
	print "Upscaled from (%d, %d) to (%d, %d)"%(downscaled.shape[1], downscaled.shape[2], result.shape[1], result.shape[2])
	
	for i in range(100):
		skimage.io.imsave("test_output/%04d_downscaled.jpeg"%(i), downscaled_test[i])
		skimage.io.imsave("test_output/%04d_prediction.jpeg"%(i), result[i].astype("uint8"))
		skimage.io.imsave("test_output/%04d_original.jpeg"%(i), test_images[i].astype("uint8"))
		
