#Original script from https://github.com/xzhang85/CNN-Conv-BatchNorm-fusion
#This script folds the batch normalization performed by the batchnorm and scale 
#caffe layers into the respective convolution layers, for efficient inference

from __future__ import print_function
import argparse
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def get_layer_by_name(proto, name):
    for i in xrange(len(proto.layer)):
        if proto.layer[i].name == name:
            return proto.layer[i]
    return None


def get_conv_layer_name(proto, name):
    layer = get_layer_by_name(proto, name)
    if not layer:
        return None
    if layer.type == u'Scale':
        bottom_layer = get_layer_by_name(proto, layer.bottom[0])
        if bottom_layer and bottom_layer.type == u'BatchNorm':
	    # not inplace
            bottom2_layer = get_layer_by_name(proto, bottom_layer.bottom[0])
            if bottom2_layer and bottom2_layer.type == u'Convolution':
                return bottom2_layer.name
	if bottom_layer and bottom_layer.type == u'Convolution':
            return bottom_layer.name
    elif layer.type == u'BatchNorm':
        bottom_layer = get_layer_by_name(proto, layer.bottom[0])
	if bottom_layer and bottom_layer.type == u'Convolution':
            return bottom_layer.name
    return None


class Convert:
    def __init__(self, network, model):
        self.network = network
        self.model = model
        caffe.set_mode_cpu()
        self.orig_net = caffe.Net(network, model, caffe.TEST)
        self.net = self.orig_net
	
    def eliminate_bn(self):
	inplace =1
        conv_layers_bn = {}
        conv_layers_sc = {}
        proto = caffe_pb2.NetParameter()
	text_format.Merge(open(self.network).read(), proto)
	
        # change network topology
        for i in xrange(len(proto.layer)):
            layer = proto.layer[i]
            if layer.type == u'BatchNorm' or layer.type == u'Scale':
		if layer.type == u'Scale':
			check = get_layer_by_name(proto, layer.bottom[0])
			if check.type == u'BatchNorm':
				inplace=0
				print ("Not inplace")
                continue
            for j in xrange(len(layer.top)):
                conv_layer_name = get_conv_layer_name(proto, layer.top[j])
                if conv_layer_name:
                    layer.top[j] = conv_layer_name + '/fold'
            for j in xrange(len(layer.bottom)):
                conv_layer_name = get_conv_layer_name(proto, layer.bottom[j])
                if conv_layer_name:
                    layer.bottom[j] = conv_layer_name + '/fold'

      
        i = len(proto.layer)
        while i > 0:
            i -= 1
            layer = proto.layer[i]
            if layer.type == u'BatchNorm' or layer.type == u'Scale':
		
                conv_layer_name = get_conv_layer_name(proto, layer.name)
		print (conv_layer_name, layer.name, layer.type, layer.bottom[0])
                if conv_layer_name:
		    
                    if layer.type == u'BatchNorm':
		                conv_layers_bn[conv_layer_name] = layer.name
		                conv_layer = get_layer_by_name(proto, conv_layer_name)
		                conv_layer.convolution_param.bias_term = True
		                for j in xrange(len(conv_layer.top)):
		                    if conv_layer.top[j] == conv_layer.name:
		                        conv_layer.top[j] = conv_layer.top[j]+ '/fold'
				    if inplace ==1:
					    if 'conv' in conv_layer.bottom[j]:
						conv_layer.bottom[j] =conv_layer.bottom[j]+ '/fold'
		                conv_layer.name = conv_layer.name + '/fold'
                    else:
			
                        conv_layers_sc[conv_layer_name] = layer.name
                    proto.layer.remove(layer)

	    elif layer.type == u'ReLU': #TODO general fix for all inplace layers
		if inplace ==1:
			
			layer.top[j] = layer.top[j]+ '/fold'
			layer.bottom[j] = layer.bottom[j]+ '/fold'
	    else:
		conv_layer_name = get_conv_layer_name(proto, layer.name)
		conv_layer = get_layer_by_name(proto, conv_layer_name)
		
		if ('conv' in layer.bottom[j])and (not 'fold' in layer.bottom[j]):
		    layer.bottom[j] =layer.bottom[j]+ '/fold'
        outproto = self.network.replace('deploy.prototxt', 'fold_deploy.prototxt')
        outmodel = self.model.replace('.caffemodel', '_fold.caffemodel')

        with open(outproto, 'w') as f:
            f.write(str(proto))

        print ('******calc new conv weights from original conv/bn/sc weights')
        new_w = {}
        new_b = {}
        for layer in conv_layers_bn:
            old_w = self.orig_net.params[layer][0].data
            if len(self.orig_net.params[layer]) > 1:
		print (layer,'Has BIAS')
                old_b = self.orig_net.params[layer][1].data
            else:
		print (layer,'No BIAS')
                old_b = np.zeros(self.orig_net.params[layer][0].data.shape[0],
                                 self.orig_net.params[layer][0].data.dtype)
	    print (layer,'Bias is')
	    
            if self.orig_net.params[conv_layers_bn[layer]][2].data[0] != 0:
                s = 1 / self.orig_net.params[conv_layers_bn[layer]][2].data[0]
            else:
                s = 0
            u = self.orig_net.params[conv_layers_bn[layer]][0].data * s #mean
	    
            v = self.orig_net.params[conv_layers_bn[layer]][1].data * s #variance
	    print (conv_layers_sc) 
	    print ('***************')
	    print (conv_layers_bn)
	    print (layer)
            alpha = self.orig_net.params[conv_layers_sc[layer]][0].data
            beta = self.orig_net.params[conv_layers_sc[layer]][1].data
            new_b[layer] = alpha * (old_b - u) / np.sqrt(v + 1e-5) + beta
            new_w[layer] = (alpha / np.sqrt(v + 1e-5))[...,
                                                       np.newaxis,
                                                       np.newaxis,
                                                       np.newaxis] * old_w
	    print ('Bias',np.shape(new_b[layer]),'Weights',np.shape(new_w[layer]))
	
        # create new net and save new model
	
        self.net = caffe.Net(outproto, self.model, caffe.TEST)
	
	
	for layer in new_w:
            self.net.params[layer+ '/fold'][0].data[...] = new_w[layer]
	    if len(self.net.params[layer+'/fold']) > 1:
		print(layer,'BIAS')
            self.net.params[layer+ '/fold'][1].data[...] = new_b[layer]        
	self.net.save(outmodel)
        self.net = caffe.Net(outproto, outmodel, caffe.TEST)
	print ('FOLDING DONE')

    def test(self):
        np.random.seed()
        rand_image = np.random.rand(1, 3, 224, 224) * 255
        self.net.blobs['data'].data[...] = rand_image
        self.orig_net.blobs['data'].data[...] = rand_image

        # compute
        out = self.net.forward()
        orig_out = self.orig_net.forward()

        # original vs fused prediction
        print(orig_out['prob'].argmax(), 'vs', out['prob'].argmax())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert')
    parser.add_argument('-d', '--deploy', action='store', dest='deploy',
                        required=True, help='deploy prototxt')
    parser.add_argument('-m', '--model', action='store', dest='model',
                        required=True, help='caffemodel')
    parser.add_argument('-t', '--test', action='store_true', dest='test',
                        help='run test')
    args = parser.parse_args()

    net = Convert(args.deploy, args.model)

    net.eliminate_bn()
    if args.test:
        net.test()
