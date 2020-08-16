#Plots weight and actvation frequencies from 
#caffe models

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import google.protobuf.pyext._message
import sys
import os
import numpy as np
import caffe
import matplotlib.pyplot as plt
import re
from collections import Counter
import scipy
net_name = str(sys.argv[1])
#net = caffe_pb2.NetParameter()
debug = True
caffe_root = '/home/caffe/'
model_dir = 'models/'
weight_dir = 'models/pre_trained/'
net_prototxt = caffe_root + model_dir + net_name + '_deploy.prototxt'
net_weights = caffe_root+weight_dir + net_name + '/' + net_name + '_original.caffemodel'
net_buff = caffe_pb2.NetParameter()
fn = caffe_root + model_dir + net_name + '_deploy.prototxt'
with open(fn) as t:
	s = t.read()
	txtf.Merge(s, net_buff) 

if debug:
	print 'Checking if network was already simulated... '
        net = caffe.Net(net_prototxt, net_weights, caffe.TEST)
        net.forward()
        print 'Forward pass done'
target = []
conv = []
fc = []
for l in net_buff.layer:
		if (l.type == 'Convolution'):
			target.append(l.name)
			conv.append(l.name)
for l in net_buff.layer:
		if (l.type == 'InnerProduct'):
			target.append(l.name)
			fc.append(l.name)
directory = caffe_root+ weight_dir + 'distribution/'+net_name
print directory
if not os.path.exists(directory):
        os.makedirs(directory)
print 'Weights'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['font.size'] = 14
plt.rcParams["legend.fontsize"]=12

plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams["figure.figsize"]=[5.7,3.5]
plt.autoscale()
for key in net.blobs.keys():
	
	if key in target:
		if key in conv: type = 'Convolution'
		if key in fc: type = 'Fully Connected'
		print key

		x= net.params[key][0].data.flatten()
		num = np.prod(np.shape(x))
		fig, ax1 = plt.subplots() #no overlaping

		plt.title('Layer:'+key+' Type:'+type)
		

		s=np.std(x)
		m=np.mean(x)
		
		plt.legend(title='std:'+str(s)+'\nmean:'+ str(m))

		norm = 500
		
		y, x, p =plt.hist(x,bins=int(norm))

		sort = np.sort(y)[::-1]
                
		print sort[0],sort[1]

		if sort[0]>4*sort[1]:
			plt.ylim(np.min(x),11*sort[1]/10)
		else:
			plt.ylim(np.min(x),11*sort[0]/10)
		if '/' in key:
			key.replace('/','')
			key = re.sub('/', '', key)
			print 'fixed problem', key

		print x.min(), x.max()
		print 'saving as '+directory+'/'+net_name+'_weight'+'_'+key+'.png'

                plt.savefig(directory+'/'+net_name+'_weight'+'_'+key+'.png')

print 'Activations'
for key, blob in net.blobs.items():
	if key in target:
		if key in conv: type = 'Convolution'
		if key in fc: type = 'Fully Connected'
		print key
		x = blob.data.flatten()
		fig, ax1 = plt.subplots()
		plt.suptitle('Activation:'+key+' Type:'+type)
		s=np.std(x)
		m=np.mean(x)
		plt.title('std:'+str(s)+' mean:'+ str(m),fontsize=11)
		norm = 100
		print norm
		y, x, _=plt.hist(x,bins=int(norm))

		sort = np.sort(y)[::-1]

		print sort[0],sort[1]

		if sort[0]>4*sort[1]:
			plt.ylim(np.min(x),11*sort[1]/10)
		else:
			plt.ylim(np.min(x),11*sort[0]/10)
		if '/' in key:
			key.replace('/','')
			key = re.sub('/', '', key)
			print 'fixed problem', key
		print 'saving as '+directory+'/'+net_name+'_act'+'_'+key+'.png'
		
		plt.savefig(directory+'/'+net_name+'_act'+'_'+key+'.png')


