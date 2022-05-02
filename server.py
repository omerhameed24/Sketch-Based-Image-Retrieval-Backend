import numpy as np
from pylab import *
import matplotlib.pyplot as plt  
import os
import sys
from flask import Flask, request,send_from_directory
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
import shutil, os
from glob import glob
import base64
from PIL import Image
from io import BytesIO


caffe_root = "/home/omer/caffe"
sys.path.insert(0, caffe_root+'/python')
import caffe 

#app = Flask(__name__)
app = Flask(__name__, static_url_path='')

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory('images', path)
api = Api(app)

CORS(app)

print("omer")

PRETRAINED_FILE = '/home/omer/sketchy/models/triplet_googlenet/triplet_googlenet_finegrain_final.caffemodel' 
sketch_model = '/home/omer/sketchy/models/triplet_googlenet/Triplet_googlenet_sketchdeploy.prototxt'
image_model = '/home/omer/sketchy/models/triplet_googlenet/Triplet_googlenet_imagedeploy.prototxt'

caffe.set_mode_cpu()
sketch_net = caffe.Net(sketch_model, PRETRAINED_FILE, caffe.TEST)
img_net = caffe.Net(image_model, PRETRAINED_FILE, caffe.TEST)
sketch_net.blobs.keys()

#TODO: set output layer name. You can use sketch_net.blobs.keys() to list all layer
output_layer_sketch = 'pool5/7x7_s1_s'
output_layer_image = 'pool5/7x7_s1_p'

#set the transformer
transformer = caffe.io.Transformer({'data': np.shape(sketch_net.blobs['data'].data)})
transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_transpose('data',(2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)


photo_paths = '/home/omer/sketchy/rendered_256x256/256x256/photo/tx_000100000000/'
with open('/home/omer/Downloads/info-06-04/info/test_img.txt','r') as my_file:
    img_list = [c.rstrip() for c in my_file.readlines()]
print(img_list)


#img_list = os.listdir(photo_paths)
#print(img_list)
N = np.shape(img_list)[0]
print ('Retrieving from '+ str(N)+' photos')



feats = []
for i,path in enumerate(img_list):
    imgname = path.split('/')[-1]
    
    
    imgname = imgname.split('.jpg')[0]
    
    imgcat = path.split('/')[0]
    print ('\r'+ str(i+1) +'/'+str(N)+ ' '+'Extracting ' +path+'...')
    full_path = photo_paths + path
    
    img = (transformer.preprocess('data', caffe.io.load_image(full_path.rstrip())))
    img_in = np.reshape([img],np.shape(sketch_net.blobs['data'].data))
    out_img = img_net.forward(data=img_in)
    out_img = np.copy(out_img[output_layer_image]) 
    feats.append(out_img)
    print('done')
np.shape(feats)
feats = np.resize(feats,[np.shape(feats)[0],np.shape(feats)[2]])  #quick fixed for size
from sklearn.neighbors import NearestNeighbors,LSHForest
nbrs  = NearestNeighbors(n_neighbors=np.size(feats,0), algorithm='brute',metric='cosine').fit(feats)



    
@app.route("/process", methods=['POST'])
def process():
    print("omer")
    #sketch_path = (request.data)
    data=(request.data)
    #sketch_path = "/home/omer/caffe/python/static/sketchy/sketches_png/png/alarm clock/98.png"
    #with open(sketch_path, 'rb') as f:
    #   a=base64.b64encode(f.read())
    r=open('output1.bin','wb') 
    r.write(data)
    r.close()
 
    #data=(request.data)
    #print(data)
    #f=open('output1.bin','wb') 
    #f.write(data)
    #f.close()
    file=open('output1.bin','rb')
    byte=file.read()
    file.close()
    fh=open('omer.png','wb')
    fh.write(base64.b64decode((byte)))
    fh.close()
    
    

    
    #sketch_path = "veri.png"
    
    #print(sketch_path)
    sketch_path = "omer.png"
    #print(sketch_path)
    sketch_in = (transformer.preprocess('data', caffe.io.load_image(sketch_path)))
    sketch_in = np.reshape([sketch_in],np.shape(sketch_net.blobs['data'].data))
    query = sketch_net.forward(data=sketch_in)
    query=np.copy(query[output_layer_sketch])
    #get nn
    distances, indices = nbrs.kneighbors(np.reshape(query,[np.shape(query)[1]])[np.newaxis,:])


    #show query
    f = plt.figure(0)
    #plt.imshow(plt.imread(sketch_path))
    #plt.axis('off')
    arr=[]
    b=[]

    #show results
    for i in range(1,5,1):
        f = plt.figure(i)
        path = photo_paths+img_list[indices[0][i-1]]
        shutil.copy(path,'/home/omer/caffe/python/images/')
        img = plt.imread(path)
        arr.append(img_list[indices[0][i-1]])
        #arr.append("localhost:5000/images/"+img_list[indices[0][i-1]])
    for a in arr:
        print("done")
        a=a.split("/")
        print(a[1])
        b.append("http://localhost:5000/images/"+a[1])
       

	

	    
        
        #plt.imshow(img)
        #plt.axis('off')
        #plt.show(block=False)
    

    return jsonify(b)


