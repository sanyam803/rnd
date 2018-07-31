from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import vgg16
from keras import backend as K
#K.set_image_data_format("channels_last")
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.vgg16 import preprocess_input
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""  ## for maipualting use of cpu of gpu
import keras
import numpy as np
import cv2
import wordninja as ninja
##*************Model**********************************************************************
input_tensor = Input(shape=(224,224,3))
model = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
model_ft = Sequential()

for layer in model.layers:
    model_ft.add(layer)


model_ft.add(Flatten())
model_ft.add(Dense(512, activation='relu'))
model_ft.add(Dropout(0.5))
model_ft.add(Dense(512, activation='relu'))
model_ft.add(Dropout(0.5))
model_ft.add(Dense(8, activation='softmax'))
##*********************Model**********************************************************
model_ft.load_weights("weights_latest_8.hdf5")## Load weights
label_map={0:"a",1:"e",2:"f",3:"i",4:"m",5:"n",6:"o",7:"u"}


def sign_prediction(img):
    x=cv2.resize(img,(224,224))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds=model_ft.predict(x)
    y_classes = preds.argmax(axis=-1)
    label=label_map[int(y_classes)]
    return (label,preds[0][int(y_classes)])



cap = cv2.VideoCapture(0)
cap.open(-1)
print(cap.isOpened())
cnt=0;
val="DEVIL";
Not_printed=0;
que=[];
while(cap.isOpened()):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    img=frame

    # Our operations on the frame come here
    label1,conf=sign_prediction(img)
    if conf<0.7:
        label="NT"
        label1="NT"
    else:

        label=label1+"--conf--"+str(conf)
    print(label)
    print(val)
    if(val!=label1):
      cnt=1;
      val=label1;
    elif(val==label1):
      cnt=cnt+1;
   
    x=img.shape[1]/2
    font = cv2.FONT_HERSHEY_SIMPLEX
    if(cnt>=10):
      cv2.putText(img,label, (10,50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.imshow('frame',img)
      if(len(que)!=0 and que[len(que)-1]!=label1 and label1!="NT"):
          que.append(label1);
      elif(len(que)==0 and label1!="NT"):
          que.append(label1);
      Not_printed=0;
      print("displaying")
    else: 
      Not_printed+=1;

    if(Not_printed>1 and Not_printed%5==0):
      cv2.putText(img,"NT", (10,50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.imshow('frame',img)
      print("not displaying"+str(cnt))
    # Display the resulting frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if label1=="o" and cnt>50 and len(que)>2:
        break;

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
while(len(que)>0 and que[len(que)-1]=="o"):
  que.pop();
while(len(que)>0 and que[0]=="o"):
  que.pop(0);

s=""
for char in que:
    s=s+char;
s1=ninja.split(s)
print(s);

print(*s1,sep=" ")

