import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
st.title('Car Brand Prediction App')

import pickle


#image_path1=image.
MODEL_PATH='model_resnet50.h5'
file=st.file_uploader('upload the image of the car',type=['png','jpg'])
model=load_model(MODEL_PATH)
import cv2
from PIL import Image,ImageOps
def input_and_predict(image_data,model):

    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)

    img_reshape=img[np.newaxis,...]
    #img_reshape=img_reshape/255
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.write('Please upload the file')
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    predictions=input_and_predict(image,model)
    class_names=['Audi','Lamborgini','Mercedes']
    string='The Car is most likely be:'+class_names[np.argmax(predictions)]
    st.success(string)

















