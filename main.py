import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model
import sys


model = load_model('incerptionv3.h5', compile=(False))# Loading the Inception model


def predict(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    
    prediction = model.predict(img) # Predict with the Inceptionv3 model

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        label = ("{}: {:.2f}%".format(label, prob * 100))
    
    #Return the predicted label and its corresponding probability
    st.markdown(label)


def predict2(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

   
    prediction = model.predict(img)  # Predict with the Inceptionv3 model

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        pred_class = label
    
   
    return pred_class  #Return the predicted class for Search comparison


def object_detection(search_key, frame, model):
    label = predict2(frame, model)
    
    
    #Convert the string to lower case for effective comparison
    label = label.lower()
    if label.find(search_key) > -1:
        st.image(frame, caption=label)

        return sys.exit()
    else:
        pass
        


# Main App
def main():
    """Deployment using Streamlit"""
    st.title("Object Detection with incerptionv3 model")
    st.text("Done By Perfect Masiiwa R204554F & Stallone Chabvuta R206536X")

    activities = ["Detect Objects"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice == "Detect Objects":
        st.subheader("Upload Video file")

        video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv","wmv"])
 
        if video_file is not None:
            path = video_file.name
            with open(path, mode='wb') as f:
                f.write(video_file.read())
                st.success("Saved File")
                video_file = open(path, "rb").read()
                st.video(video_file)
            cap = cv2.VideoCapture(path)
            

            if st.button("Detect"):

                # video prediction loop
                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret:
                        break

                    
                    predict(frame, model)# Perform object detection

                   

                cap.release()
                
                

            key = st.text_input('search here.....')
            key = key.lower()

            if key is not None:

                if st.button("Search"):

                    #video prediction loop
                    while cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                            break

                        
                        object_detection(key, frame, model) # Perform object detection
                        
                    cap.release()
                   
                    

                    #Return statement if object is not found
                    st.text("Object searched not found")
        st.text("Perfect Masiiwa R204554f\nStallone Chabvuta r206536x")

if __name__ == '__main__':
    main()