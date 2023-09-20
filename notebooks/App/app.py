# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 

#Utils
import joblib 
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_03.pkl","rb"))

def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

st.title("Emotion Classifier App")
def main(): 
    menu = ["Home"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Detecting Emotion In Text!")

        with st.form(key='emotion_clf_form'):
             raw_text = st.text_area("Type Here")
             submit_text = st.form_submit_button(label='Submit')
    
        if submit_text:

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

        
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            st.write(prediction)    


if __name__ == '__main__':
	main()
