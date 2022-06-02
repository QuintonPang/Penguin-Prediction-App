#import modules
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# header and description
# one hash means h1
# double asterisk means bold
st.write("""
# Penguin Prediction App

This app predicts the species of **Palmer Penguin**!
""")

# sidebar header
st.sidebar.header("User Input Parameters")

# collects file uploaded by user 
uploaded_file = st.sidebar.file_uploader("Upload your CSV file",type=['csv'])

# if there is an uploaded file, read it
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:  
    # checkbox
    island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Sex',('male','female'))

    # set sliders
    # first parameter is label, second parameter is min value
    # third parameter is max value, fourth parameter is default value
    bill_length_mm = st.sidebar.slider("Bill Length(mm)",32.1,59.6,43.9)
    bill_depth_mm = st.sidebar.slider("Bill Depth(mm)",13.1,21.5,17.2)
    flipper_length_mm = st.sidebar.slider("Flipper Length(mm)",172.0,231.0,201.0)
    body_mass_g = st.sidebar.slider("Body mass(g)",2700.0,6300.0,4207.0)

    # set value into a dictionary
    data = {
        'island':island,
        'sex':sex,
        'bill_length_mm':bill_length_mm,
        'bill_depth_mm':bill_depth_mm,
        'flipper_length_mm':flipper_length_mm,
        'body_mass_g':body_mass_g,
    }

    # set to dataframe
    input_df = pd.DataFrame(data,index=[0])

# combine user input features with entire penguins dataset
penguins_raw = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# encoding of ordinal features
encode = ['sex','island']
for col in encode:
    # Convert categorical variable into dummy/indicator variables.
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]

# select only first row which is user input features
df = df[:1]

# output DataFrame
st.subheader('User Input Parameters')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting csv file to be uploaded...')
    st.write(input_df)

# Reads in saved classification model
# rb means read in binary mode
clf = pickle.load(open('penguins_clf.pkl','rb'))

# prediction
prediction = clf.predict(df)

# predicition probability (how probable the prediction is)
prediction_probability = clf.predict_proba(df)

# species in array corresponding to index
# numpy array is faster
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])

# output prediction
st.subheader("Prediction")
st.write(penguins_species[prediction])

# output prediction probability
st.subheader("Prediction Probability")
st.write(prediction_probability)
