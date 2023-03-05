import streamlit as st
import pandas as pd
import numpy as np
import pickle

island = st.sidebar.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])

bill_length = st.sidebar.slider("Bill length in mm", 29.0, 63.0, 37.0, 0.1)
bill_depth = st.sidebar.slider("Bill depth in mm", 12.0, 23.0, 18.0, 0.2)
flipper_length = st.sidebar.slider("Flipper length in mm", 166.0, 240.0, 200.0, 0.5)
body_mass = st.sidebar.slider("Body mass in g", 2340, 6660, 5600, 10)

df = pd.read_csv('Lesson_2_Penguin_Classification/penguins_cleaned.csv')
x = df.drop(columns='species')

user_value = dict(zip(x.columns, [island, bill_length, bill_depth, flipper_length, body_mass, sex]))
user_value = pd.DataFrame(user_value, index=[0])

x = pd.concat([x, user_value])
x = pd.get_dummies(x)
x = x[-1:] 

classifier = pickle.load(open('Lesson_2_Penguin_Classification/model.pickle', 'rb')) 
prediction = classifier.predict(x)

species = {0:'Adelie', 1:'Gentoo', 2:'Chinstrap'}

st.write(f'Species of your penguin is {species[prediction[0]]}!!')


