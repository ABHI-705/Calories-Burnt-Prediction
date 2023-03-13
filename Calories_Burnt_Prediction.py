import pickle
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title='Calories Prediction',
    layout='wide'
)

Calories_model=pickle.load(open('calories.pkl','rb'))

coll1,coll2,coll3=st.columns([2,3,1])
with coll2:
    st.title("Calories Burnt Prediction                                         Using ML")


st.write(" ")
st.write(" ")
col1,col2=st.columns(2)

with col2:

    gende=st.radio("Select Gender",['Male','Female'])
    if gende=='Male':
        gender=0
    else:
        gender=1



    Age=st.number_input("Enter the Age")

    Height=st.number_input("Enter the Height")



    Weight=st.number_input("Enter the weight")

    Duration=st.number_input("Enter the Duration of your Exercise")



    Heart_Rate=st.number_input("Enter the Heart Rate")


    Body_Temp=st.number_input("Enter the Body Temperature")



    if st.button("Predict Calories Burnt"):
        input_df = pd.DataFrame({'Gender':[gender],'Age':[Age],'Height':[Height],
                                'Weight':[Weight],'Duration':[Duration],'Heart_Rate':[Heart_Rate],
                                'Body_Temp':[Body_Temp]})
        
        resul=Calories_model.predict(input_df)
        
        result=resul[0]
        
        st.header("Burnt calories is :👉 " + str(round(result)))

with col1:
    image=Image.open('cal.png')
    edited=image.resize((2600,1950))
    st.image(edited,use_column_width=True)
