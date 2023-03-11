import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models
Calories_model=pickle.load(open('calories_model.sav','rb'))




#Sidebar for navigation
#with st.sidebar:

#   selected=option_menu("Calories Burnt Prediction System",
#                       ["Calories Burnt Prediction"],
#                         icons=["activity"],
#                        default_index=0)

# CaloriesBurnt Prediction
st.title("Calories Burnt Prediction Using ML")

Age=st.text_input("Enter the Age")
Height=st.text_input("Enter the Height")
Weight=st.text_input("Enter the weight")
Duration=st.text_input("Enter the Duration of your Exercise")
Heart_Rate=st.text_input("Enter the Heart Rate")
Body_Temp=st.text_input("Enter the Body Temperature")