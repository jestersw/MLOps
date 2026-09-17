import streamlit as st
import requests

st.title("Wine Classification App")
st.write("Enter the chemical features of the wine to predict its class (0, 1, or 2).")

# Input fields
features = {
    "alcohol": st.number_input("Alcohol", value=13.0),
    "malic_acid": st.number_input("Malic Acid", value=2.0),
    "ash": st.number_input("Ash", value=2.3),
    "alcalinity_of_ash": st.number_input("Alcalinity of Ash", value=19.0),
    "magnesium": st.number_input("Magnesium", value=100.0),
    "total_phenols": st.number_input("Total Phenols", value=2.8),
    "flavanoids": st.number_input("Flavanoids", value=2.5),
    "nonflavanoid_phenols": st.number_input("Nonflavanoid Phenols", value=0.3),
    "proanthocyanins": st.number_input("Proanthocyanins", value=1.5),
    "color_intensity": st.number_input("Color Intensity", value=4.0),
    "hue": st.number_input("Hue", value=1.0),
    "od280_od315_of_diluted_wines": st.number_input("OD280/OD315", value=3.0),
    "proline": st.number_input("Proline", value=1000.0)
}

if st.button("Predict"):
    try:
        # 'api' is the service name defined in docker-compose.yml
        response = requests.post("http://api:8000/predict", json=features)
        if response.status_code == 200:
            st.success(f"Predicted Wine Class: {response.json()['prediction']}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")