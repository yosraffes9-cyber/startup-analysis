
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

st.set_page_config(page_title="Startup Analysis", layout="wide")
st.title("🚀 Interface d'Analyse des Startups")

# Upload Database
uploaded_file = st.sidebar.file_uploader("Upload 50_Startups.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Preprocessing
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X = df_encoded.drop('Profit', axis=1)
    y = df_encoded['Profit']

    # Inputs Sidebar
    st.sidebar.header("Predire pour un nouveau prix")
    user_inputs = {}
    for col in X.columns:
        user_inputs[col] = st.sidebar.number_input(f"Prix {col}", value=float(df_encoded[col].mean()))

    # Boutons
    if st.button("Lancer Backward Elimination"):
        X_pd = sm.add_constant(X)
        while True:
            model = sm.OLS(y, X_pd).fit()
            if model.pvalues.max() > 0.05:
                var = model.pvalues.idxmax()
                st.write(f"Suppression de: {var}")
                X_pd = X_pd.drop(columns=[var])
            else:
                break
        st.success(f"Variables finales: {list(X_pd.columns)}")
        st.text(model.summary())
else:
    st.info("Veuillez uploader le fichier CSV dans la barre à gauche.")
