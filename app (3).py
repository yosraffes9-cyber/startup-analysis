import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Configuration de la page
st.set_page_config(page_title="Analyse de Rentabilité", layout="wide")

# Titre principal de l'application
st.title("Outil d'Aide à la Décision : Analyse des Startups")
st.write("Cette interface vous permet d'analyser les facteurs influençant le profit et de simuler des résultats basés sur vos investissements.")

# --- 1. CHARGEMENT DES DONNÉES ---
st.header("1. Importation des données")
uploaded_file = st.file_uploader("Sélectionnez votre fichier CSV pour commencer l'analyse", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    with st.expander("Consulter la base de données chargée"):
        st.dataframe(df.head(10))
    
    # Préparation des variables (Preprocessing)
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X_data = df_encoded.drop('Profit', axis=1).astype(float)
    y_data = df_encoded['Profit'].astype(float)

    # --- 2. PARAMÈTRES DE SIMULATION ---
    st.markdown("---")
    st.header("2. Simulation de profit")
    st.write("Modifiez les valeurs ci-dessous pour calculer une estimation du profit prévisionnel.")
    
    col_in = st.columns(5)
    feature_names = ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New_York']
    user_inputs = {}

    for i, col_name in enumerate(feature_names):
        with col_in[i % 5]:
            # Remplacement des underscores pour un affichage plus propre
            label_propre = col_name.replace('_', ' ')
            # Valeur par défaut basée sur la moyenne du dataset
            valeur_moyenne = float(X_data[col_name].mean())
            user_inputs[col_name] = st.number_input(label_propre, value=valeur_moyenne)

    # --- 3. ANALYSE ET OPTIMISATION ---
    st.markdown("---")
    st.header("3. Résultats et Optimisation du Modèle")
    
    zone_gauche, zone_droite = st.columns([1, 1])
    
    with zone_gauche:
        if st.button("Calculer le Modèle Optimal"):
            # Algorithme Backward Elimination
            X_opt = sm.add_constant(X_data).astype(float)
            while True:
                regresseur = sm.OLS(y_data, X_opt).fit()
                if regresseur.pvalues.max() > 0.05:
                    variable_max = regresseur.pvalues.idxmax()
                    X_opt = X_opt.drop(columns=[variable_max])
                else:
                    break
            
            st.success("Analyse terminée avec succès.")
            st.write(f"**Variables retenues pour la précision du modèle :** {', '.join(list(X_opt.columns[1:]))}")
            
            # Calcul de la prédiction
            input_df = pd.DataFrame([user_inputs])
            input_df = sm.add_constant(input_df, has_constant='add')
            # On ne garde que les colonnes sélectionnées par l'algorithme
            input_final = input_df[X_opt.columns]
            prediction = regresseur.predict(input_final)[0]
            
            st.metric("Estimation du Profit", f"{prediction:,.2f} $")

    with zone_droite:
        if st.button("Détails des étapes techniques"):
            st.write("**Historique de l'élimination des variables (P-value > 0.05) :**")
            X_step = sm.add_constant(X_data).astype(float)
            etape = 1
            while True:
                model_step = sm.OLS(y_data, X_step).fit()
                with st.expander(f"Étape {etape} : Analyse de {len(X_step.columns)} variables"):
                    st.write(model_step.summary())
                
                if model_step.pvalues.max() > 0.05:
                    var_a_supprimer = model_step.pvalues.idxmax()
                    X_step = X_step.drop(columns=[var_a_supprimer])
                    etape += 1
                else:
                    st.info("Le modèle est désormais optimisé.")
                    break
else:
    st.info("Veuillez charger un fichier de données pour activer les fonctionnalités d'analyse.")
