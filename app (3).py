import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Configuration de la page pour un rendu professionnel
st.set_page_config(page_title="Analyse Décisionnelle Startups", layout="wide")

# Titre et introduction
st.title("Outil d'Analyse de Performance des Startups")
st.write("Cette plateforme permet d'identifier les leviers de rentabilité et de simuler des prévisions de profit basées sur vos investissements.")

# --- SECTION 1 : CHARGEMENT DES DONNÉES ---
st.markdown("---")
st.header("1. Importation du fichier de données")
uploaded_file = st.file_uploader("Veuillez charger votre fichier 50_Startups.csv", type="csv")

if uploaded_file is not None:
    # Lecture des données
    df = pd.read_csv(uploaded_file)
    
    with st.expander("Afficher un aperçu du jeu de données"):
        st.dataframe(df.head(10))
    
    # Préparation des variables pour le modèle
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X_data = df_encoded.drop('Profit', axis=1).astype(float)
    y_data = df_encoded['Profit'].astype(float)

    # --- SECTION 2 : SIMULATION ---
    st.markdown("---")
    st.header("2. Simulation de profit prévisionnel")
    st.write("Ajustez les paramètres suivants pour calculer une estimation du profit.")
    
    col_in = st.columns(5)
    feature_names = ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New_York']
    user_inputs = {}

    for i, col_name in enumerate(feature_names):
        with col_in[i % 5]:
            # Nettoyage de l'affichage du nom de la variable
            label_affichage = col_name.replace('_', ' ')
            valeur_moyenne = float(X_data[col_name].mean())
            user_inputs[col_name] = st.number_input(label_affichage, value=valeur_moyenne)

    # --- SECTION 3 : ANALYSE STATISTIQUE ---
    st.markdown("---")
    st.header("3. Analyse de Régression et Optimisation")
    
    zone_gauche, zone_droite = st.columns([1, 1])
    
    with zone_gauche:
        if st.button("Calculer le modèle optimal"):
            # Algorithme d'élimination descendante
            X_opt = sm.add_constant(X_data).astype(float)
            while True:
                modele = sm.OLS(y_data, X_opt).fit()
                if modele.pvalues.max() > 0.05:
                    variable_max = modele.pvalues.idxmax()
                    X_opt = X_opt.drop(columns=[variable_max])
                else:
                    break
            
            st.success("Le modèle a été optimisé avec succès.")
            st.write(f"**Variables retenues pour la prédiction :** {', '.join(list(X_opt.columns[1:]))}")
            
            # Calcul de la prédiction finale
            input_df = pd.DataFrame([user_inputs])
            input_df = sm.add_constant(input_df, has_constant='add')
            input_final = input_df[X_opt.columns]
            prediction = modele.predict(input_final)[0]
            
            st.metric("Profit Estimé", f"{prediction:,.2f} $")
            st.text("Résumé détaillé du modèle :")
            st.text(modele.summary())

    with zone_droite:
        if st.button("Consulter l'historique d'élimination"):
            st.write("**Détails des étapes de calcul (P-value > 0.05) :**")
            X_step = sm.add_constant(X_data).astype(float)
            etape = 1
            while True:
                model_step = sm.OLS(y_data, X_step).fit()
                with st.expander(f"Étape {etape} : Analyse de {len(X_step.columns)} variables"):
                    st.text(model_step.summary())
                
                if model_step.pvalues.max() > 0.05:
                    var_supprimee = model_step.pvalues.idxmax()
                    X_step = X_step.drop(columns=[var_supprimee])
                    etape += 1
                else:
                    st.info("L'optimisation est terminée.")
                    break
else:
    st.info("En attente de l'importation du fichier CSV pour lancer l'analyse.")
