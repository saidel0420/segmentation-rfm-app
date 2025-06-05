import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import plotly.express as px
from streamlit_option_menu import option_menu

# ------------------- CONFIGURATION -------------------
st.set_page_config(page_title="Segmentation Client RFM - OCP", layout="wide")

# ------------------- NAVBAR HORIZONTALE EN HAUT -------------------
selected = option_menu(
    menu_title=None,
    options=[" Accueil", " Données RFM", " Segmentation", " Visualisation", " Interprétation"],
    icons=['house', 'table', 'gear', 'bar-chart', 'lightbulb'],
    menu_icon="cast",
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f2f6"},
        "icon": {"color": "blue", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px 10px"},
        "nav-link-selected": {"background-color": "#0e76a8", "color": "white"},
    }
)

st.markdown("---")
st.markdown(" Projet PFE - Juin 2025  \n Réalisé par : SAID EL ALAOUI & HIND BOUMAZA")


# ------------------- DONNÉES -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_segmentation_clients_9000.csv")
    df['Dernier_Achat'] = pd.to_datetime(df['Dernier_Achat'])
    df['Recence'] = (pd.Timestamp.today() - df['Dernier_Achat']).dt.days
    df['Frequence'] = df['Revenu_Annuel']
    df['Montant'] = df['Total_Dépensé']
    return df

df = load_data()

rfm = df[['ID_Client', 'Recence', 'Frequence', 'Montant']].copy()

q1 = rfm.quantile(0.01)
q3 = rfm.quantile(0.99)
rfm_clean = rfm[(rfm >= q1) & (rfm <= q3)].dropna()

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_clean[['Recence', 'Frequence', 'Montant']])

n_clusters = st.slider("Nombre de segments (GMM)", 2, 8, 4)
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
rfm_clean['Segment'] = gmm.fit_predict(rfm_scaled)

# ------------------- PAGES -------------------

# 1. ACCUEIL
if selected == " Accueil":
    st.title(" Segmentation RFM des Clients")
    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px'>
        <h4> Objectifs :</h4>
        <ul>
            <li>Identifier les groupes de clients basés sur leur comportement</li>
            <li>Aider à la prise de décisions marketing</li>
            <li>Proposer des stratégies ciblées pour chaque segment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    # Si tu veux afficher une image marketing, décommente la ligne suivante et remplace par ton image
    # st.image("marketing_clients.jpg", use_column_width=True)

# 2. DONNÉES RFM
elif selected == " Données RFM":
    st.title(" Données RFM")
    st.markdown("Voici un aperçu des données après traitement RFM :")
    st.dataframe(rfm_clean.head(50))
    st.success(f"✔️ Nombre total de clients après nettoyage : {len(rfm_clean)}")

# 3. SEGMENTATION
elif selected == " Segmentation":
    st.title(" Résultats de la segmentation")
    summary = rfm_clean.groupby('Segment')[['Recence', 'Frequence', 'Montant']].mean().round(1)
    summary['Nombre de clients'] = rfm_clean['Segment'].value_counts()
    st.dataframe(summary.style.highlight_max(axis=0))

# 4. VISUALISATION
elif selected == " Visualisation":
    st.title(" Visualisation des segments")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2D : Recence vs Montant")
        fig1 = px.scatter(
            rfm_clean, x='Recence', y='Montant',
            color='Segment', hover_name='ID_Client',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("3D : RFM")
        fig2 = px.scatter_3d(
            rfm_clean, x='Recence', y='Frequence', z='Montant',
            color='Segment', hover_name='ID_Client',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)

# 5. INTERPRÉTATION
elif selected == " Interprétation":
    st.title(" Interprétation des Segments")
    for segment in sorted(rfm_clean['Segment'].unique()):
        seg_data = rfm_clean[rfm_clean['Segment'] == segment]
        rec = seg_data['Recence'].mean()
        freq = seg_data['Frequence'].mean()
        mont = seg_data['Montant'].mean()
        nb = len(seg_data)

        st.markdown(f"""
        <div style="background-color:#f9f9f9; padding:10px; border-left: 4px solid #0e76a8;">
        <h5>🔹 Segment {segment}</h5>
        <ul>
            <li><b>Recence moyenne :</b> {rec:.1f} jours</li>
            <li><b>Fréquence moyenne :</b> {freq:.1f}</li>
            <li><b>Montant moyen :</b> {mont:.1f}</li>
            <li><b>Nombre de clients :</b> {nb}</li>
        </ul>
        <i>Interprétation : {"Clients fidèles à fort potentiel" if rec < 50 and mont > 1000 else "Clients à relancer ou inactifs"}</i>
        </div>
        """, unsafe_allow_html=True)
