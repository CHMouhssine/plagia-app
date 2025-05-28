# -*- coding: utf-8 -*-
"""
DÉTECTEUR DE PLAGIAT COMPLET AVEC INTERFACE STREAMLIT
Installation: pip install streamlit scikit-learn plotly pandas numpy
Exécution: streamlit run app.py
"""
import zipfile
from xml.dom.minidom import Document

import PyPDF2
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer

# ============================================================================
# CLASSE PRINCIPALE - DÉTECTEUR DE PLAGIAT
# ============================================================================
import joblib
st.set_page_config(
    page_title="Détecteur de Plagiat IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger une seule fois
if 'svm_model' not in st.session_state:
    st.session_state.svm_model = joblib.load("svm_model.joblib")

if 'scaler' not in st.session_state:
    st.session_state.scaler = joblib.load("scaler.joblib")


if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # same as training




def detect_plagiarism_svm(documents, model, scaler, threshold):
        # Extraire les contenus
        texts = [doc['processed_content'] for doc in documents]

        # Re-vectoriser avec le même scaler que lors de l'entraînement
        X = scaler.transform(texts)

        similarities = []
        n_docs = len(documents)
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                # Créer la différence vectorielle ou concaténer les deux
                pair_features = np.abs(X[i] - X[j])  # ou np.concatenate([X[i], X[j]])
                pair_features = pair_features.reshape(1, -1)

                pred = model.predict(pair_features)[0]
                prob = getattr(model, "predict_proba", lambda x: [[0.0, 1.0]])(pair_features)[0][1]

                similarities.append({
                    'doc1_id': i,
                    'doc2_id': j,
                    'doc1_title': documents[i]['title'],
                    'doc2_title': documents[j]['title'],
                    'similarity': prob,
                    'is_plagiarism': pred == 1,
                    'percentage': prob * 100
                })

        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)
class PlagiarismDetector:
    """
    Détecteur de plagiat utilisant TF-IDF et similarité cosinus
    """

    def __init__(self, threshold=0.7, language='french'):
        """
        Initialise le détecteur de plagiat

        Args:
            threshold (float): Seuil de détection (0-1)
            language (str): Langue pour les stop words
        """
        self.threshold = threshold
        self.language = language
        self.documents = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self.similarities = []
        self.vocabulary = []

    def preprocess_text(self, text):
        """
        Préprocesse le texte pour l'analyse

        Args:
            text (str): Texte à préprocesser

        Returns:
            str: Texte préprocessé
        """
        # Conversion en minuscules
        text = text.lower()

        # Suppression des caractères spéciaux (garde les accents français)
        text = re.sub(r'[^\w\sàâäéèêëïîôöùûüÿç]', ' ', text)

        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def add_document(self, title, content):
        """
        Ajoute un document à analyser

        Args:
            title (str): Titre du document
            content (str): Contenu du document
        """
        processed_content = self.preprocess_text(content)
        self.documents.append({
            'id': len(self.documents) + 1,
            'title': title,
            'content': content,
            'processed_content': processed_content,
            'word_count': len(processed_content.split())
        })

    def calculate_tfidf(self):
        """
        Calcule la matrice TF-IDF pour tous les documents
        """
        if len(self.documents) < 2:
            raise ValueError("Au moins 2 documents sont nécessaires pour l'analyse")

        # Extraction du contenu préprocessé
        texts = [doc['processed_content'] for doc in self.documents]

        # Configuration du vectoriseur TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Limite le vocabulaire
            ngram_range=(1, 2), # Unigrams et bigrams
            min_df=1,           # Minimum document frequency
            stop_words=None     # Pas de stop words pour plus de précision
        )

        # Calcul de la matrice TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Extraction du vocabulaire
        self.vocabulary = self.vectorizer.get_feature_names_out()

    def calculate_similarities(self):
        """
        Calcule les similarités entre toutes les paires de documents
        """
        if self.tfidf_matrix is None:
            self.calculate_tfidf()

        # Calcul de la matrice de similarité cosinus
        similarity_matrix = cosine_similarity(self.tfidf_matrix)

        # Extraction des paires et leurs similarités
        self.similarities = []
        n_docs = len(self.documents)

        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                similarity_score = similarity_matrix[i][j]

                self.similarities.append({
                    'doc1_id': i,
                    'doc2_id': j,
                    'doc1_title': self.documents[i]['title'],
                    'doc2_title': self.documents[j]['title'],
                    'similarity': similarity_score,
                    'is_plagiarism': similarity_score > self.threshold,
                    'percentage': similarity_score * 100
                })

        # Tri par similarité décroissante
        self.similarities.sort(key=lambda x: x['similarity'], reverse=True)

    def detect_plagiarism(self):
        """
        Lance la détection complète de plagiat

        Returns:
            list: Liste des résultats de détection
        """
        self.calculate_similarities()
        return self.similarities

    def get_detailed_results(self):
        """
        Retourne les résultats détaillés sous forme de DataFrame

        Returns:
            pd.DataFrame: Résultats détaillés
        """
        if not self.similarities:
            self.detect_plagiarism()

        df = pd.DataFrame(self.similarities)
        df['similarity_percentage'] = df['similarity'] * 100

        return df[['doc1_title', 'doc2_title', 'similarity_percentage', 'is_plagiarism']]



    def get_vocabulary_stats(self, top_n=20):
        """
        Retourne les statistiques sur le vocabulaire

        Args:
            top_n (int): Nombre de termes les plus fréquents à retourner

        Returns:
            dict: Statistiques du vocabulaire
        """
        if self.tfidf_matrix is None:
            self.calculate_tfidf()

        # Calcul des scores TF-IDF moyens pour chaque terme
        mean_scores = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
        vocab_scores = list(zip(self.vocabulary, mean_scores))
        vocab_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            'total_terms': len(self.vocabulary),
            'top_terms': vocab_scores[:top_n],
            'matrix_shape': self.tfidf_matrix.shape
        }

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

# Configuration de la page

# Titre principal avec style
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #2E86C1; font-size: 3rem; margin-bottom: 0.5rem;">
         Détecteur de Plagiat IA
    </h1>
    <p style="color: #5D6D7E; font-size: 1.2rem;">
        Analyse de similarité textuelle basée sur TF-IDF et similarité cosinus
    </p>
</div>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
if 'detector' not in st.session_state:
    st.session_state.detector = PlagiarismDetector()
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Sidebar pour la configuration
st.sidebar.header("⚙️ Configuration")

# Seuil de détection
threshold = st.sidebar.slider(
    "Seuil de détection (%)",
    min_value=30,
    max_value=90,
    value=70,
    help="Pourcentage de similarité au-dessus duquel un plagiat est détecté"
)
st.session_state.detector.threshold = threshold / 100

# Section d'ajout de documents
st.sidebar.header("🧠 Choix du Modèle de Similarité")

selected_model = st.sidebar.selectbox(
    "Modèle de Similarité",
    ["Cosine Similarity (TF-IDF)", "SBERT", "LSTM", "SVM"],
    help="Choisissez la méthode utilisée pour comparer les documents"
)

# Stocker le modèle choisi dans la session
st.session_state["selected_model"] = selected_model

st.sidebar.header("📄 Gestion des Documents")

with st.sidebar.expander("➕ Ajouter un Document", expanded=True):
    with st.form("add_document"):
        doc_title = st.text_input("Titre du document")
        doc_content = st.text_area("Contenu du document", height=150)

        if st.form_submit_button("Ajouter Document"):
            if doc_title and doc_content:
                st.session_state.detector.add_document(doc_title, doc_content)
                st.session_state.documents.append({
                    'title': doc_title,
                    'content': doc_content,
                    'word_count': len(doc_content.split())
                })
                st.session_state.analysis_done = False
                st.success(f"Document '{doc_title}' ajouté!")
            else:
                st.error("Veuillez remplir le titre et le contenu")

# Charger des exemples
    st.sidebar.button("📚 Charger des Exemples")
    uploaded_files = st.sidebar.file_uploader(
        "Uploader un ou plusieurs fichiers (TXT, PDF, DOCX ou ZIP)",
        type=['txt', 'pdf', 'docx', 'zip'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.detector = PlagiarismDetector(threshold / 100)
        st.session_state.documents = []


        def extract_text(file):
            file_type = file.name.split('.')[-1].lower()
            if file_type == 'txt':
                return file.read().decode('utf-8', errors='ignore')
            elif file_type == 'pdf':
                reader = PyPDF2.PdfReader(file)
                return '\n'.join(page.extract_text() or '' for page in reader.pages)
            elif file_type == 'docx':
                doc = Document(file)
                return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
            return ''


        for file in uploaded_files:
            if file.name.endswith('.zip'):
                with zipfile.ZipFile(file, 'r') as z:
                    for inner_file_name in z.namelist():
                        if inner_file_name.endswith(('.txt', '.pdf', '.docx')):
                            with z.open(inner_file_name) as inner_file:
                                content = extract_text(inner_file)
                                st.session_state.detector.add_document(inner_file_name, content)
                                st.session_state.documents.append({
                                    'title': inner_file_name,
                                    'content': content,
                                    'word_count': len(content.split())
                                })
            else:
                content = extract_text(file)
                st.session_state.detector.add_document(file.name, content)
                st.session_state.documents.append({
                    'title': file.name,
                    'content': content,
                    'word_count': len(content.split())
                })

        st.sidebar.success("Fichiers importés avec succès!")
        st.session_state.analysis_done = False

# Bouton de suppression des documents
if st.sidebar.button("🗑️ Supprimer tous les documents"):
    st.session_state.detector = PlagiarismDetector(threshold/100)
    st.session_state.documents = []
    st.session_state.analysis_done = False
    st.sidebar.success("Documents supprimés!")

# Interface principale
col1, col2 = st.columns([1, 1])





with col1:
    st.header("📋 Documents Chargés")

    if st.session_state.documents:
        to_remove = None

        for i, doc in enumerate(st.session_state.documents):
            col1_doc, col2_doc = st.columns([0.9, 0.1])
            with col1_doc:
                with st.expander(f"📄 {doc['title']} ({doc['word_count']} mots)"):
                    st.text_area("Contenu", value=doc['content'][:500], height=100, disabled=True, key=f"doc_content_{i}")
            with col2_doc:
                if st.button("❌", key=f"remove_{i}"):
                    to_remove = i

        # Supprimer le document immédiatement
        if to_remove is not None:
            del st.session_state.documents[to_remove]
            del st.session_state.detector.documents[to_remove]
            st.session_state.analysis_done = False  # Réinitialiser état d'analyse
            st.success("Document supprimé avec succès.")
            st.rerun()
 # ✅ Recharge immédiat

        # Bouton d'analyse
        if st.button("🔍 Analyser les Documents", type="primary", use_container_width=True):
            if len(st.session_state.documents) < 2:
                st.error("Au moins 2 documents sont nécessaires pour l'analyse")
            else:
                with st.spinner("Analyse en cours..."):
                    try:
                        model = st.session_state.get("selected_model", "Cosine Similarity (TF-IDF)")
                        if model == "Cosine Similarity (TF-IDF)":
                            results = st.session_state.detector.detect_plagiarism()
                        elif model == "SBERT":
                            st.warning("SBERT pas encore implémenté.")
                        elif model == "LSTM":
                            st.warning("LSTM pas encore implémenté.")
                        elif model == "SVM":

                            results = detect_plagiarism_svm(
                                documents=st.session_state.detector.documents,
                                model=st.session_state.svm_model,
                                scaler=st.session_state.scaler,
                                threshold=st.session_state.detector.threshold
                            )

                            st.session_state.detector.similarities = results

                            if not results:
                                st.warning("Aucune similarité détectée par le modèle SVM.")
                            else:
                                st.write("🔍 Résultats SVM bruts :")
                                st.write(results)
                        st.session_state.analysis_done = True
                        st.success("Analyse terminée!")
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse: {str(e)}")
    else:
        st.info("Aucun document chargé. Utilisez la barre latérale pour ajouter des documents.")

with col2:
    st.header("📊 Résultats de l'Analyse")

    if st.session_state.analysis_done and st.session_state.detector.similarities:
        # Tableau des résultats
        df_results = st.session_state.detector.get_detailed_results()
        st.subheader("Résultats Détaillés")

        # Formatage du DataFrame pour l'affichage
        df_display = df_results.copy()
        df_display['similarity_percentage'] = df_display['similarity_percentage'].round(2)
        df_display['is_plagiarism'] = df_display['is_plagiarism'].map({True: '⚠️ Plagiat', False: '✅ Original'})
        df_display.columns = ['Document 1', 'Document 2', 'Similarité (%)', 'Statut']

        st.dataframe(df_display, use_container_width=True)

        # Statistiques rapides
        plagiarism_count = sum(1 for s in st.session_state.detector.similarities if s['is_plagiarism'])
        total_comparisons = len(st.session_state.detector.similarities)

        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Documents", len(st.session_state.documents))
        with col_stat2:
            st.metric("Comparaisons", total_comparisons)
        with col_stat3:
            st.metric("Cas de plagiat", plagiarism_count)

        # Graphique des similarités
        st.subheader("Visualisation des Similarités")

        similarities_data = []
        colors = []

        for s in st.session_state.detector.similarities:
            comparison = f"{s['doc1_title'][:15]}... vs {s['doc2_title'][:15]}..."
            similarities_data.append({
                'Comparaison': comparison,
                'Similarité (%)': s['percentage'],
                'Type': 'Plagiat Détecté' if s['is_plagiarism'] else 'Document Original'
            })

        df_viz = pd.DataFrame(similarities_data)

        fig = px.bar(df_viz,
                    x='Comparaison',
                    y='Similarité (%)',
                    color='Type',
                    color_discrete_map={
                        'Plagiat Détecté': '#FF6B6B',
                        'Document Original': '#4ECDC4'
                    },
                    title='Scores de Similarité par Paire de Documents')

        # Ligne de seuil
        fig.add_hline(y=threshold,
                     line_dash="dash",
                     line_color="red",
                     annotation_text=f"Seuil de plagiat ({threshold}%)")

        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Matrice de similarité
        if len(st.session_state.documents) <= 10:  # Limite pour la lisibilité
            st.subheader("Matrice de Similarité")

            # Calcul de la matrice de similarité complète
            similarity_matrix = cosine_similarity(st.session_state.detector.tfidf_matrix)

            # Création des labels
            labels = [doc['title'][:20] + '...' if len(doc['title']) > 20
                     else doc['title'] for doc in st.session_state.documents]

            # Heatmap avec Plotly
            fig_heatmap = px.imshow(similarity_matrix,
                                   x=labels,
                                   y=labels,
                                   color_continuous_scale='RdYlBu_r',
                                   aspect='auto',
                                   title='Matrice de Similarité entre Documents')

            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)

    elif st.session_state.documents and not st.session_state.analysis_done:
        st.info("Cliquez sur 'Analyser les Documents' pour voir les résultats.")
    else:
        st.info("Aucune analyse disponible. Ajoutez des documents et lancez l'analyse.")

# Section d'aide
with st.expander("ℹ️ Comment utiliser ce détecteur de plagiat"):
    st.markdown("""
    ### Instructions d'utilisation:
    
    1. **Ajouter des documents**: Utilisez la barre latérale pour ajouter vos documents un par un
    2. **Ou charger des exemples**: Cliquez sur "Charger des Exemples" pour tester avec des données de démonstration
    3. **Configurer le seuil**: Ajustez le seuil de détection selon vos besoins (70% par défaut)
    4. **Analyser**: Cliquez sur "Analyser les Documents" pour détecter les similarités
    5. **Interpréter les résultats**: 
       - Rouge = Plagiat détecté (au-dessus du seuil)
       - Vert = Document original (en-dessous du seuil)
    
    ### Méthode utilisée:
    - **TF-IDF**: Calcul de l'importance des mots dans chaque document
    - **Similarité cosinus**: Mesure de l'angle entre les vecteurs de documents
    - **Seuil personnalisable**: Vous définissez à partir de quel pourcentage considérer un plagiat
    
    ### Limites:
    - Détection basée sur la similarité lexicale
    - Ne détecte pas la paraphrase sophistiquée
    - Sensible à la longueur des documents
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7F8C8D;'>"
    "Détecteur de Plagiat IA - Basé sur TF-IDF et Similarité Cosinus"
    "</div>",
    unsafe_allow_html=True
)
