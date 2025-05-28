# -*- coding: utf-8 -*-
"""
DÉTECTEUR DE PLAGIAT COMPLET AVEC INTERFACE STREAMLIT
Installation: pip install streamlit scikit-learn plotly pandas numpy tensorflow
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
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================================
# CLASSE PRINCIPALE - DÉTECTEUR DE PLAGIAT
# ============================================================================
import joblib
st.set_page_config(
    page_title="Détecteur de Plagiat IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger les modèles une seule fois
if 'svm_model' not in st.session_state:
    try:
        st.session_state.svm_model = joblib.load("svm_model2.joblib")
    except FileNotFoundError:
        st.session_state.svm_model = None
        st.warning("Modèle SVM non trouvé (svm_model2.joblib)")

if 'scaler' not in st.session_state:
    try:
        st.session_state.scaler = joblib.load("scaler2.joblib")
    except FileNotFoundError:
        st.session_state.scaler = None
        st.warning("Scaler non trouvé (scaler2.joblib)")

if 'sbert_model' not in st.session_state:
    try:
        st.session_state.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.session_state.sbert_model = None
        st.warning(f"Modèle SBERT non disponible: {e}")

# Charger le modèle LSTM et le tokenizer
if 'lstm_model' not in st.session_state:
    try:
        st.session_state.lstm_model = tf.keras.models.load_model("lstm_model2.h5")
    except FileNotFoundError:
        st.session_state.lstm_model = None
        st.warning("Modèle LSTM non trouvé (lstm_model2.h5)")

if 'tokenizer' not in st.session_state:
    try:
        st.session_state.tokenizer = joblib.load("tokenizer2.joblib")
    except FileNotFoundError:
        st.session_state.tokenizer = None
        st.warning("Tokenizer non trouvé (tokenizer2.joblib)")

def create_similarity_features(emb1, emb2):
    cosine_sim = np.sum(emb1 * emb2, axis=1).reshape(-1, 1)
    euclidean_dist = np.linalg.norm(emb1 - emb2, axis=1).reshape(-1, 1)
    manhattan_dist = np.sum(np.abs(emb1 - emb2), axis=1).reshape(-1, 1)

    abs_diff = np.abs(emb1 - emb2)
    variances = np.var(abs_diff, axis=0)
    top50_idx = np.argsort(variances)[-50:]  # ✅ comme à l'entraînement

    prod_top = (emb1 * emb2)[:, top50_idx]
    diff_top = abs_diff[:, top50_idx]

    stats = np.concatenate([
        np.max(abs_diff, axis=1).reshape(-1, 1),
        np.min(abs_diff, axis=1).reshape(-1, 1),
        np.mean(abs_diff, axis=1).reshape(-1, 1),
        np.std(abs_diff, axis=1).reshape(-1, 1)
    ], axis=1)

    return np.concatenate([
        cosine_sim, euclidean_dist, manhattan_dist,
        diff_top, prod_top, stats
    ], axis=1)

def detect_plagiarism_svm(documents, model, scaler, threshold):
    if model is None or scaler is None or st.session_state.sbert_model is None:
        st.error("Modèle SVM, scaler ou SBERT non disponible")
        return []
    
    texts = [doc['processed_content'] for doc in documents]
    titles = [doc['title'] for doc in documents]

    # Embeddings SBERT
    emb = st.session_state.sbert_model.encode(texts, normalize_embeddings=True)

    similarities = []
    n_docs = len(texts)
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            emb1 = emb[i].reshape(1, -1)
            emb2 = emb[j].reshape(1, -1)

            # Créer les features avancées
            feats = create_similarity_features(emb1, emb2)
            feats_scaled = scaler.transform(feats)

            pred = model.predict(feats_scaled)[0]
            prob = model.predict_proba(feats_scaled)[0][1]

            similarities.append({
                'doc1_id': i,
                'doc2_id': j,
                'doc1_title': titles[i],
                'doc2_title': titles[j],
                'similarity': prob,
                'is_plagiarism': pred == 1,
                'percentage': prob * 100
            })

    return sorted(similarities, key=lambda x: x['similarity'], reverse=True)

def detect_plagiarism_lstm(documents, model, tokenizer, threshold, max_length=40):
    """
    Détection de plagiat utilisant le modèle LSTM
    
    Args:
        documents: Liste des documents
        model: Modèle LSTM chargé
        tokenizer: Tokenizer pour la vectorisation
        threshold: Seuil de détection
        max_length: Longueur maximale des séquences (doit correspondre à l'entraînement)
    
    Returns:
        Liste des résultats de similarité
    """
    if model is None or tokenizer is None:
        st.error("Modèle LSTM ou tokenizer non disponible")
        return []
    
    texts = [doc['processed_content'] for doc in documents]
    titles = [doc['title'] for doc in documents]

    similarities = []
    n_docs = len(texts)
    
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            # Préparer les textes pour le modèle LSTM
            text1 = texts[i]
            text2 = texts[j]
            
            # Tokenisation et padding - SÉPARÉMENT comme dans l'entraînement
            seq1 = tokenizer.texts_to_sequences([text1])
            seq2 = tokenizer.texts_to_sequences([text2])
            
            # Padding des séquences avec les mêmes paramètres qu'à l'entraînement
            padded_seq1 = pad_sequences(seq1, maxlen=max_length, padding='post', truncating='post')
            padded_seq2 = pad_sequences(seq2, maxlen=max_length, padding='post', truncating='post')
            
            # Préparer l'entrée pour le modèle 
            # Le modèle attend probablement deux entrées séparées [X1, X2]
            try:
                # Si votre modèle utilise deux entrées séparées (architecture siamoise)
                prediction = model.predict([padded_seq1, padded_seq2], verbose=0)[0][0]
                
                # OU si votre modèle attend une seule entrée concaténée:
                # model_input = np.concatenate([padded_seq1, padded_seq2], axis=1)
                # prediction = model.predict(model_input, verbose=0)[0][0]
                
                similarities.append({
                    'doc1_id': i,
                    'doc2_id': j,
                    'doc1_title': titles[i],
                    'doc2_title': titles[j],
                    'similarity': prediction,
                    'is_plagiarism': prediction > threshold,
                    'percentage': prediction * 100
                })
            except Exception as e:
                st.error(f"Erreur lors de la prédiction LSTM: {e}")
                continue

    return sorted(similarities, key=lambda x: x['similarity'], reverse=True)
def detect_plagiarism_sbert(documents, threshold):
    """
    Détection de plagiat utilisant SBERT
    """
    if st.session_state.sbert_model is None:
        st.error("Modèle SBERT non disponible")
        return []
    
    texts = [doc['processed_content'] for doc in documents]
    titles = [doc['title'] for doc in documents]

    # Embeddings SBERT
    embeddings = st.session_state.sbert_model.encode(texts, normalize_embeddings=True)
    
    similarities = []
    n_docs = len(texts)
    
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            # Calcul de la similarité cosinus
            similarity_score = np.dot(embeddings[i], embeddings[j])
            
            similarities.append({
                'doc1_id': i,
                'doc2_id': j,
                'doc1_title': titles[i],
                'doc2_title': titles[j],
                'similarity': similarity_score,
                'is_plagiarism': similarity_score > threshold,
                'percentage': similarity_score * 100
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

# Titre principal avec style
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #2E86C1; font-size: 3rem; margin-bottom: 0.5rem;">
         Détecteur de Plagiat IA
    </h1>
    <p style="color: #5D6D7E; font-size: 1.2rem;">
        Analyse de similarité textuelle avec multiple modèles IA
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

# Vérifier la disponibilité des modèles
available_models = ["Cosine Similarity (TF-IDF)"]
if st.session_state.sbert_model is not None:
    available_models.append("SBERT")
if st.session_state.lstm_model is not None and st.session_state.tokenizer is not None:
    available_models.append("LSTM")
if st.session_state.svm_model is not None and st.session_state.scaler is not None:
    available_models.append("SVM")

selected_model = st.sidebar.selectbox(
    "Modèle de Similarité",
    available_models,
    help="Choisissez la méthode utilisée pour comparer les documents"
)

# Stocker le modèle choisi dans la session
st.session_state["selected_model"] = selected_model

# Paramètres spécifiques au modèle LSTM
# Paramètres spécifiques au modèle LSTM
if selected_model == "LSTM":
    max_sequence_length = st.sidebar.slider(
        "Longueur maximale des séquences (LSTM)",
        min_value=20,
        max_value=100,
        value=40,  # Changé de 100 à 40 pour correspondre à votre entraînement
        help="Longueur maximale des séquences pour le modèle LSTM (doit correspondre à l'entraînement)"
    )
    st.session_state["max_sequence_length"] = max_sequence_length

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
            st.session_state.analysis_done = False
            st.success("Document supprimé avec succès.")
            st.rerun()

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
                            results = detect_plagiarism_sbert(
                                documents=st.session_state.detector.documents,
                                threshold=st.session_state.detector.threshold
                            )
                            st.session_state.detector.similarities = results
                            
                        elif model == "LSTM":
                            max_len = st.session_state.get("max_sequence_length", 100)
                            results = detect_plagiarism_lstm(
                                documents=st.session_state.detector.documents,
                                model=st.session_state.lstm_model,
                                tokenizer=st.session_state.tokenizer,
                                threshold=st.session_state.detector.threshold,
                                max_length=max_len
                            )
                            st.session_state.detector.similarities = results
                            
                        elif model == "SVM":
                            results = detect_plagiarism_svm(
                                documents=st.session_state.detector.documents,
                                model=st.session_state.svm_model,
                                scaler=st.session_state.scaler,
                                threshold=st.session_state.detector.threshold
                            )
                            st.session_state.detector.similarities = results

                        st.session_state.analysis_done = True
                        if results:
                            st.success(f"Analyse terminée avec {model}!")
                        else:
                            st.warning("Aucun résultat obtenu. Vérifiez vos modèles et données.")
                            
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse: {str(e)}")
                        st.error("Détails de l'erreur pour le débogage:")
                        st.code(str(e))
    else:
        st.info("Aucun document chargé. Utilisez la barre latérale pour ajouter des documents.")

with col2:
    st.header("📊 Résultats de l'Analyse")

    if st.session_state.analysis_done and st.session_state.detector.similarities:
        # Afficher le modèle utilisé
        current_model = st.session_state.get("selected_model", "Cosine Similarity (TF-IDF)")
        st.info(f"Résultats obtenus avec: **{current_model}**")
        
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
                    title=f'Scores de Similarité par Paire de Documents ({current_model})')

        # Ligne de seuil
        fig.add_hline(y=threshold,
                     line_dash="dash",
                     line_color="red",
                     annotation_text=f"Seuil de plagiat ({threshold}%)")

        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Matrice de similarité (seulement pour TF-IDF)
        if current_model == "Cosine Similarity (TF-IDF)" and len(st.session_state.documents) <= 10:
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
    2. **Ou charger des fichiers**: Importez des fichiers TXT, PDF, DOCX ou ZIP
    3. **Choisir le modèle**: Sélectionnez le modèle de similarité (TF-IDF, SBERT, LSTM, SVM)
    4. **Configurer le seuil**: Ajustez le seuil de détection selon vos besoins (70% par défaut)
    5. **Analyser**: Cliquez sur "Analyser les Documents" pour détecter les similarités
    6. **Interpréter les résultats**: 
       - Rouge = Plagiat détecté (au-dessus du seuil)
       - Vert = Document original (en-dessous du seuil)
    
    ### Modèles disponibles:
    
    #### 1. **Cosine Similarity (TF-IDF)**
    - Méthode classique basée sur la fréquence des termes
    - Rapide et efficace pour la plupart des cas
    - Idéal pour détecter les similarités lexicales directes
    
    #### 2. **SBERT (Sentence-BERT)**
    - Modèle de langue pré-entraîné
    - Comprend le sens sémantique des phrases
    - Excellent pour détecter les paraphrases
    
    #### 3. **LSTM (Long Short-Term Memory)**
    - Réseau de neurones récurrents
    - Analyse les séquences de mots
    - Bon pour capturer les dépendances à long terme
    
    #### 4. **SVM (Support Vector Machine)**
    - Modèle d'apprentissage supervisé
    - Utilise des features avancées basées sur SBERT
    - Très précis avec un entraînement approprié
    
    ### Paramètres spéciaux:
    - **LSTM**: Vous pouvez ajuster la longueur maximale des séquences
    - **Tous les modèles**: Le seuil de détection est personnalisable
    
    ### Formats de fichiers supportés:
    - **TXT**: Fichiers texte brut
    - **PDF**: Documents PDF (extraction automatique du texte)
    - **DOCX**: Documents Word
    - **ZIP**: Archives contenant plusieurs fichiers
    
    ### Limites et considérations:
    - **TF-IDF**: Sensible à la similarité lexicale, moins efficace pour les paraphrases
    - **SBERT**: Nécessite une connexion internet pour le premier téléchargement
    - **LSTM**: Dépend de la qualité du modèle pré-entraîné
    - **SVM**: Nécessite les fichiers de modèle (svm_model2.joblib, scaler2.joblib)
    - **Performance**: Les modèles deep learning (SBERT, LSTM, SVM) sont plus lents mais plus précis
    
    ### Conseils d'utilisation:
    - Utilisez **TF-IDF** pour une analyse rapide de similarité lexicale
    - Utilisez **SBERT** pour détecter les paraphrases et reformulations
    - Utilisez **LSTM** pour analyser les structures séquentielles
    - Utilisez **SVM** pour la plus haute précision (si le modèle est bien entraîné)
    - Ajustez le seuil selon votre contexte (plus strict = moins de faux positifs)
    
    ### Interprétation des résultats:
    - **Scores élevés (>80%)**: Plagiat très probable
    - **Scores moyens (50-80%)**: Similarité significative, à examiner
    - **Scores faibles (<50%)**: Documents probablement originaux
    - **Matrice de similarité**: Vue d'ensemble des relations entre tous les documents
    """)

# Informations sur les modèles chargés
st.sidebar.header("📊 État des Modèles")
with st.sidebar.expander("Modèles Disponibles", expanded=False):
    # Vérification TF-IDF
    st.write("✅ **TF-IDF**: Toujours disponible")
    
    # Vérification SBERT
    if st.session_state.sbert_model is not None:
        st.write("✅ **SBERT**: Modèle chargé")
    else:
        st.write("❌ **SBERT**: Non disponible")
    
    # Vérification LSTM
    if st.session_state.lstm_model is not None and st.session_state.tokenizer is not None:
        st.write("✅ **LSTM**: Modèle et tokenizer chargés")
        st.write(f"   - Modèle: lstm_model2.h5")
        st.write(f"   - Tokenizer: tokenizer2.joblib")
    else:
        st.write("❌ **LSTM**: Modèle ou tokenizer manquant")
        if st.session_state.lstm_model is None:
            st.write("   - ❌ lstm_model2.h5 non trouvé")
        if st.session_state.tokenizer is None:
            st.write("   - ❌ tokenizer2.joblib non trouvé")
    
    # Vérification SVM
    if st.session_state.svm_model is not None and st.session_state.scaler is not None:
        st.write("✅ **SVM**: Modèle et scaler chargés")
        st.write(f"   - Modèle: svm_model2.joblib")
        st.write(f"   - Scaler: scaler2.joblib")
    else:
        st.write("❌ **SVM**: Modèle ou scaler manquant")
        if st.session_state.svm_model is None:
            st.write("   - ❌ svm_model2.joblib non trouvé")
        if st.session_state.scaler is None:
            st.write("   - ❌ scaler2.joblib non trouvé")

# Footer avec informations techniques
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7F8C8D;'>
    <p><strong>Détecteur de Plagiat IA Multi-Modèles</strong></p>
    <p>Supportant TF-IDF, SBERT, LSTM et SVM • Développé avec Streamlit et TensorFlow/Scikit-learn</p>
    <p>Pour de meilleurs résultats, assurez-vous que tous les fichiers de modèles sont présents dans le dossier de l'application</p>
</div>
""", unsafe_allow_html=True)
