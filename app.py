import re
import io
import json
import numpy as np
import pandas as pd
import trafilatura
import streamlit as st

import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline as hf_pipeline
from flair.data import Sentence
from flair.models import SequenceTagger

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

# --------------------------
# Config Streamlit
# --------------------------
st.set_page_config(page_title="Extraction & Clustering d'entités nommées", layout="wide")
st.title("🔍 Extraction d'entités nommées (5 méthodes) + 🧠 Clustering sémantique")
st.markdown(
    "Cette app extrait les entités depuis des pages web via **Regex**, **spaCy**, **CamemBERT**, **Flair**, "
    "**Embeddings sémantiques**, puis **fusionne et regroupe** les entités par **similarité sémantique**."
)

# --------------------------
# Modèles (mis en cache)
# --------------------------
@st.cache_resource
def load_spacy():
    return spacy.load("fr_core_news_lg")

@st.cache_resource
def load_camembert():
    model_name = "Jean-Baptiste/camembert-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return hf_pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

@st.cache_resource
def load_flair():
    return SequenceTagger.load("ner")

@st.cache_resource
def load_embeddings_model():
    # Multilingue, léger et efficace pour la similarité
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

nlp_spacy = load_spacy()
camembert_ner = load_camembert()
flair_tagger = load_flair()
embedding_model = load_embeddings_model()

# --------------------------
# Fonctions d'extraction
# --------------------------

def extract_with_regex(text):
    """Extraction par expressions régulières"""
    entities = []
    
    # Patterns regex pour différents types d'entités
    patterns = {
        "PERSON": [
            r'\b[A-ZÀ-ÿ][a-zà-ÿ]+\s+[A-ZÀ-ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]+)?',  # Noms propres
            r'\b(?:M\.|Mme|Dr|Prof\.)\s+[A-ZÀ-ÿ][a-zà-ÿ]+',  # Titres + noms
        ],
        "ORG": [
            r'\b[A-ZÀ-ÿ][a-zà-ÿ]*(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]*)*\s+(?:SA|SAS|SARL|EURL|SNC|GIE)',
            r'\b(?:Société|Entreprise|Compagnie|Association)\s+[A-ZÀ-ÿ][a-zà-ÿ]*(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]*)*',
        ],
        "LOC": [
            r'\b(?:Paris|Lyon|Marseille|Toulouse|Nice|Nantes|Strasbourg|Montpellier|Bordeaux|Lille)',
            r'\b[A-ZÀ-ÿ][a-zà-ÿ]*(?:-[A-ZÀ-ÿ][a-zà-ÿ]*)*(?:\s+sur\s+[A-ZÀ-ÿ][a-zà-ÿ]*)?',  # Villes composées
        ],
        "DATE": [
            r'\b\d{1,2}(?:/|-)\d{1,2}(?:/|-)\d{2,4}',  # Dates numériques
            r'\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{2,4}',
        ],
        "EMAIL": [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        ],
        "PHONE": [
            r'\b(?:\+33|0)[1-9](?:[0-9]{8})',
            r'\b\d{2}(?:\.\d{2}){4}',
        ]
    }
    
    for entity_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "label": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8,  # Score arbitraire pour regex
                    "method": "Regex"
                })
    
    return entities

def extract_with_spacy(text):
    """Extraction avec spaCy"""
    doc = nlp_spacy(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "confidence": float(ent._.get("confidence", 0.9)),
            "method": "spaCy"
        })
    
    return entities

def extract_with_camembert(text):
    """Extraction avec CamemBERT"""
    try:
        results = camembert_ner(text)
        entities = []
        
        for entity in results:
            entities.append({
                "text": entity["word"],
                "label": entity["entity_group"],
                "start": entity.get("start", 0),
                "end": entity.get("end", len(entity["word"])),
                "confidence": float(entity["score"]),
                "method": "CamemBERT"
            })
        
        return entities
    except Exception as e:
        st.error(f"Erreur CamemBERT: {e}")
        return []

def extract_with_flair(text):
    """Extraction avec Flair"""
    try:
        sentence = Sentence(text)
        flair_tagger.predict(sentence)
        entities = []
        
        for entity in sentence.get_spans('ner'):
            entities.append({
                "text": entity.text,
                "label": entity.tag,
                "start": entity.start_position,
                "end": entity.end_position,
                "confidence": float(entity.score),
                "method": "Flair"
            })
        
        return entities
    except Exception as e:
        st.error(f"Erreur Flair: {e}")
        return []

def extract_with_embeddings(text, similarity_threshold=0.75):
    """Extraction par similarité sémantique avec des entités connues"""
    
    # Base d'entités connues par catégorie
    known_entities = {
        "PERSON": ["Emmanuel Macron", "Marine Le Pen", "Jean Dupont", "Marie Martin"],
        "ORG": ["Google", "Microsoft", "Total", "SNCF", "EDF", "Orange"],
        "LOC": ["Paris", "France", "Europe", "Amérique", "Lyon", "Marseille"],
        "MISC": ["COVID-19", "Intelligence artificielle", "Blockchain", "5G"]
    }
    
    entities = []
    
    # Découper le texte en phrases/segments
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue
            
        # Extraire les mots/phrases candidates (capitalisés ou expressions)
        candidates = re.findall(r'\b[A-ZÀ-ÿ][a-zà-ÿ]*(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]*)*', sentence)
        
        for candidate in candidates:
            if len(candidate) < 3:
                continue
                
            candidate_embedding = embedding_model.encode([candidate])
            
            # Comparer avec les entités connues
            for label, known_list in known_entities.items():
                known_embeddings = embedding_model.encode(known_list)
                similarities = util.cos_sim(candidate_embedding, known_embeddings)
                max_similarity = float(similarities.max())
                
                if max_similarity > similarity_threshold:
                    start_pos = text.find(candidate)
                    if start_pos != -1:
                        entities.append({
                            "text": candidate,
                            "label": label,
                            "start": start_pos,
                            "end": start_pos + len(candidate),
                            "confidence": max_similarity,
                            "method": "Embeddings"
                        })
                    break
    
    return entities

# --------------------------
# Fusion et déduplication
# --------------------------

def merge_entities(all_entities, similarity_threshold=0.8):
    """Fusionne les entités similaires de différentes méthodes"""
    if not all_entities:
        return []
    
    # Créer un DataFrame pour faciliter la manipulation
    df = pd.DataFrame(all_entities)
    
    # Supprimer les doublons exacts
    df = df.drop_duplicates(subset=['text', 'label'])
    
    # Regrouper par similarité textuelle et sémantique
    merged_entities = []
    processed_indices = set()
    
    for i, entity in df.iterrows():
        if i in processed_indices:
            continue
            
        similar_group = [entity]
        
        # Chercher des entités similaires
        for j, other_entity in df.iterrows():
            if j <= i or j in processed_indices:
                continue
                
            # Similarité textuelle (Levenshtein simplifié)
            text_similarity = calculate_text_similarity(entity['text'], other_entity['text'])
            
            # Similarité sémantique
            semantic_similarity = 0.0
            try:
                embeddings = embedding_model.encode([entity['text'], other_entity['text']])
                semantic_similarity = float(util.cos_sim(embeddings[0:1], embeddings[1:2]).item())
            except:
                pass
            
            # Si suffisamment similaire, grouper
            if (text_similarity > similarity_threshold or 
                semantic_similarity > similarity_threshold) and entity['label'] == other_entity['label']:
                similar_group.append(other_entity)
                processed_indices.add(j)
        
        # Créer une entité fusionnée
        merged_entity = create_merged_entity(similar_group)
        merged_entities.append(merged_entity)
        processed_indices.add(i)
    
    return merged_entities

def calculate_text_similarity(text1, text2):
    """Calcule la similarité textuelle simple"""
    text1, text2 = text1.lower(), text2.lower()
    
    if text1 == text2:
        return 1.0
    
    # Similarité par mots communs
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def create_merged_entity(entity_group):
    """Crée une entité fusionnée à partir d'un groupe d'entités similaires"""
    if len(entity_group) == 1:
        return entity_group[0].to_dict() if hasattr(entity_group[0], 'to_dict') else entity_group[0]
    
    # Prendre l'entité avec la meilleure confiance comme base
    best_entity = max(entity_group, key=lambda x: x['confidence'] if isinstance(x, dict) else x.confidence)
    
    if hasattr(best_entity, 'to_dict'):
        merged = best_entity.to_dict()
    else:
        merged = dict(best_entity)
    
    # Agréger les méthodes utilisées
    methods = []
    total_confidence = 0
    
    for entity in entity_group:
        ent_dict = entity if isinstance(entity, dict) else entity.to_dict()
        methods.append(ent_dict['method'])
        total_confidence += ent_dict['confidence']
    
    merged['method'] = ', '.join(set(methods))
    merged['confidence'] = total_confidence / len(entity_group)  # Moyenne
    merged['detection_count'] = len(entity_group)
    
    return merged

# --------------------------
# Clustering sémantique
# --------------------------

def cluster_entities(entities, n_clusters=None):
    """Regroupe les entités par similarité sémantique"""
    if len(entities) < 2:
        return entities, []
    
    # Extraire les textes
    texts = [entity['text'] for entity in entities]
    
    # Calculer les embeddings
    embeddings = embedding_model.encode(texts)
    
    # Clustering hiérarchique
    if n_clusters is None:
        # Déterminer automatiquement le nombre de clusters
        n_clusters = min(max(2, len(entities) // 3), 10)
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric='cosine', 
        linkage='average'
    )
    
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Ajouter les labels de cluster aux entités
    clustered_entities = []
    for entity, cluster_id in zip(entities, cluster_labels):
        entity_copy = entity.copy()
        entity_copy['cluster'] = int(cluster_id)
        clustered_entities.append(entity_copy)
    
    # Créer un résumé des clusters
    clusters_summary = []
    for cluster_id in range(n_clusters):
        cluster_entities = [e for e in clustered_entities if e['cluster'] == cluster_id]
        cluster_texts = [e['text'] for e in cluster_entities]
        cluster_labels = [e['label'] for e in cluster_entities]
        
        clusters_summary.append({
            'cluster_id': cluster_id,
            'size': len(cluster_entities),
            'entities': cluster_texts,
            'dominant_label': max(set(cluster_labels), key=cluster_labels.count),
            'avg_confidence': np.mean([e['confidence'] for e in cluster_entities])
        })
    
    return clustered_entities, clusters_summary

# --------------------------
# Interface Streamlit
# --------------------------

def main():
    # Sidebar pour la configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Sélection des méthodes
    st.sidebar.subheader("Méthodes d'extraction")
    use_regex = st.sidebar.checkbox("Regex", value=True)
    use_spacy = st.sidebar.checkbox("spaCy", value=True)
    use_camembert = st.sidebar.checkbox("CamemBERT", value=True)
    use_flair = st.sidebar.checkbox("Flair", value=True)
    use_embeddings = st.sidebar.checkbox("Embeddings sémantiques", value=False)
    
    # Paramètres de fusion et clustering
    st.sidebar.subheader("Paramètres")
    merge_threshold = st.sidebar.slider("Seuil de fusion", 0.5, 1.0, 0.8, 0.05)
    embedding_threshold = st.sidebar.slider("Seuil similarité embeddings", 0.5, 1.0, 0.75, 0.05)
    enable_clustering = st.sidebar.checkbox("Activer le clustering", value=True)
    n_clusters = st.sidebar.number_input("Nombre de clusters (auto si 0)", 0, 20, 0)
    
    # Zone principale
    st.header("📝 Saisie du texte")
    
    # Options d'entrée
    input_method = st.radio("Source du texte:", ["Texte direct", "URL web"])
    
    text_content = ""
    
    if input_method == "Texte direct":
        text_content = st.text_area("Entrez votre texte:", height=200, 
                                   placeholder="Copiez-collez votre texte ici...")
    
    elif input_method == "URL web":
        url = st.text_input("URL de la page web:")
        if url and st.button("Extraire le contenu"):
            with st.spinner("Extraction du contenu web..."):
                try:
                    downloaded = trafilatura.fetch_url(url)
                    text_content = trafilatura.extract(downloaded)
                    if text_content:
                        st.success("Contenu extrait avec succès!")
                        st.text_area("Contenu extrait:", value=text_content[:1000] + "..." if len(text_content) > 1000 else text_content, height=150)
                    else:
                        st.error("Impossible d'extraire le contenu de cette URL")
                except Exception as e:
                    st.error(f"Erreur lors de l'extraction: {e}")
    
    # Traitement
    if text_content and st.button("🚀 Extraire les entités", type="primary"):
        with st.spinner("Extraction des entités en cours..."):
            all_entities = []
            
            # Extraction avec les différentes méthodes
            col1, col2 = st.columns(2)
            
            with col1:
                if use_regex:
                    with st.expander("📋 Regex"):
                        regex_entities = extract_with_regex(text_content)
                        st.write(f"Trouvé {len(regex_entities)} entités")
                        if regex_entities:
                            st.json(regex_entities[:3])  # Afficher les 3 premières
                        all_entities.extend(regex_entities)
                
                if use_spacy:
                    with st.expander("🤖 spaCy"):
                        spacy_entities = extract_with_spacy(text_content)
                        st.write(f"Trouvé {len(spacy_entities)} entités")
                        if spacy_entities:
                            st.json(spacy_entities[:3])
                        all_entities.extend(spacy_entities)
                
                if use_camembert:
                    with st.expander("🥖 CamemBERT"):
                        camembert_entities = extract_with_camembert(text_content)
                        st.write(f"Trouvé {len(camembert_entities)} entités")
                        if camembert_entities:
                            st.json(camembert_entities[:3])
                        all_entities.extend(camembert_entities)
            
            with col2:
                if use_flair:
                    with st.expander("⚡ Flair"):
                        flair_entities = extract_with_flair(text_content)
                        st.write(f"Trouvé {len(flair_entities)} entités")
                        if flair_entities:
                            st.json(flair_entities[:3])
                        all_entities.extend(flair_entities)
                
                if use_embeddings:
                    with st.expander("🧠 Embeddings"):
                        embedding_entities = extract_with_embeddings(text_content, embedding_threshold)
                        st.write(f"Trouvé {len(embedding_entities)} entités")
                        if embedding_entities:
                            st.json(embedding_entities[:3])
                        all_entities.extend(embedding_entities)
        
        # Fusion des entités
        if all_entities:
            st.header("🔗 Fusion et déduplication")
            merged_entities = merge_entities(all_entities, merge_threshold)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entités brutes", len(all_entities))
            with col2:
                st.metric("Après fusion", len(merged_entities))
            with col3:
                reduction = round((1 - len(merged_entities)/len(all_entities)) * 100, 1) if all_entities else 0
                st.metric("Réduction", f"{reduction}%")
            
            # Clustering sémantique
            if enable_clustering and merged_entities:
                st.header("🧠 Clustering sémantique")
                
                n_clusters_final = n_clusters if n_clusters > 0 else None
                clustered_entities, clusters_summary = cluster_entities(merged_entities, n_clusters_final)
                
                # Affichage des clusters
                for cluster_info in clusters_summary:
                    with st.expander(f"Cluster {cluster_info['cluster_id']} - {cluster_info['dominant_label']} ({cluster_info['size']} entités)"):
                        st.write("**Entités:**", ", ".join(cluster_info['entities']))
                        st.write(f"**Confiance moyenne:** {cluster_info['avg_confidence']:.3f}")
                
                final_entities = clustered_entities
            else:
                final_entities = merged_entities
            
            # Tableau final
            st.header("📊 Résultats finaux")
            
            if final_entities:
                df_results = pd.DataFrame(final_entities)
                
                # Filtres
                col1, col2 = st.columns(2)
                with col1:
                    selected_labels = st.multiselect("Filtrer par type:", 
                                                    options=df_results['label'].unique(),
                                                    default=df_results['label'].unique())
                with col2:
                    min_confidence = st.slider("Confiance minimale:", 0.0, 1.0, 0.5, 0.05)
                
                # Appliquer les filtres
                filtered_df = df_results[
                    (df_results['label'].isin(selected_labels)) & 
                    (df_results['confidence'] >= min_confidence)
                ]
                
                # Affichage du tableau
                st.dataframe(filtered_df, use_container_width=True)
                
                # Statistiques
                st.subheader("📈 Statistiques")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total entités", len(filtered_df))
                with col2:
                    st.metric("Types uniques", filtered_df['label'].nunique())
                with col3:
                    avg_conf = filtered_df['confidence'].mean()
                    st.metric("Confiance moy.", f"{avg_conf:.3f}")
                with col4:
                    methods_count = len(set(', '.join(filtered_df['method'].values).split(', ')))
                    st.metric("Méthodes utilisées", methods_count)
                
                # Graphiques
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Distribution par type")
                    label_counts = filtered_df['label'].value_counts()
                    st.bar_chart(label_counts)
                
                with col2:
                    st.subheader("Distribution de confiance")
                    st.histogram_chart(filtered_df['confidence'].values, bins=20)
                
                # Export
                st.subheader("💾 Export")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button("Télécharger CSV", csv_data, "entites.csv", "text/csv")
                
                with col2:
                    json_data = filtered_df.to_json(orient='records', indent=2)
                    st.download_button("Télécharger JSON", json_data, "entites.json", "application/json")
            
            else:
                st.warning("Aucune entité trouvée avec les paramètres actuels.")
        
        else:
            st.warning("Aucune entité détectée dans le texte.")

if __name__ == "__main__":
    main()
