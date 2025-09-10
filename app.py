import re
import json
import numpy as np
import pandas as pd
import streamlit as st

# Imports conditionnels avec gestion d'erreurs
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    st.error("Trafilatura non disponible")

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    st.error("Transformers non disponible")

try:
    from sentence_transformers import SentenceTransformer, util
    from sklearn.cluster import AgglomerativeClustering
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    st.error("Sentence Transformers non disponible")

# --------------------------
# Config Streamlit
# --------------------------
st.set_page_config(page_title="Extraction d'entités nommées", layout="wide")
st.title("🔍 Extraction d'entités nommées (NER) multi-méthodes")
st.markdown(
    "Cette app extrait les entités depuis des pages web via **Regex**, **CamemBERT**, "
    "**Embeddings sémantiques**, puis **fusionne et regroupe** les entités par **similarité sémantique**."
)

# Vérification des dépendances
missing_deps = []
if not HAS_TRAFILATURA:
    missing_deps.append("trafilatura")
if not HAS_TRANSFORMERS:
    missing_deps.append("transformers")
if not HAS_SENTENCE_TRANSFORMERS:
    missing_deps.append("sentence-transformers")

if missing_deps:
    st.warning(f"Dépendances manquantes: {', '.join(missing_deps)}. Certaines fonctionnalités seront désactivées.")

# --------------------------
# Modèles (mis en cache)
# --------------------------
@st.cache_resource
def load_camembert():
    if not HAS_TRANSFORMERS:
        return None
    try:
        model_name = "Jean-Baptiste/camembert-ner"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        return hf_pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    except Exception as e:
        st.error(f"Erreur lors du chargement de CamemBERT: {e}")
        return None

@st.cache_resource
def load_embeddings_model():
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle d'embeddings: {e}")
        return None

# Chargement des modèles
camembert_ner = load_camembert()
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
            r'\b[A-ZÀ-ÿ][a-zà-ÿ]+\s+[A-ZÀ-ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]+)?',
            r'\b(?:M\.|Mme|Dr|Prof\.)\s+[A-ZÀ-ÿ][a-zà-ÿ]+',
        ],
        "ORG": [
            r'\b[A-ZÀ-ÿ][a-zà-ÿ]*(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]*)*\s+(?:SA|SAS|SARL|EURL|SNC|GIE)',
            r'\b(?:Société|Entreprise|Compagnie|Association)\s+[A-ZÀ-ÿ][a-zà-ÿ]*(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]*)*',
        ],
        "LOC": [
            r'\b(?:Paris|Lyon|Marseille|Toulouse|Nice|Nantes|Strasbourg|Montpellier|Bordeaux|Lille)',
            r'\b[A-ZÀ-ÿ][a-zà-ÿ]*(?:-[A-ZÀ-ÿ][a-zà-ÿ]*)*(?:\s+sur\s+[A-ZÀ-ÿ][a-zà-ÿ]*)?',
        ],
        "DATE": [
            r'\b\d{1,2}(?:/|-)\d{1,2}(?:/|-)\d{2,4}',
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
                    "confidence": 0.8,
                    "method": "Regex"
                })
    
    return entities

def extract_with_camembert(text):
    """Extraction avec CamemBERT"""
    if not camembert_ner:
        return []
    
    try:
        # Limiter la taille du texte pour éviter les erreurs de mémoire
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
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

def extract_with_embeddings(text, similarity_threshold=0.75):
    """Extraction par similarité sémantique avec des entités connues"""
    if not embedding_model:
        return []
    
    try:
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
        
        for sentence in sentences[:5]:  # Limiter le nombre de phrases
            if len(sentence.strip()) < 10:
                continue
                
            # Extraire les mots/phrases candidates (capitalisés ou expressions)
            candidates = re.findall(r'\b[A-ZÀ-ÿ][a-zà-ÿ]*(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]*)*', sentence)
            
            for candidate in candidates[:10]:  # Limiter le nombre de candidats
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
    except Exception as e:
        st.error(f"Erreur Embeddings: {e}")
        return []

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
    
    # Simple regroupement par texte identique
    merged_entities = []
    processed_texts = set()
    
    for _, entity in df.iterrows():
        if entity['text'] in processed_texts:
            continue
            
        # Chercher des entités avec le même texte
        similar = df[df['text'] == entity['text']]
        
        if len(similar) > 1:
            # Fusionner les méthodes
            methods = ', '.join(similar['method'].unique())
            avg_confidence = similar['confidence'].mean()
            
            merged_entity = {
                'text': entity['text'],
                'label': entity['label'],
                'start': entity['start'],
                'end': entity['end'],
                'confidence': avg_confidence,
                'method': methods,
                'detection_count': len(similar)
            }
        else:
            merged_entity = entity.to_dict()
            merged_entity['detection_count'] = 1
        
        merged_entities.append(merged_entity)
        processed_texts.add(entity['text'])
    
    return merged_entities

# --------------------------
# Clustering sémantique
# --------------------------

def cluster_entities(entities, n_clusters=None):
    """Regroupe les entités par similarité sémantique"""
    if not HAS_SENTENCE_TRANSFORMERS or not embedding_model or len(entities) < 2:
        return entities, []
    
    try:
        # Extraire les textes
        texts = [entity['text'] for entity in entities]
        
        # Calculer les embeddings
        embeddings = embedding_model.encode(texts)
        
        # Clustering hiérarchique
        if n_clusters is None:
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
                'dominant_label': max(set(cluster_labels), key=cluster_labels.count) if cluster_labels else "UNKNOWN",
                'avg_confidence': np.mean([e['confidence'] for e in cluster_entities])
            })
        
        return clustered_entities, clusters_summary
    
    except Exception as e:
        st.error(f"Erreur clustering: {e}")
        return entities, []

# --------------------------
# Interface Streamlit
# --------------------------

def main():
    # Sidebar pour la configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Sélection des méthodes
    st.sidebar.subheader("Méthodes d'extraction")
    use_regex = st.sidebar.checkbox("Regex", value=True)
    use_camembert = st.sidebar.checkbox("CamemBERT", value=HAS_TRANSFORMERS and camembert_ner is not None)
    use_embeddings = st.sidebar.checkbox("Embeddings sémantiques", value=HAS_SENTENCE_TRANSFORMERS and embedding_model is not None)
    
    # Paramètres de fusion et clustering
    st.sidebar.subheader("Paramètres")
    merge_threshold = st.sidebar.slider("Seuil de fusion", 0.5, 1.0, 0.8, 0.05)
    embedding_threshold = st.sidebar.slider("Seuil similarité embeddings", 0.5, 1.0, 0.75, 0.05)
    enable_clustering = st.sidebar.checkbox("Activer le clustering", value=HAS_SENTENCE_TRANSFORMERS and embedding_model is not None)
    n_clusters = st.sidebar.number_input("Nombre de clusters (auto si 0)", 0, 20, 0)
    
    # Zone principale
    st.header("📝 Saisie du texte")
    
    # Options d'entrée
    input_method = st.radio("Source du texte:", ["Texte direct", "URL web"] if HAS_TRAFILATURA else ["Texte direct"])
    
    text_content = ""
    
    if input_method == "Texte direct":
        text_content = st.text_area("Entrez votre texte:", height=200, 
                                   placeholder="Copiez-collez votre texte ici...")
    
    elif input_method == "URL web" and HAS_TRAFILATURA:
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
                            st.json(regex_entities[:3])
                        all_entities.extend(regex_entities)
                
                if use_camembert and camembert_ner:
                    with st.expander("🥖 CamemBERT"):
                        camembert_entities = extract_with_camembert(text_content)
                        st.write(f"Trouvé {len(camembert_entities)} entités")
                        if camembert_entities:
                            st.json(camembert_entities[:3])
                        all_entities.extend(camembert_entities)
            
            with col2:
                if use_embeddings and embedding_model:
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
            if enable_clustering and merged_entities and embedding_model:
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
