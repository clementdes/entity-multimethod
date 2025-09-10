import re
import io
import json
import numpy as np
import pandas as pd
import trafilatura
import streamlit as st

import spacy
from transformers import pipeline
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
    return pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)

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
embeddin
