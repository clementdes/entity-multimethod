import re
import trafilatura
import streamlit as st
import spacy
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger

# --------------------------
# Initialisation des modèles
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

nlp_spacy = load_spacy()
camembert_ner = load_camembert()
flair_tagger = load_flair()

# --------------------------
# Extraction des entités par regex
# --------------------------
def extract_entities_regex(text):
    entities = []
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"\b(?:\+33|0)[1-9](?:[\s.-]?\d{2}){4}\b"
    date_pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b"

    entities.extend(re.findall(email_pattern, text))
    entities.extend(re.findall(phone_pattern, text))
    entities.extend(re.findall(date_pattern, text))

    return list(set(entities))

# --------------------------
# Extraction avec spaCy
# --------------------------
def extract_entities_spacy(text):
    doc = nlp_spacy(text)
    return list(set([ent.text for ent in doc.ents]))

# --------------------------
# Extraction avec CamemBERT
# --------------------------
def extract_entities_camembert(text):
    results = camembert_ner(text)
    return list(set([ent["word"] for ent in results]))

# --------------------------
# Extraction avec Flair
# --------------------------
def extract_entities_flair(text):
    sentence = Sentence(text)
    flair_tagger.predict(sentence)
    return list(set([entity.text for entity in sentence.get_spans("ner")]))

# --------------------------
# Extraction du texte d'une URL
# --------------------------
def get_clean_text_from_url(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded)
    return ""

# --------------------------
# Interface Streamlit
# --------------------------
st.set_page_config(page_title="Extraction d'entités nommées", layout="wide")
st.title("🔍 Application d'extraction d'entités nommées")
st.markdown("Cette application analyse le contenu des pages web et extrait les entités avec plusieurs méthodes NLP.")

urls_input = st.text_area("Entrez une liste d'URLs (une par ligne) :")

if st.button("Analyser"):
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

    if not urls:
        st.warning("⚠️ Merci de saisir au moins une URL.")
    else:
        all_entities = set()

        for url in urls:
            st.subheader(f"🌐 Analyse de : {url}")
            text = get_clean_text_from_url(url)

            if not text:
                st.error("Impossible d'extraire le contenu de la page.")
                continue

            col1, col2, col3, col4 = st.columns(4)

            # Méthode Regex
            regex_entities = extract_entities_regex(text)
            with col1:
                st.write("**Regex & Règles**")
                st.write(regex_entities)
            all_entities.update(regex_entities)

            # Méthode spaCy
            spacy_entities = extract_entities_spacy(text)
            with col2:
                st.write("**spaCy**")
                st.write(spacy_entities)
            all_entities.update(spacy_entities)

            # Méthode CamemBERT
            camembert_entities = extract_entities_camembert(text)
            with col3:
                st.write("**CamemBERT**")
                st.write(camembert_entities)
            all_entities.update(camembert_entities)

            # Méthode Flair
            flair_entities = extract_entities_flair(text)
            with col4:
                st.write("**Flair**")
                st.write(flair_entities)
            all_entities.update(flair_entities)

        st.subheader("🧩 Liste consolidée des entités uniques")
        st.write(sorted(all_entities))
