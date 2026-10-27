from pathlib import Path
import json

import streamlit as st
import joblib

from banking77 import predire_intention


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "modele_intention_banking77.joblib"

RESPONSES_PATH = BASE_DIR / "reponses.json"

EXEMPLES = [
    "I have not received my new card yet",
    "Why was I charged twice for this payment?",
    "Why have I been charged a fee for cash withdrawal?",
    "How can I create a virtual card for online payments?",
]


@st.cache_resource
def charger_modele():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Prior execution of the notebook through "
            "its model-save cell is required."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def charger_reponses():
    return json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))


st.set_page_config(
    page_title="Banking Assistant — Banking77",
    page_icon="💳",
    layout="centered",
)

st.title("💳 Banking Assistant")
st.caption("NLP demo · Intent classification with Banking77")
st.write(
    "The application classifies banking questions written in English. "
    "The model predicts the intent and displays a "
    "predefined demo response."
)

with st.sidebar:
    st.header("Examples")
    st.selectbox("Example question", EXEMPLES, key="question_exemple")
    if st.button("Example insertion", use_container_width=True):
        st.session_state["question"] = st.session_state["question_exemple"]

    st.divider()
    st.caption(
        "These are generic demo responses. They do not replace official information "
        "from a bank."
    )

with st.form("formulaire_question"):
    question = st.text_area(
        "Banking question",
        key="question",
        height=120,
        placeholder="Ex. How can I create a virtual card for online payments?",
    )
    envoyer = st.form_submit_button("Question classification", type="primary")

if envoyer:
    if not question.strip():
        st.warning("A non-empty question is required for classification.")
    else:
        try:
            modele = charger_modele()
            reponses = charger_reponses()
            if set(modele.classes_) != set(reponses):
                raise ValueError("The response catalogue does not match the model's intent labels.")
            intention = predire_intention(question.strip(), modele)
            reponse = reponses[intention]

            st.subheader("Result")
            st.markdown(f"**Predicted intent:** `{intention}`")
            st.success(reponse)
        except FileNotFoundError as erreur:
            st.error(str(erreur))
        except ValueError as erreur:
            st.warning(str(erreur))
        except Exception as erreur:
            st.error(f"Could not load or run the model: {erreur}")

st.divider()
st.caption("77 banking intents · Predefined demonstration responses · English questions")
