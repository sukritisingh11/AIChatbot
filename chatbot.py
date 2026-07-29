import streamlit as st
import pandas as pd
import pickle
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI CSV Chatbot",
    page_icon="🤖",
    layout="centered"
)


# =========================
# CSS
# =========================

st.markdown("""
<style>

.stApp {
    background:#0e1117;
}

h1 {
    text-align:center;
    color:#00ffff;
}

[data-testid="stSidebar"] {
    background:#171717;
}

</style>
""", unsafe_allow_html=True)



# =========================
# MODEL CLASS
# =========================

class ChatBotModel:

    def __init__(self, df):

        self.df = df

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

        self.vectors = self.vectorizer.fit_transform(
            df["text"].astype(str)
        )


    def predict(self, question):

        query_vector = self.vectorizer.transform(
            [question]
        )


        similarity = cosine_similarity(
            query_vector,
            self.vectors
        )


        index = similarity.argmax()

        confidence = similarity.max()


        if confidence < 0.25:

            return (
                "Sorry, I don't understand.",
                "unknown",
                confidence
            )


        return (
            self.df.iloc[index]["response"],
            self.df.iloc[index]["intent"],
            confidence
        )


    def suggestions(self):

        return self.df["text"].sample(
            min(5,len(self.df))
        ).tolist()



# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():

    return pd.read_csv(
        "chatbot.csv"
    )



df = load_data()



# =========================
# CREATE / LOAD PKL MODEL
# =========================

MODEL_FILE = "chatbot_model.pkl"



def load_model():

    if os.path.exists(MODEL_FILE):

        with open(
            MODEL_FILE,
            "rb"
        ) as file:

            model = pickle.load(file)

    else:

        model = ChatBotModel(df)

        with open(
            MODEL_FILE,
            "wb"
        ) as file:

            pickle.dump(
                model,
                file
            )

    return model



bot = load_model()



# =========================
# TITLE
# =========================

st.title("🤖 AI CSV Chatbot")

st.caption(
    "TF-IDF model loaded from PKL file"
)



# =========================
# SESSION
# =========================

if "messages" not in st.session_state:

    st.session_state.messages=[]



# =========================
# SIDEBAR
# =========================

with st.sidebar:


    st.header("⚙ Settings")


    if st.button("🗑 Clear Chat"):

        st.session_state.messages=[]

        st.rerun()



    st.subheader("💡 Suggested Questions")


    for q in bot.suggestions():

        st.write(
            "• "+q
        )



    st.success(
        "✅ Model Loaded: chatbot_model.pkl"
    )



    chat=""


    for msg in st.session_state.messages:

        chat += (
            msg["role"]
            + ": "
            + msg["content"]
            + "\n\n"
        )


    st.download_button(
        "📥 Download Chat",
        chat,
        file_name="chat.txt"
    )



# =========================
# CHAT DISPLAY
# =========================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )



# =========================
# USER INPUT
# =========================

prompt = st.chat_input(
    "Ask something..."
)



if prompt:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
        }
    )


    with st.chat_message(
        "user"
    ):

        st.write(prompt)



    response,intent,confidence = bot.predict(
        prompt
    )


    answer=f"""
{response}

🏷 Intent: {intent}

🎯 Confidence: {confidence*100:.2f}%
"""


    with st.chat_message(
        "assistant"
    ):

        box=st.empty()

        text=""

        for word in answer.split():

            text += word+" "

            box.write(text)

            time.sleep(0.03)



    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )