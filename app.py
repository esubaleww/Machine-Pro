import streamlit as st  
import pickle

st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📩",
    layout="centered",
)

st.markdown(
    """
<style>


html, body, [data-testid="stAppViewContainer"] {
    margin: 0 !important;
    padding: 0 !important;
    height: 100%;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1f2937) !important;
    color: #e5e7eb !important;
}


.title-text {
    font-size: 2.3rem;
    font-weight: 700;
    color: #f9fafb !important;
    text-align: center;
    margin-bottom: 0.25rem;
}

.subtitle-text {
    font-size: 0.95rem;
    color: #9ca3af !important;
    text-align: center;
    margin-bottom: 1rem;
}


.card {
    background-color: #111827 !important;
    border-radius: 12px;
    padding: 1rem 1.8rem 1.5rem;
    border: 1px solid #1f2937;
    box-shadow: 0 18px 45px rgba(0,0,0,0.35);
}


textarea, 
[data-testid="stTextArea"] textarea {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
    border: 1px solid #4b5563 !important;
    font-size: 0.95rem !important;
}
textarea::placeholder,
input::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;   /* Firefox fix */
}


textarea::-moz-placeholder,
input::-moz-placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}


textarea:-ms-input-placeholder,
input:-ms-input-placeholder {
    color: #9ca3af !important;
}


textarea::-webkit-input-placeholder,
input::-webkit-input-placeholder {
    color: #9ca3af !important;
}


label, 
[data-testid="stWidgetLabel"] {
    color: #e5e7eb !important;
    font-weight: 500;
}


button,
.stForm button,
[data-testid="baseButton-primary"],
[data-testid="baseButton-secondary"] {
     background: linear-gradient(135deg, #064e3b, #022c22) !important;
    color: white !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    -webkit-appearance: none;
    appearance: none;
}

button:hover {
    filter: brightness(1.05);
}

button:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* Clear button */
[data-testid="baseButton-secondary"] {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
}

[data-testid="baseButton-secondary"]:hover {
    background: #374151 !important;
}

/* ---------------- ALERTS ---------------- */
[data-testid="stAlert"] {
    background-color: #020617 !important;
    border: 1px solid #374151 !important;
    color: #e5e7eb !important;
}

/* ---------------- FOOTER ---------------- */
.app-footer span {
    background: rgba(15, 23, 42, 0.8);
    color: white !important;
    border-radius: 999px;
    border: 1px solid #374151;
}

</style>
""",
unsafe_allow_html=True,
)



with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.markdown('<div class="title-text">Spam Detection App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Paste any SMS or message below to see how likely it is to be spam.</div>',
    unsafe_allow_html=True,
)

if "message" not in st.session_state:
    st.session_state.message = ""

def clear_message():
    st.session_state.message = ""

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="result-wrapper">', unsafe_allow_html=True)
    result_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("spam_form"):
        st.text_area(
            "Message",
            height=150,
            placeholder="Type or paste a message here...",
            key="message",
        )

        col1, col2 = st.columns(2)

        with col1:
            submitted = st.form_submit_button("Check message")

        with col2:
            st.form_submit_button("Clear", on_click=clear_message)

    if submitted:
        text = st.session_state.message.strip()
        if not text:
            result_placeholder.warning("Please enter a message before checking.")
        else:
            vec = vectorizer.transform([text])
            prob_spam = model.predict_proba(vec)[0][1]

            low = 0.48
            high = 0.54

            if prob_spam >= high:
                result_placeholder.error(f"🚫 Likely SPAM (p_spam={prob_spam:.2f})")
            elif prob_spam <= low:
                result_placeholder.success(
                    f"✔ Likely NOT spam (p_spam={prob_spam:.2f})"
                )
            else:
                result_placeholder.warning(
                    f"⚠ Borderline message – could be spam (p_spam={prob_spam:.2f})"
                )

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="app-footer">
        <span>Built by Group 2 members</span>
    </div>
    """,
    unsafe_allow_html=True,
)
