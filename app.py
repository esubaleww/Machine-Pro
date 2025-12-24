import streamlit as st  # type: ignore
import pickle

st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📩",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        position: fixed;
        height: 100vh;
        overflow: hidden;
        background: linear-gradient(135deg, #0f172a, #1f2937);
        color: #e5e7eb;
    }

    .title-text {
        font-size: 2.3rem;
        font-weight: 700;
        color: #f9fafb;
        text-align: center;
        margin-bottom: 0.25rem;
    }

    .subtitle-text {
        font-size: 0.95rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 1rem;
    }

    .card {
        background-color: #111827;
        border-radius: 12px;
        padding: 1.0rem 1.8rem 1.5rem 1.8rem;  /* slightly smaller top padding */
        box-shadow: 0 18px 45px rgba(0,0,0,0.35);
        border: 1px solid #1f2937;
    }

    /* Custom wrapper for the result to reduce top margin */
    .result-wrapper > div {
        margin-top: 0.25rem !important;  /* pull the alert closer to the top */
        margin-bottom: 0.5rem !important;
    }

    textarea {
        border-radius: 8px !important;
        border: 1px solid #4b5563 !important;
        background-color: #020617 !important;
        color: #e5e7eb !important;
        font-size: 0.95rem !important;
    }

    .stForm button[kind="primary"] {
        border-radius: 999px;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        font-weight: 600;
        border: none;
    }

    .stForm button[kind="primary"]:hover {
        filter: brightness(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load model & vectorizer ---
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.markdown('<div class="title-text">Spam Detection App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Paste any SMS or message below to see how likely it is to be spam.</div>',
    unsafe_allow_html=True,
)

# Initialize session state
if "message" not in st.session_state:
    st.session_state.message = ""

def clear_message():
    st.session_state.message = ""

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Wrap the result placeholder in a div with a custom class
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
