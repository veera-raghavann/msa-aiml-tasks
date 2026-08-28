import streamlit as st

from rag_bot import MiniRAG

st.set_page_config(page_title="NimbusNote RAG Q&A", page_icon="Q")
st.title("NimbusNote — Mini RAG Q&A")
st.caption("Answers are grounded in the provided documents and show their retrieval source.")

@st.cache_resource
def load_bot():
    return MiniRAG()

bot = load_bot()

question = st.text_input(
    "Ask a question about NimbusNote",
    placeholder="e.g. How often does NimbusNote sync in the background?",
)

if question:
    result = bot.answer(question)

    st.subheader("Answer")
    if result["grounded"]:
        st.write(result["answer"])
        st.success(f"Source: {result['citation']} | similarity: {result['top_score']:.3f}")
    else:
        st.info(result["answer"])

    st.subheader("Retrieved evidence")
    for passage, score in result["sources"]:
        with st.expander(f"{passage.doc_name} — passage {passage.passage_id} — {score:.3f}"):
            st.write(passage.text)

st.divider()
st.caption("If the best retrieved passage is below the evidence threshold, the bot refuses to answer rather than guessing.")
