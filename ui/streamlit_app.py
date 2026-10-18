import streamlit as st
import requests

st.title("RAG Knowledge Assistant")

question = st.text_input("Ask a question, The Assistant will answer based on the ingested documents.")

if st.button("Submit") and question:
    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"question": question},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        st.error(f"The request failed: {exc}")
    else:
        st.subheader("Answer")
        st.write(data["answer"])

        st.subheader("Retrieved Sources")
        for source in data["sources"]:
            ranking = [f"RRF {source['hybrid_score']:.5f}"]
            if source["vector_similarity"] is not None:
                ranking.append(f"vector {source['vector_similarity']:.3f}")
            if source["keyword_rank"] is not None:
                ranking.append(f"keyword rank {source['keyword_rank']}")
            label = (
                f"[{source['id']}] {source['source']} · chunk {source['chunk_index']} "
                f"· {' · '.join(ranking)}"
            )
            with st.expander(label):
                st.write(source["content"])
