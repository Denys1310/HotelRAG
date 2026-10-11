
import streamlit as st

from rag import ask_question, load_vectorstore

st.set_page_config(
    page_title="Hotel RAG Assistant",
    page_icon="🏨",
    layout="wide",
)

st.title("🏨 Hotel RAG Assistant")
st.write(
    "Інтелектуальний асистент готелю SMALLHOTEL. "
    "Відповіді формуються на основі PDF-документів."
)


@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


with st.sidebar:
    st.header("Налаштування пошуку")

    top_k = st.slider(
        "Кількість фрагментів (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
    )

    search_type = st.selectbox(
        "Метод пошуку",
        options=["mmr", "similarity"],
    )

    st.caption(
        "MMR враховує різноманітність результатів. "
        "Similarity шукає найбільш схожі фрагменти."
    )


question = st.text_area(
    "Поставте запитання про готель",
    placeholder="Наприклад: О котрій годині заселення?",
    height=100,
)

if st.button("Отримати відповідь", type="primary"):
    if not question.strip():
        st.warning("Введіть запитання.")
    else:
        try:
            vectorstore = get_vectorstore()

            with st.spinner("Шукаємо відповідь..."):
                answer, sources = ask_question(
                    question=question.strip(),
                    vectorstore=vectorstore,
                    top_k=top_k,
                    search_type=search_type,
                )

            st.subheader("Відповідь")
            st.write(answer)

            st.subheader("Використані фрагменти")

            if not sources:
                st.info("Релевантних фрагментів не знайдено.")

            for index, document in enumerate(sources, start=1):
                filename = document.metadata.get(
                    "source", "Невідомий документ"
                )
                page = document.metadata.get("page", "?")

                with st.expander(
                    f"Джерело {index}: {filename}, сторінка {page}"
                ):
                    st.write(document.page_content)

        except Exception as error:
            st.error(f"Помилка: {error}")
