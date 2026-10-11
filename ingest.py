import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import RateLimitError

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_BATCH_SIZE = 10
BATCH_DELAY_SECONDS = 15
MAX_RETRIES = 5


def normalize_text(text: str) -> str:
    """Прибирає зайві пробіли та переноси рядків із тексту PDF."""
    return re.sub(r"\s+", " ", text).strip()


def create_embeddings(chunks, embeddings):
    vectorstore = None

    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(
                        documents=batch,
                        embedding=embeddings,
                    )
                else:
                    vectorstore.add_documents(batch)

                print(
                    f"Оброблено чанків: {start + len(batch)}/{len(chunks)}",
                    flush=True,
                )
                break

            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise

                wait_time = 60 * (attempt + 1)
                print(
                    f"Перевищено ліміт OpenAI API. "
                    f"Повторна спроба через {wait_time} с.",
                    flush=True,
                )
                time.sleep(wait_time)

        if start + EMBEDDING_BATCH_SIZE < len(chunks):
            time.sleep(BATCH_DELAY_SECONDS)

    return vectorstore


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("У папці documents немає PDF-файлів.")

    documents = []

    for pdf_file in pdf_files:
        pages = PyPDFLoader(str(pdf_file)).load()

        for page in pages:
            page.metadata["source"] = pdf_file.name
            page.metadata["page"] = page.metadata.get("page", 0) + 1
            page.page_content = normalize_text(page.page_content)

        readable_pages = [page for page in pages if page.page_content]

        print(
            f"{pdf_file.name}: "
            f"{len(readable_pages)}/{len(pages)} сторінок із текстом",
            flush=True,
        )

        if len(readable_pages) != len(pages):
            print(
                f"Увага: {pdf_file.name} містить сторінки "
                "без доступного тексту. Може знадобитися OCR.",
                flush=True,
            )

        documents.extend(readable_pages)

    if not documents:
        raise ValueError("Не вдалося витягнути текст із PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)
    print(f"Створено чанків: {len(chunks)}", flush=True)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=EMBEDDING_BATCH_SIZE,
        check_embedding_ctx_length=False,
        max_retries=2,
    )

    vectorstore = create_embeddings(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print("FAISS-індекс успішно збережено.", flush=True)


if __name__ == "__main__":
    main()
