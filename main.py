import os
from pathlib import Path
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="AI Document Analyst")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the document excerpts. "
            "If the excerpts do not contain the answer, say that you "
            "cannot find it in the uploaded documents. "
            "Do not add facts from outside the excerpts.\n\n"
            "Excerpts:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


class Question(BaseModel):
    question: str = Field(min_length=1)


@app.get("/health")
async def health_check():
    return {"message": "Hello World", "status": "ok"}


@app.post("/documents")
def upload_document(file: UploadFile):
    contents = file.file.read()
    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF must be 10 MB or smaller.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not any(page.page_content.strip() for page in pages):
        raise HTTPException(
            status_code=422,
            detail="No text found. A scanned PDF needs OCR before this step.",
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    ).split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source"] = file.filename or "upload.pdf"

    vectorstore.add_documents(chunks)
    return {
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
    }


@app.post("/ask/")
def ask(body: Question):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set.")

    docs = vectorstore.similarity_search(body.question, k=4)
    if not docs:
        raise HTTPException(
            status_code=404,
            detail="No documents are stored yet. Upload a PDF first.",
        )

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'document')}, "
        f"page {doc.metadata.get('page', '?')}\n{doc.page_content}"
        for doc in docs
    )
    message = (PROMPT | llm).invoke(
        {"context": context, "question": body.question}
    )

    return {
        "question": body.question,
        "answer": message.content,
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
            }
            for doc in docs
        ],
    }