from langchain_core.prompts import ChatPromptTemplate

from src.rag.retriever import get_retriever
from src.utils.llm import get_llm


PROMPT = """
You are InsightAI, an enterprise decision intelligence assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not invent facts.
2. If the answer cannot be found in the context, say:
   "I could not find this information in the uploaded document."
3. Keep the answer clear and concise.
4. Preserve numbers and facts exactly as provided.
5. Do not use outside knowledge.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""


def answer_question(
    question: str,
    document_id: str,
) -> dict:
    """
    Retrieve information only from the specified
    document and generate a grounded answer.
    """

    if not document_id:
        raise ValueError(
            "document_id is required."
        )

    retriever = get_retriever(
        document_id=document_id,
        k=3,
    )

    documents = retriever.invoke(
        question
    )

    if not documents:

        return {
            "answer": (
                "I could not find this information "
                "in the uploaded document."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = ChatPromptTemplate.from_template(
        PROMPT
    )

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    llm = get_llm()

    response = llm.invoke(messages)

    sources = []

    seen = set()

    for document in documents:

        metadata = document.metadata

        source_key = (
            metadata.get("document_id"),
            metadata.get("page"),
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        sources.append(
            {
                "document_id": metadata.get(
                    "document_id"
                ),
                "document_name": metadata.get(
                    "document_name",
                    "Unknown",
                ),
                "source": metadata.get(
                    "source",
                    "Unknown",
                ),
                "page": metadata.get(
                    "page",
                    "Unknown",
                ),
            }
        )

    return {
        "answer": response.content,
        "sources": sources,
    }