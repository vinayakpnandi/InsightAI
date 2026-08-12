from src.rag.ingest import ingest_pdf


PDF_PATH = "data/test_documents/company_report.pdf"


if __name__ == "__main__":

    result = ingest_pdf(
        file_path=PDF_PATH,
        document_name="company_report.pdf",
    )

    print()
    print("=" * 50)
    print("DOCUMENT INGESTION SUCCESS")
    print("=" * 50)

    print(
        f"Document ID: "
        f"{result['document_id']}"
    )

    print(
        f"Document: "
        f"{result['document_name']}"
    )

    print(
        f"Pages: "
        f"{result['page_count']}"
    )

    print(
        f"Chunks: "
        f"{result['chunk_count']}"
    )