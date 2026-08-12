import re

from src.database.database import execute_query
from src.database.schema import format_schema_for_llm
from src.utils.llm import get_sql_llm


# ============================================================
# SQL GENERATION PROMPT
# ============================================================

SQL_PROMPT = """
You are the SQL engine for InsightAI.

Your task is to convert the user's business question into
ONE valid SQLite read-only SQL query.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

STRICT RULES:

1. Return ONLY the SQL query.
2. The query must begin with SELECT.
3. Use ONLY tables and columns present in the schema.
4. Use SQLite syntax.
5. The table containing uploaded data is uploaded_data.
6. Do NOT use INSERT.
7. Do NOT use UPDATE.
8. Do NOT use DELETE.
9. Do NOT use DROP.
10. Do NOT use ALTER.
11. Do NOT use CREATE.
12. Do NOT use REPLACE.
13. Do NOT use TRUNCATE.
14. Do NOT use ATTACH.
15. Do NOT use DETACH.
16. Do NOT use PRAGMA.
17. Do NOT return multiple queries.
18. Do NOT use Markdown.
19. Do NOT explain the query.
20. Do NOT output <think>...</think>.
21. Do NOT include comments.

Example:

Question:
show revenue by country

Output:
SELECT Country, SUM(Revenue) AS total_revenue
FROM uploaded_data
GROUP BY Country;

Return ONLY the SQL.
"""


# ============================================================
# BUSINESS ANSWER PROMPT
# ============================================================

ANSWER_PROMPT = """
You are InsightAI, an enterprise decision intelligence
assistant.

Answer the user's business question using ONLY the
provided SQL result.

USER QUESTION:
{question}

SQL QUERY:
{sql_query}

QUERY RESULT:
{query_result}

RULES:

1. Answer the user's question directly.
2. Use only information contained in the query result.
3. Do not invent values.
4. Keep the answer concise.
5. Highlight the most important business insight.
6. If the result contains categories, identify important
   high/low performers when appropriate.
7. Do not generate SQL.
8. Do not mention these instructions.

Return a concise business answer.
"""


# ============================================================
# CLEAN SQL RESPONSE
# ============================================================

def clean_sql_response(response: str) -> str:
    """
    Extract SQL from an LLM response.

    Handles:

        <think>...</think>
        Markdown code fences
        SQL:
        Query:
        explanatory text
        trailing explanations
    """

    if response is None:
        return ""

    sql = str(response).strip()

    if not sql:
        return ""

    # --------------------------------------------------------
    # Remove Qwen thinking blocks
    # --------------------------------------------------------

    sql = re.sub(
        r"<think>.*?</think>",
        "",
        sql,
        flags=re.DOTALL | re.IGNORECASE,
    )

    sql = re.sub(
        r"<think>.*",
        "",
        sql,
        flags=re.DOTALL | re.IGNORECASE,
    )

    sql = re.sub(
        r"</think>",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = sql.strip()

    # --------------------------------------------------------
    # Remove Markdown fences
    # --------------------------------------------------------

    sql = re.sub(
        r"```(?:sql|sqlite)?",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = sql.replace(
        "```",
        "",
    )

    sql = sql.strip()

    # --------------------------------------------------------
    # Remove common prefixes
    # --------------------------------------------------------

    sql = re.sub(
        r"^(SQL|QUERY)\s*:\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = sql.strip()

    # --------------------------------------------------------
    # Find SELECT or WITH
    # --------------------------------------------------------

    select_match = re.search(
        r"\bSELECT\b",
        sql,
        flags=re.IGNORECASE,
    )

    with_match = re.search(
        r"\bWITH\b",
        sql,
        flags=re.IGNORECASE,
    )

    # Pick whichever occurs first.
    matches = [
        match
        for match in (
            select_match,
            with_match,
        )
        if match is not None
    ]

    if not matches:
        return ""

    first_match = min(
        matches,
        key=lambda match: match.start(),
    )

    sql = sql[
        first_match.start():
    ]

    # --------------------------------------------------------
    # Keep only first SQL statement
    # --------------------------------------------------------

    semicolon_index = sql.find(";")

    if semicolon_index != -1:

        sql = sql[
            :semicolon_index + 1
        ]

    else:

        # Remove common explanation after SQL.
        sql = re.split(
            r"\n\s*(?:Explanation|Reason|Here|Note|This query)\s*:",
            sql,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

    return sql.strip()


# ============================================================
# SQL VALIDATION
# ============================================================

def validate_sql(sql: str) -> bool:
    """
    Validate that the query is a safe read-only SQL query.
    """

    if not sql:
        return False

    cleaned_sql = clean_sql_response(
        sql
    )

    if not cleaned_sql:
        return False

    # --------------------------------------------------------
    # Must start with SELECT or WITH
    # --------------------------------------------------------

    if not re.match(
        r"^(SELECT|WITH)\b",
        cleaned_sql,
        flags=re.IGNORECASE,
    ):
        return False

    # --------------------------------------------------------
    # Reject multiple SQL statements
    # --------------------------------------------------------

    sql_without_final_semicolon = (
        cleaned_sql
        .rstrip()
        .rstrip(";")
        .strip()
    )

    if ";" in sql_without_final_semicolon:
        return False

    # --------------------------------------------------------
    # Remove comments before checking dangerous operations
    # --------------------------------------------------------

    check_sql = re.sub(
        r"--.*?$",
        "",
        cleaned_sql,
        flags=re.MULTILINE,
    )

    check_sql = re.sub(
        r"/\*.*?\*/",
        "",
        check_sql,
        flags=re.DOTALL,
    )

    # --------------------------------------------------------
    # Forbidden SQL operations
    # --------------------------------------------------------

    forbidden_operations = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "TRUNCATE",
        "ATTACH",
        "DETACH",
        "VACUUM",
        "PRAGMA",
    ]

    upper_sql = check_sql.upper()

    for operation in forbidden_operations:

        if re.search(
            rf"\b{operation}\b",
            upper_sql,
        ):
            return False

    return True


# ============================================================
# GENERATE SQL
# ============================================================

def generate_sql(
    question: str,
    table_name: str = "uploaded_data",
) -> str:
    """
    Generate SQL using the dedicated SQL LLM.
    """

    if not question or not question.strip():
        raise ValueError(
            "Please enter a business question."
        )

    # IMPORTANT:
    # Use dedicated SQL model instead of generic get_llm().
    llm = get_sql_llm()

    schema = format_schema_for_llm(
        table_name
    )

    prompt = SQL_PROMPT.format(
        schema=schema,
        question=question.strip(),
    )

    print()
    print(
        "[InsightAI] Generating SQL..."
    )

    response = llm.invoke(
        prompt
    )

    raw_response = getattr(
        response,
        "content",
        "",
    )

    # --------------------------------------------------------
    # Handle empty content safely
    # --------------------------------------------------------

    if raw_response is None:
        raw_response = ""

    raw_response = str(
        raw_response
    )

    print()
    print(
        "[InsightAI] RAW SQL RESPONSE:"
    )

    print(
        repr(raw_response)
    )

    # --------------------------------------------------------
    # If content is empty, inspect additional kwargs
    # --------------------------------------------------------

    if not raw_response.strip():

        additional_kwargs = getattr(
            response,
            "additional_kwargs",
            {},
        )

        reasoning_content = (
            additional_kwargs.get(
                "reasoning_content",
                "",
            )
            if isinstance(
                additional_kwargs,
                dict,
            )
            else ""
        )

        if reasoning_content:
            print(
                "[InsightAI] SQL response content "
                "was empty; reasoning content was present."
            )

    sql = clean_sql_response(
        raw_response
    )

    print()
    print(
        "[InsightAI] CLEANED SQL:"
    )

    print(
        repr(sql)
    )

    return sql


# ============================================================
# RETRY SQL GENERATION
# ============================================================

def retry_sql_generation(
    question: str,
    table_name: str = "uploaded_data",
) -> str:
    """
    Retry SQL generation using a minimal prompt.

    This is used when the first response is empty or invalid.
    """

    llm = get_sql_llm()

    schema = format_schema_for_llm(
        table_name
    )

    retry_prompt = f"""
Return exactly ONE SQLite SELECT statement.

SCHEMA:
{schema}

QUESTION:
{question}

Rules:
- Return ONLY SQL.
- Start with SELECT.
- Use uploaded_data.
- No explanation.
- No markdown.
- No <think>.
- No comments.
- No INSERT.
- No UPDATE.
- No DELETE.
- No DROP.
- No ALTER.
- No CREATE.
- No multiple statements.

Example:
SELECT Country, SUM(Revenue) AS total_revenue
FROM uploaded_data
GROUP BY Country;
""".strip()

    print()
    print(
        "[InsightAI] Retrying SQL generation..."
    )

    response = llm.invoke(
        retry_prompt
    )

    raw_response = getattr(
        response,
        "content",
        "",
    )

    if raw_response is None:
        raw_response = ""

    raw_response = str(
        raw_response
    )

    print()
    print(
        "[InsightAI] RETRY RAW RESPONSE:"
    )

    print(
        repr(raw_response)
    )

    sql = clean_sql_response(
        raw_response
    )

    print()
    print(
        "[InsightAI] RETRY CLEANED SQL:"
    )

    print(
        repr(sql)
    )

    return sql


# ============================================================
# EXECUTE SQL
# ============================================================

def execute_sql(
    sql: str,
):
    """
    Validate and execute a read-only SQL query.
    """

    cleaned_sql = clean_sql_response(
        sql
    )

    if not validate_sql(
        cleaned_sql
    ):
        raise ValueError(
            "The generated SQL query was rejected "
            "because it was not a safe read-only query."
        )

    print()
    print(
        "[InsightAI] Executing SQL:"
    )

    print(
        cleaned_sql
    )

    return execute_query(
        cleaned_sql
    )


# ============================================================
# GENERATE BUSINESS ANSWER
# ============================================================

def generate_business_answer(
    question: str,
    sql_query: str,
    columns: list,
    rows: list,
) -> str:
    """
    Convert SQL results into a concise business answer.
    """

    # Business answer can use the same reliable Qwen3:4b model.
    llm = get_sql_llm()

    query_result = {
        "columns": columns,
        "rows": rows,
    }

    prompt = ANSWER_PROMPT.format(
        question=question,
        sql_query=sql_query,
        query_result=query_result,
    )

    response = llm.invoke(
        prompt
    )

    answer = getattr(
        response,
        "content",
        "",
    )

    if answer is None:
        return ""

    return str(
        answer
    ).strip()


# ============================================================
# COMPLETE SQL AGENT
# ============================================================

def ask_sql_agent(
    question: str,
    table_name: str = "uploaded_data",
) -> dict:
    """
    Complete InsightAI SQL workflow.

    User question
          ↓
    Generate SQL
          ↓
    Clean SQL
          ↓
    Validate SQL
          ↓
    Retry if necessary
          ↓
    Execute SQL
          ↓
    Generate business answer
    """

    if not question or not question.strip():
        raise ValueError(
            "Please enter a business question."
        )

    # --------------------------------------------------------
    # FIRST ATTEMPT
    # --------------------------------------------------------

    sql_query = generate_sql(
        question=question,
        table_name=table_name,
    )

    # --------------------------------------------------------
    # VALIDATE FIRST ATTEMPT
    # --------------------------------------------------------

    if not validate_sql(
        sql_query
    ):

        print()
        print(
            "[InsightAI] First SQL validation failed."
        )

        # ----------------------------------------------------
        # RETRY
        # ----------------------------------------------------

        sql_query = retry_sql_generation(
            question=question,
            table_name=table_name,
        )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if not validate_sql(
        sql_query
    ):

        print()
        print(
            "[InsightAI] FINAL SQL:"
        )

        print(
            repr(sql_query)
        )

        raise ValueError(
            "The generated SQL query was rejected "
            "because it was not a safe read-only query."
        )

    # --------------------------------------------------------
    # EXECUTE QUERY
    # --------------------------------------------------------

    columns, rows = execute_sql(
        sql_query
    )

    # --------------------------------------------------------
    # BUSINESS ANSWER
    # --------------------------------------------------------

    answer = generate_business_answer(
        question=question,
        sql_query=sql_query,
        columns=columns,
        rows=rows,
    )

    # --------------------------------------------------------
    # RETURN COMPLETE RESULT
    # --------------------------------------------------------

    return {
        "question": question,
        "sql": sql_query,
        "columns": columns,
        "rows": rows,
        "answer": answer,
    }