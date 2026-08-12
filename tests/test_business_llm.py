import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.llm import get_business_llm


print("=" * 60)
print("BUSINESS LLM TEST")
print("=" * 60)

print("\nLoading Qwen3:4b...")

llm = get_business_llm()

print("Model loaded.")

print("\nSending test prompt...")

response = llm.invoke(
    'Return ONLY this JSON: {"test": "hello"}'
)

print("\nRESPONSE OBJECT:")
print(response)

print("\nCONTENT:")
print(repr(response.content))

print("\nADDITIONAL KWARGS:")
print(repr(response.additional_kwargs))

print("\nMETADATA:")
print(response.response_metadata)

print("\n" + "=" * 60)