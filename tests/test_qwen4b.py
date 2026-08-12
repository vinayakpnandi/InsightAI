import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.llm import get_business_llm


print("=" * 60)
print("QWEN 4B DIRECT TEST")
print("=" * 60)

llm = get_business_llm()

response = llm.invoke(
    'Return ONLY this JSON: {"test": "hello"}'
)

print()
print("RESPONSE OBJECT:")
print(response)

print()
print("CONTENT:")
print(repr(response.content))

print()
print("ADDITIONAL KWARGS:")
print(repr(response.additional_kwargs))

print()
print("RESPONSE TYPE:")
print(type(response))

print()
print("=" * 60)