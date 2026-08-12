import sys
from pathlib import Path


# ==================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# IMPORT DATABASE
# ==================================================

from src.database.database import execute_query


# ==================================================
# DATABASE CHECK
# ==================================================

print()
print("=" * 60)
print("INSIGHTAI DATABASE CHECK")
print("=" * 60)


# ==================================================
# CHECK COLUMN TYPES
# ==================================================

print()
print("1. Column types and values")
print("-" * 60)

try:

    result = execute_query(
        """
        SELECT
            typeof(unit_cost),
            unit_cost
        FROM uploaded_data
        LIMIT 10
        """
    )

    print(result)

except Exception as error:

    print(
        f"Error: {error}"
    )


# ==================================================
# CHECK STATISTICS
# ==================================================

print()
print("2. Unit_Cost statistics")
print("-" * 60)

try:

    result = execute_query(
        """
        SELECT
            COUNT(*),
            MIN(unit_cost),
            MAX(unit_cost),
            AVG(unit_cost)
        FROM uploaded_data
        """
    )

    print(result)

except Exception as error:

    print(
        f"Error: {error}"
    )


# ==================================================
# COMPLETE
# ==================================================

print()
print("=" * 60)
print("DATABASE CHECK COMPLETE")
print("=" * 60)