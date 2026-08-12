import json
import re
from typing import Any, Dict, Optional

from src.utils.llm import get_business_llm


# ============================================================
# DEFAULT RESULT
# ============================================================

def _default_result() -> Dict[str, Any]:
    return {
        "executive_summary": (
            "InsightAI could not generate business insights."
        ),
        "key_insights": [],
        "risks": [],
        "recommendations": [],
        "prediction_interpretation": "",
    }


# ============================================================
# SAFE CONVERSION
# ============================================================

def _safe(value: Any) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):

        return {
            str(k): _safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):

        return [
            _safe(v)
            for v in value
        ]

    if hasattr(
        value,
        "to_dict",
    ):

        try:
            return value.to_dict(
                orient="records"
            )
        except Exception:
            pass

    if hasattr(
        value,
        "tolist",
    ):

        try:
            return value.tolist()
        except Exception:
            pass

    return str(value)


# ============================================================
# COMPACT ML CONTEXT
# ============================================================

def _build_context(
    ml_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    if not isinstance(
        ml_result,
        dict,
    ):
        return {}

    context = {}

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    target = ml_result.get(
        "target_column",
        ml_result.get(
            "target",
            "Unknown",
        ),
    )

    problem_type = ml_result.get(
        "problem_type",
        "Unknown",
    )

    best_model = ml_result.get(
        "best_model",
        "Unknown",
    )

    best_score = ml_result.get(
        "best_score",
        ml_result.get(
            "best_model_score",
            None,
        ),
    )

    context["target"] = _safe(
        target
    )

    context["problem_type"] = _safe(
        problem_type
    )

    context["best_model"] = _safe(
        best_model
    )

    context["best_score"] = _safe(
        best_score
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    features = ml_result.get(
        "top_features",
        ml_result.get(
            "feature_importance",
            None,
        ),
    )

    if features is not None:

        if hasattr(
            features,
            "head",
        ):

            try:
                features = features.head(5)
            except Exception:
                pass

        features = _safe(
            features
        )

    context[
        "top_features"
    ] = features

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    leakage = ml_result.get(
        "leakage",
        {},
    )

    if isinstance(
        leakage,
        dict,
    ):

        context["leakage"] = {
            "risk_level": leakage.get(
                "risk_level",
                "unknown",
            ),
            "warning_count": leakage.get(
                "total_warnings",
                0,
            ),
        }

        # Only include concise leakage warnings.
        warnings = []

        for key in [
            "name_warnings",
            "correlation_warnings",
            "derived_warnings",
        ]:

            values = leakage.get(
                key,
                [],
            )

            if not isinstance(
                values,
                list,
            ):
                continue

            for warning in values[:3]:

                if isinstance(
                    warning,
                    dict,
                ):

                    reason = (
                        warning.get(
                            "reason"
                        )
                        or warning.get(
                            "column"
                        )
                        or str(warning)
                    )

                else:

                    reason = str(
                        warning
                    )

                warnings.append(
                    str(reason)
                )

        context[
            "leakage_warnings"
        ] = warnings[:3]

    return context


# ============================================================
# BUILD PROMPT
# ============================================================

def _build_prompt(
    context: Dict[str, Any],
    prediction: Any,
    prediction_inputs: Any,
) -> str:

    context_json = json.dumps(
        context,
        ensure_ascii=False,
        default=str,
    )

    prediction_json = json.dumps(
        _safe(prediction),
        ensure_ascii=False,
        default=str,
    )

    inputs_json = json.dumps(
        _safe(prediction_inputs),
        ensure_ascii=False,
        default=str,
    )

    return f"""
You are InsightAI.

Analyze this ML result and produce a concise business report.

ML:
{context_json}

Prediction:
{prediction_json}

Inputs:
{inputs_json}

Return ONLY valid JSON.

Use exactly:

{{
  "executive_summary": "short summary",
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "risks": ["risk 1", "risk 2"],
  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
  "prediction_interpretation": "short prediction explanation"
}}

Rules:
- Do not invent facts.
- Do not invent numbers.
- Mention leakage risks.
- Feature importance is not causation.
- If prediction is null, write "No prediction was generated."
- Keep every item short.
- Return JSON immediately.
""".strip()


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(
    text: str,
) -> Optional[Dict[str, Any]]:

    if not text:
        return None

    # Remove thinking blocks.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(
        r"</?think>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```json",
        "",
    )

    text = text.replace(
        "```",
        "",
    )

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        result = json.loads(
            text
        )

        if isinstance(
            result,
            dict,
        ):
            return result

    except Exception:
        pass

    # --------------------------------------------------------
    # Find JSON object
    # --------------------------------------------------------

    start = text.find(
        "{"
    )

    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(
        start,
        len(text),
    ):

        char = text[i]

        if escaped:

            escaped = False

            continue

        if (
            char == "\\"
            and in_string
        ):

            escaped = True

            continue

        if char == '"':

            in_string = not in_string

            continue

        if in_string:

            continue

        if char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                candidate = text[
                    start:i + 1
                ]

                try:

                    result = json.loads(
                        candidate
                    )

                    if isinstance(
                        result,
                        dict,
                    ):
                        return result

                except Exception:

                    return None

    return None


# ============================================================
# NORMALIZE
# ============================================================

def _normalize(
    data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    if not isinstance(
        data,
        dict,
    ):
        return _default_result()

    result = _default_result()

    summary = data.get(
        "executive_summary",
        "",
    )

    if summary:

        result[
            "executive_summary"
        ] = str(
            summary
        ).strip()

    for field in [
        "key_insights",
        "risks",
        "recommendations",
    ]:

        value = data.get(
            field,
            [],
        )

        if isinstance(
            value,
            str,
        ):

            value = [
                value
            ]

        if not isinstance(
            value,
            list,
        ):

            value = []

        result[field] = [
            str(item).strip()
            for item in value
            if item is not None
            and str(item).strip()
        ]

    prediction = data.get(
        "prediction_interpretation",
        "",
    )

    if prediction:

        result[
            "prediction_interpretation"
        ] = str(
            prediction
        ).strip()

    return result


# ============================================================
# GENERATE BUSINESS INSIGHTS
# ============================================================

def generate_business_insights(
    ml_result: Optional[Dict[str, Any]],
    prediction: Any = None,
    prediction_inputs: Any = None,
) -> Dict[str, Any]:

    print()
    print(
        "=" * 60
    )

    print(
        "[InsightAI] Generating business insights..."
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Compact context
    # --------------------------------------------------------

    context = _build_context(
        ml_result
    )

    print(
        "[InsightAI] Context:"
    )

    print(
        json.dumps(
            context,
            indent=2,
            default=str,
        )
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = _build_prompt(
        context,
        prediction,
        prediction_inputs,
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    try:

        llm = get_business_llm()

        response = llm.invoke(
            prompt
        )

        raw = getattr(
            response,
            "content",
            "",
        )

        raw = str(
            raw or ""
        )

    except Exception as error:

        print(
            "[InsightAI] LLM ERROR:"
        )

        print(
            repr(error)
        )

        result = _default_result()

        result[
            "_error"
        ] = str(error)

        return result

    # --------------------------------------------------------
    # Raw response
    # --------------------------------------------------------

    print()
    print(
        "[InsightAI] BUSINESS RAW RESPONSE:"
    )

    print(
        repr(raw)
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    parsed = _extract_json(
        raw
    )

    if parsed is not None:

        print(
            "[InsightAI] JSON parsed successfully."
        )

        return _normalize(
            parsed
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    print(
        "[InsightAI] JSON parsing failed."
    )

    result = _default_result()

    result[
        "_raw_response"
    ] = raw

    return result