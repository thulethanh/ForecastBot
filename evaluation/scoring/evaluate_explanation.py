from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(
        "Thiếu package openai. Cài bằng: pip install -U openai"
    ) from exc


# ============================================================
# PROJECT / CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GT_ROOT = PROJECT_ROOT / "evaluation" / "ground_truth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RESULT_ROOT = PROJECT_ROOT / "evaluation" / "results" / "llm_judge"

# Có thể override trong .env:
# OPENAI_JUDGE_MODEL=gpt-5.6-terra
#
# Không đọc biến môi trường ở import-time vì .env chỉ được load trong main().
# parse_args() sẽ đọc OPENAI_JUDGE_MODEL sau khi load_dotenv().
FALLBACK_JUDGE_MODEL = "gpt-5.6-terra"

MAX_ATTEMPTS = 2
RETRY_SLEEP_S = 2.0


TARGET_MODELS = {
    "gemini": {
        "display_name": "Gemini 2.5 Pro",
        "M5": OUTPUT_ROOT / "m5_exp" / "gemini_2_5_pro" / "cases",
        "HCMC_BUS": (
            OUTPUT_ROOT
            / "hcmcbus_exp"
            / "gemini_2_5_pro"
            / "cases"
        ),
    },
    "gemma": {
        "display_name": "Gemma 4",
        "M5": OUTPUT_ROOT / "m5_exp" / "gemma4" / "cases",
        "HCMC_BUS": (
            OUTPUT_ROOT
            / "hcmcbus_exp"
            / "gemma4"
            / "cases"
        ),
    },
}


GT_CONFIG = {
    "M5": {
        "prefix": "m5",
        "case_prefix": "M",
        "gt_base": GT_ROOT / "m5",
    },
    "HCMC_BUS": {
        "prefix": "hcmcbus",
        "case_prefix": "H",
        "gt_base": GT_ROOT / "hcmcbus",
    },
}


# ============================================================
# RUBRIC
# ============================================================

JUDGE_INSTRUCTIONS = r"""
Bạn là LLM-as-a-Judge độc lập, đánh giá chất lượng phần giải thích kết quả
dự báo của ForecastBot.

Bạn KHÔNG đánh giá độ chính xác của bản thân dự báo bằng dữ liệu tương lai.
Bạn chỉ được sử dụng:
1. yêu cầu dự báo của người dùng;
2. tóm tắt dữ liệu lịch sử được phép sử dụng;
3. kết quả dự báo;
4. nội dung giải thích cần chấm.

Không suy diễn từ Ground Truth tương lai, test set, MAE hoặc RMSE.

Hãy chấm 4 tiêu chí, mỗi tiêu chí từ 1 đến 5:

1. relevance
- 5: giải thích trực tiếp, đầy đủ và đúng trọng tâm yêu cầu dự báo.
- 4: phù hợp, chỉ thiếu chi tiết nhỏ.
- 3: nhìn chung phù hợp nhưng còn chung chung hoặc thiếu một phần đáng kể.
- 2: chỉ liên quan một phần, có nhiều nội dung lệch trọng tâm.
- 1: không trả lời đúng nhu cầu người dùng.

2. faithfulness
- 5: mọi nhận định định lượng/xu hướng đều được hỗ trợ bởi history summary
  hoặc forecast; không bịa số liệu hay nguyên nhân.
- 4: nhìn chung trung thực, có diễn đạt hơi mạnh nhưng không làm sai bản chất.
- 3: có một số nhận định chưa được hỗ trợ rõ.
- 2: có nhiều nhận định không có căn cứ hoặc mâu thuẫn dữ liệu.
- 1: bịa đặt/mâu thuẫn nghiêm trọng với dữ liệu được cung cấp.

3. clarity
- 5: rõ ràng, mạch lạc, dễ hiểu, ngắn gọn phù hợp người dùng.
- 4: rõ ràng, chỉ có lỗi trình bày nhỏ.
- 3: hiểu được nhưng còn dài dòng hoặc cấu trúc chưa tốt.
- 2: khó theo dõi, diễn đạt mơ hồ.
- 1: rất khó hiểu hoặc không thể sử dụng.

4. usefulness
- 5: giúp người dùng hiểu được mức dự báo, xu hướng/biến động chính và ý nghĩa
  thực tế của kết quả mà không vượt quá bằng chứng.
- 4: hữu ích, nhưng còn thiếu một khía cạnh nhỏ.
- 3: có ích ở mức cơ bản.
- 2: ít giá trị thực tế.
- 1: hầu như không giúp người dùng hiểu kết quả.

QUY TẮC QUAN TRỌNG:
- Không trừ điểm chỉ vì forecast có thể sai so với tương lai; bạn không được
  nhìn thấy tương lai.
- Không yêu cầu Explanation phải giải thích nguyên nhân nhân quả nếu dữ liệu
  không cung cấp nguyên nhân.
- Nếu Explanation khẳng định nguyên nhân, sự kiện, mùa vụ hoặc số liệu mà
  history/forecast không hỗ trợ, phải giảm faithfulness.

QUY TẮC DIỄN GIẢI HISTORY / FORECAST WINDOW / TRAINING:
- "Historical summary" mô tả phạm vi dữ liệu lịch sử khả dụng của case
  ForecastBot. start_date/end_date trong historical summary có thể được
  Explanation nhắc lại như phạm vi dữ liệu lịch sử mà không bị xem là sai.
- "window_info" mô tả các cửa sổ lịch sử thực tế được cung cấp cho các nhánh
  dự báo. Ví dụ short_window=180 và long_window=365 nghĩa là hai nhánh dự báo
  sử dụng 180 và 365 quan sát gần nhất.
- Phải phân biệt rõ ba khái niệm:
  (1) phạm vi dữ liệu lịch sử khả dụng;
  (2) cửa sổ lịch sử đưa vào nhánh dự báo;
  (3) dữ liệu dùng để huấn luyện/fine-tune mô hình.
  Ba khái niệm này KHÔNG được tự động xem là tương đương.
- KHÔNG được diễn giải câu như "Dữ liệu lịch sử được sử dụng từ A đến B",
  "Dữ liệu lịch sử bao phủ từ A đến B", hoặc chart_note có ý nghĩa tương tự
  thành khẳng định rằng mọi quan sát trong khoảng A-B đều được đưa trực tiếp
  vào từng nhánh dự báo, trừ khi Explanation nói rõ như vậy.
- KHÔNG được tự đưa khái niệm "huấn luyện mô hình", "training", "fine-tuning"
  vào đánh giá nếu Explanation không trực tiếp khẳng định điều đó.
- Nếu chart_note chỉ mô tả khoảng thời gian lịch sử hiển thị/khả dụng và
  khoảng này khớp historical summary thì phải xem là faithful, kể cả khi
  Forecast Agent chỉ dùng các cửa sổ gần nhất trong window_info.
- Chỉ ghi một nội dung vào unsupported_claims khi chính Explanation thực sự
  đưa ra nhận định không được request, historical summary hoặc forecast hỗ trợ.
  Không được tạo unsupported claim từ suy diễn bổ sung của Judge.

QUY TẮC BẰNG CHỨNG CHO unsupported_claims:
- Mỗi unsupported_claim phải phản ánh một câu hoặc ý THỰC SỰ xuất hiện trong
  Explanation. Khi mô tả unsupported_claim, phải giữ nguyên nghĩa của câu gốc
  và KHÔNG được thêm khái niệm mới.
- Đặc biệt, nếu Explanation KHÔNG có các từ/ý như "huấn luyện", "training",
  "fine-tuning", "fit mô hình" hoặc tương đương thì Judge TUYỆT ĐỐI KHÔNG được
  tạo unsupported_claim nói rằng Explanation đã khẳng định dữ liệu được dùng
  để huấn luyện/fine-tune mô hình.
- Trước khi giảm faithfulness vì một unsupported_claim, Judge phải tự kiểm tra:
  (a) nhận định đó có thật sự xuất hiện trong Explanation hay không;
  (b) nhận định đó có mâu thuẫn hoặc không được hỗ trợ bởi request, historical
      summary hoặc forecast hay không.
  Nếu (a) không đúng thì KHÔNG được trừ điểm.
- Không được chuyển câu mô tả phạm vi dữ liệu, ví dụ
  "Dữ liệu lịch sử được sử dụng từ A đến B"
  thành câu mạnh hơn như
  "Toàn bộ dữ liệu A-B được dùng để huấn luyện mô hình".
  Hai câu này không tương đương.
- Nếu không thể chỉ ra bằng chứng trực tiếp trong Explanation cho một
  unsupported_claim thì phải loại claim đó khỏi unsupported_claims và không
  được giảm faithfulness vì claim đó.

QUY TẮC CHẤM ĐIỂM:
- Điểm phải là số nguyên 1, 2, 3, 4 hoặc 5.
- overall_score = trung bình cộng của 4 điểm, làm tròn 3 chữ số.
- unsupported_claims là danh sách ngắn các nhận định không được hỗ trợ;
  nếu không có thì trả [].
- reason_vi là giải thích ngắn bằng tiếng Việt, tối đa khoảng 120 từ.

Chỉ trả về MỘT JSON object hợp lệ, không markdown, đúng cấu trúc:
{
  "relevance": 1,
  "faithfulness": 1,
  "clarity": 1,
  "usefulness": 1,
  "overall_score": 1.0,
  "unsupported_claims": [],
  "reason_vi": ""
}
""".strip()


# ============================================================
# BASIC HELPERS
# ============================================================

CASE_RE = re.compile(r"^[MH]\d{2}$", re.I)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_json(path: Path) -> Any | None:
    try:
        if path.exists():
            return read_json(path)
    except Exception:
        pass
    return None


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def as_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    return json.dumps(
        obj,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def parse_json_object(text: str) -> dict:
    """
    Parse JSON trực tiếp; nếu model lỡ trả markdown fence thì vẫn recover.
    """
    text = (text or "").strip()

    if not text:
        raise ValueError("Empty judge response")

    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Judge response is not a JSON object")
        return obj
    except json.JSONDecodeError:
        pass

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.I | re.S,
    )
    if fenced:
        obj = json.loads(fenced.group(1))
        if isinstance(obj, dict):
            return obj

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        obj = json.loads(text[first:last + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("Cannot parse judge JSON")


def get_usage(response: Any) -> dict:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "input_tokens": np.nan,
            "output_tokens": np.nan,
            "total_tokens": np.nan,
        }

    def pick(*names):
        for name in names:
            value = getattr(usage, name, None)
            if value is not None:
                return value

        if isinstance(usage, dict):
            for name in names:
                if name in usage:
                    return usage[name]

        return np.nan

    return {
        "input_tokens": pick(
            "input_tokens",
            "prompt_tokens",
        ),
        "output_tokens": pick(
            "output_tokens",
            "completion_tokens",
        ),
        "total_tokens": pick(
            "total_tokens",
        ),
    }


# ============================================================
# GROUND-TRUTH CASE DISCOVERY
# ============================================================

def discover_gt_case_ids(dataset: str) -> list[str]:
    """
    Dùng forecast_requests của Ground Truth làm denominator chính.
    """
    cfg = GT_CONFIG[dataset]
    prefix = cfg["prefix"]
    gt_base = cfg["gt_base"]

    candidates = [
        gt_base / f"{prefix}_forecast_requests",
        gt_base / "forecast_requests",
    ]

    case_ids: set[str] = set()

    for folder in candidates:
        if not folder.exists():
            continue

        for path in folder.rglob("*.json"):
            match = re.search(
                r"([MH]\d{2})",
                path.stem,
                flags=re.I,
            )
            if match:
                case_ids.add(match.group(1).upper())

    # Fallback: lấy từ cases output nếu GT naming khác dự kiến.
    return sorted(case_ids)


# ============================================================
# INPUT EXTRACTION
# ============================================================

def extract_request(case_dir: Path) -> dict:
    obj = safe_json(case_dir / "request.json")
    return obj if isinstance(obj, dict) else {}


def extract_history_summary(case_dir: Path) -> dict:
    """
    Chỉ gửi summary, KHÔNG gửi toàn bộ history để:
    - giảm token;
    - giữ Judge tập trung;
    - không đưa test/actual tương lai.
    """
    obj = safe_json(case_dir / "history.json")

    if not isinstance(obj, dict):
        return {}

    summary = obj.get("summary")

    if isinstance(summary, dict):
        return summary

    history = obj.get("history")

    if not isinstance(history, list) or not history:
        return {}

    rows = []

    for item in history:
        if not isinstance(item, dict):
            continue

        ds = item.get("ds")
        y = item.get("y")

        try:
            y = float(y)
        except Exception:
            continue

        if np.isfinite(y):
            rows.append((ds, y))

    if not rows:
        return {}

    ys = np.asarray(
        [y for _, y in rows],
        dtype=float,
    )

    return {
        "n_points": len(rows),
        "start_date": rows[0][0],
        "end_date": rows[-1][0],
        "min_y": float(np.min(ys)),
        "max_y": float(np.max(ys)),
        "mean_y": float(np.mean(ys)),
        "median_y": float(np.median(ys)),
    }


def extract_forecast(case_dir: Path) -> Any:
    obj = safe_json(case_dir / "forecast.json")
    return obj if obj is not None else {}


def extract_explanation(case_dir: Path) -> Any:
    obj = safe_json(case_dir / "explanation.json")
    return obj if obj is not None else None


def build_judge_input(
    case_id: str,
    dataset: str,
    target_model_name: str,
    case_dir: Path,
) -> dict:
    return {
        "case_id": case_id,
        "dataset": dataset,
        "forecast_model": target_model_name,
        "forecast_request": extract_request(
            case_dir
        ),
        "history_summary": extract_history_summary(
            case_dir
        ),
        "forecast_result": extract_forecast(
            case_dir
        ),
        "explanation_to_evaluate": extract_explanation(
            case_dir
        ),
    }


# ============================================================
# JUDGE OUTPUT VALIDATION
# ============================================================

SCORE_FIELDS = [
    "relevance",
    "faithfulness",
    "clarity",
    "usefulness",
]


def validate_judge_result(obj: dict) -> dict:
    clean: dict[str, Any] = {}

    scores = []

    for field in SCORE_FIELDS:
        value = obj.get(field)

        try:
            score = int(value)
        except Exception as exc:
            raise ValueError(
                f"Invalid score for {field}: {value}"
            ) from exc

        if score < 1 or score > 5:
            raise ValueError(
                f"{field} out of range 1..5: {score}"
            )

        clean[field] = score
        scores.append(score)

    # Tự tính lại để tránh tin tuyệt đối con số model trả.
    clean["overall_score"] = round(
        float(np.mean(scores)),
        3,
    )

    claims = obj.get(
        "unsupported_claims",
        [],
    )

    if claims is None:
        claims = []

    if not isinstance(claims, list):
        claims = [str(claims)]

    clean["unsupported_claims"] = [
        str(x).strip()
        for x in claims
        if str(x).strip()
    ]

    clean["reason_vi"] = str(
        obj.get("reason_vi", "")
    ).strip()

    return clean


# ============================================================
# OPENAI JUDGE
# ============================================================

def call_openai_judge(
    client: OpenAI,
    judge_model: str,
    payload: dict,
) -> tuple[dict, dict]:

    user_input = (
        "Hãy đánh giá Explanation sau theo rubric đã cho.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        started = time.perf_counter()

        try:
            response = client.responses.create(
                model=judge_model,
                instructions=JUDGE_INSTRUCTIONS,
                input=user_input,
            )

            latency_s = (
                time.perf_counter()
                - started
            )

            text = getattr(
                response,
                "output_text",
                "",
            )

            parsed = parse_json_object(text)
            validated = validate_judge_result(
                parsed
            )

            usage = get_usage(response)

            meta = {
                "attempt": attempt,
                "latency_s": latency_s,
                "input_tokens": usage[
                    "input_tokens"
                ],
                "output_tokens": usage[
                    "output_tokens"
                ],
                "total_tokens": usage[
                    "total_tokens"
                ],
                "raw_response": text,
            }

            return validated, meta

        except Exception as exc:
            last_error = exc

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_S)

    raise RuntimeError(
        f"Judge failed after "
        f"{MAX_ATTEMPTS} attempts: "
        f"{last_error}"
    )


# ============================================================
# PER-MODEL / PER-DATASET EVALUATION
# ============================================================

def evaluate_dataset(
    *,
    client: OpenAI,
    target_model_key: str,
    dataset: str,
    judge_model: str,
    resume: bool,
    case_id: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:

    target_cfg = TARGET_MODELS[
        target_model_key
    ]

    target_model_name = target_cfg[
        "display_name"
    ]

    cases_dir = target_cfg[dataset]

    result_dir = (
        RESULT_ROOT
        / target_model_key
    )

    raw_dir = (
        result_dir
        / "judge_raw"
        / dataset.lower()
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_case_ids = discover_gt_case_ids(
        dataset
    )

    # Nếu không discover được GT do naming khác,
    # fallback sang case directories.
    if not gt_case_ids and cases_dir.exists():
        gt_case_ids = sorted(
            p.name.upper()
            for p in cases_dir.iterdir()
            if (
                p.is_dir()
                and CASE_RE.fullmatch(
                    p.name.upper()
                )
            )
        )

    # --------------------------------------------------------
    # SINGLE CASE / PILOT FILTER
    # --------------------------------------------------------
    # --case-id: chấm đúng 1 case, ví dụ M01 hoặc H01.
    # --limit: chấm N case đầu tiên để pilot.
    if case_id is not None:
        requested = case_id.strip().upper()

        if not CASE_RE.fullmatch(requested):
            raise ValueError(
                f"--case-id không hợp lệ: {case_id}. "
                "Định dạng đúng: M01 hoặc H01."
            )

        expected_prefix = GT_CONFIG[dataset]["case_prefix"]

        if not requested.startswith(expected_prefix):
            raise ValueError(
                f"Case {requested} không thuộc dataset {dataset}. "
                f"{dataset} yêu cầu prefix {expected_prefix}."
            )

        if requested not in gt_case_ids:
            raise ValueError(
                f"Không tìm thấy case {requested} trong "
                f"Ground Truth/output của {dataset}."
            )

        gt_case_ids = [requested]

    elif limit is not None:
        if limit <= 0:
            raise ValueError("--limit phải là số nguyên > 0.")

        gt_case_ids = gt_case_ids[:limit]

    rows = []

    print()
    print("=" * 72)
    print(
        f"{dataset} / "
        f"{target_model_name} / "
        f"Judge={judge_model}"
    )
    print("=" * 72)
    print(
        f"Expected cases       : "
        f"{len(gt_case_ids)}"
    )

    for idx, case_id in enumerate(
        gt_case_ids,
        start=1,
    ):
        case_dir = cases_dir / case_id

        explanation_path = (
            case_dir
            / "explanation.json"
        )

        raw_path = raw_dir / f"{case_id}.json"

        base_row = {
            "case_id": case_id,
            "dataset": dataset,
            "target_model": target_model_name,
            "target_model_key": target_model_key,
            "judge_model": judge_model,
            "case_dir_exists": case_dir.exists(),
            "explanation_available": (
                explanation_path.exists()
            ),
        }

        print(
            f"[{idx:02d}/{len(gt_case_ids):02d}] "
            f"{case_id}",
            end=" ",
            flush=True,
        )

        if (
            not case_dir.exists()
            or not explanation_path.exists()
        ):
            row = {
                **base_row,
                "judge_status": (
                    "missing_explanation"
                ),
                "relevance": np.nan,
                "faithfulness": np.nan,
                "clarity": np.nan,
                "usefulness": np.nan,
                "overall_score": np.nan,
                "unsupported_claims": "[]",
                "reason_vi": "",
                "judge_attempt": np.nan,
                "judge_latency_s": np.nan,
                "input_tokens": np.nan,
                "output_tokens": np.nan,
                "total_tokens": np.nan,
                "error_message": "",
            }

            rows.append(row)
            print("SKIP - missing explanation")
            continue

        # Resume từ raw JSON đã chấm thành công.
        if resume and raw_path.exists():
            cached = safe_json(raw_path)

            if (
                isinstance(cached, dict)
                and cached.get(
                    "judge_status"
                ) == "success"
            ):
                result = cached.get(
                    "scores",
                    {},
                )
                meta = cached.get(
                    "meta",
                    {},
                )

                row = {
                    **base_row,
                    "judge_status": "success",
                    **{
                        field: result.get(
                            field,
                            np.nan,
                        )
                        for field
                        in SCORE_FIELDS
                    },
                    "overall_score": (
                        result.get(
                            "overall_score",
                            np.nan,
                        )
                    ),
                    "unsupported_claims": json.dumps(
                        result.get(
                            "unsupported_claims",
                            [],
                        ),
                        ensure_ascii=False,
                    ),
                    "reason_vi": result.get(
                        "reason_vi",
                        "",
                    ),
                    "judge_attempt": meta.get(
                        "attempt",
                        np.nan,
                    ),
                    "judge_latency_s": meta.get(
                        "latency_s",
                        np.nan,
                    ),
                    "input_tokens": meta.get(
                        "input_tokens",
                        np.nan,
                    ),
                    "output_tokens": meta.get(
                        "output_tokens",
                        np.nan,
                    ),
                    "total_tokens": meta.get(
                        "total_tokens",
                        np.nan,
                    ),
                    "error_message": "",
                }

                rows.append(row)
                print(
                    "CACHED "
                    f"{row['overall_score']:.3f}"
                )
                continue

        payload = build_judge_input(
            case_id,
            dataset,
            target_model_name,
            case_dir,
        )

        try:
            result, meta = call_openai_judge(
                client=client,
                judge_model=judge_model,
                payload=payload,
            )

            raw_record = {
                "case_id": case_id,
                "dataset": dataset,
                "target_model": target_model_name,
                "judge_model": judge_model,
                "judge_status": "success",
                "scores": result,
                "meta": meta,
            }

            dump_json(
                raw_path,
                raw_record,
            )

            row = {
                **base_row,
                "judge_status": "success",
                **{
                    field: result[field]
                    for field
                    in SCORE_FIELDS
                },
                "overall_score": result[
                    "overall_score"
                ],
                "unsupported_claims": json.dumps(
                    result[
                        "unsupported_claims"
                    ],
                    ensure_ascii=False,
                ),
                "reason_vi": result[
                    "reason_vi"
                ],
                "judge_attempt": meta[
                    "attempt"
                ],
                "judge_latency_s": meta[
                    "latency_s"
                ],
                "input_tokens": meta[
                    "input_tokens"
                ],
                "output_tokens": meta[
                    "output_tokens"
                ],
                "total_tokens": meta[
                    "total_tokens"
                ],
                "error_message": "",
            }

            rows.append(row)

            print(
                "OK "
                f"{result['overall_score']:.3f}"
            )

        except Exception as exc:
            error_text = str(exc)

            raw_record = {
                "case_id": case_id,
                "dataset": dataset,
                "target_model": target_model_name,
                "judge_model": judge_model,
                "judge_status": "judge_error",
                "error_message": error_text,
            }

            dump_json(
                raw_path,
                raw_record,
            )

            rows.append({
                **base_row,
                "judge_status": "judge_error",
                "relevance": np.nan,
                "faithfulness": np.nan,
                "clarity": np.nan,
                "usefulness": np.nan,
                "overall_score": np.nan,
                "unsupported_claims": "[]",
                "reason_vi": "",
                "judge_attempt": np.nan,
                "judge_latency_s": np.nan,
                "input_tokens": np.nan,
                "output_tokens": np.nan,
                "total_tokens": np.nan,
                "error_message": error_text,
            })

            print(
                f"ERROR - {error_text}"
            )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    per_case_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_cols = [
        "dataset",
        "target_model",
        "target_model_key",
        "judge_model",
    ]

    for keys, group in per_case_df.groupby(
        group_cols,
        dropna=False,
    ):
        (
            dataset,
            target_model,
            target_model_key,
            judge_model,
        ) = keys

        available = group[
            group[
                "explanation_available"
            ] == True
        ]

        judged = group[
            group[
                "judge_status"
            ] == "success"
        ]

        n_cases = len(group)
        n_available = len(available)
        n_judged = len(judged)

        rows.append({
            "dataset": dataset,
            "target_model": target_model,
            "target_model_key": (
                target_model_key
            ),
            "judge_model": judge_model,

            "n_cases": n_cases,

            "n_explanation_available": (
                n_available
            ),

            "explanation_availability_rate": (
                n_available / n_cases
                if n_cases
                else np.nan
            ),

            "n_judged_success": n_judged,

            "judge_success_rate_among_available": (
                n_judged / n_available
                if n_available
                else np.nan
            ),

            "mean_relevance": (
                judged[
                    "relevance"
                ].mean()
                if n_judged
                else np.nan
            ),

            "mean_faithfulness": (
                judged[
                    "faithfulness"
                ].mean()
                if n_judged
                else np.nan
            ),

            "mean_clarity": (
                judged[
                    "clarity"
                ].mean()
                if n_judged
                else np.nan
            ),

            "mean_usefulness": (
                judged[
                    "usefulness"
                ].mean()
                if n_judged
                else np.nan
            ),

            "mean_overall_score": (
                judged[
                    "overall_score"
                ].mean()
                if n_judged
                else np.nan
            ),

            "median_overall_score": (
                judged[
                    "overall_score"
                ].median()
                if n_judged
                else np.nan
            ),

            "min_overall_score": (
                judged[
                    "overall_score"
                ].min()
                if n_judged
                else np.nan
            ),

            "max_overall_score": (
                judged[
                    "overall_score"
                ].max()
                if n_judged
                else np.nan
            ),

            "n_cases_with_unsupported_claims": (
                judged[
                    "unsupported_claims"
                ].apply(
                    lambda x:
                    x not in (
                        "[]",
                        "",
                        np.nan,
                    )
                ).sum()
                if n_judged
                else 0
            ),

            "total_judge_tokens": (
                pd.to_numeric(
                    judged[
                        "total_tokens"
                    ],
                    errors="coerce",
                ).sum(
                    min_count=1
                )
                if n_judged
                else np.nan
            ),

            "mean_judge_latency_s": (
                pd.to_numeric(
                    judged[
                        "judge_latency_s"
                    ],
                    errors="coerce",
                ).mean()
                if n_judged
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def build_score_distribution(
    per_case_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Phân bố overall_score theo các mức để tiện báo cáo.
    Không biến score thành tiêu chí E2E success.
    """
    judged = per_case_df[
        per_case_df[
            "judge_status"
        ] == "success"
    ].copy()

    if judged.empty:
        return pd.DataFrame()

    bins = [
        0,
        2,
        3,
        4,
        4.5,
        5.000001,
    ]

    labels = [
        "(1,2]",
        "(2,3]",
        "(3,4]",
        "(4,4.5]",
        "(4.5,5]",
    ]

    judged["score_band"] = pd.cut(
        judged["overall_score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    result = (
        judged.groupby(
            [
                "dataset",
                "target_model",
                "score_band",
            ],
            observed=False,
        )
        .size()
        .reset_index(
            name="n_cases"
        )
    )

    totals = (
        judged.groupby(
            [
                "dataset",
                "target_model",
            ]
        )
        .size()
        .rename("n_total_judged")
        .reset_index()
    )

    result = result.merge(
        totals,
        on=[
            "dataset",
            "target_model",
        ],
        how="left",
    )

    result["rate"] = (
        result["n_cases"]
        / result["n_total_judged"]
    )

    return result


# ============================================================
# CLI
# ============================================================

def configure_architecture(architecture: str) -> None:
    """Keep source artifacts and judge cache isolated by architecture."""
    global RESULT_ROOT
    if architecture not in {"multi_agent", "single_agent"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    single = architecture == "single_agent"
    source_root = OUTPUT_ROOT / "single_agent_ablation" if single else OUTPUT_ROOT
    result_base = PROJECT_ROOT / "evaluation" / "results"
    RESULT_ROOT = (
        result_base / "single_agent_ablation" / "llm_judge"
        if single else result_base / "llm_judge"
    )
    for model, slug in (("gemini", "gemini_2_5_pro"), ("gemma", "gemma4")):
        for dataset, folder in (("M5", "m5_exp"), ("HCMC_BUS", "hcmcbus_exp")):
            TARGET_MODELS[model][dataset] = source_root / folder / slug / "cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ForecastBot explanations "
            "with OpenAI LLM-as-a-Judge."
        )
    )

    parser.add_argument(
        "--target-model",
        choices=[
            "gemini",
            "gemma",
            "all",
        ],
        default="gemini",
        help=(
            "Model outputs cần chấm. "
            "Default: gemini"
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "M5",
            "HCMC_BUS",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--judge-model",
        default=os.getenv(
            "OPENAI_JUDGE_MODEL",
            FALLBACK_JUDGE_MODEL,
        ),
        help=(
            "OpenAI model dùng làm Judge. "
            "Default lấy OPENAI_JUDGE_MODEL trong .env "
            f"hoặc {FALLBACK_JUDGE_MODEL}."
        ),
    )

    selection = parser.add_mutually_exclusive_group()

    selection.add_argument(
        "--case-id",
        default=None,
        help=(
            "Chỉ đánh giá đúng 1 case, ví dụ M01 hoặc H01. "
            "Dùng để test trước khi chạy pilot/full."
        ),
    )

    selection.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Chỉ đánh giá N case đầu tiên của mỗi dataset được chọn. "
            "Ví dụ --limit 10 để chạy pilot."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Bỏ qua các case đã có "
            "judge_raw success."
        ),
    )

    parser.add_argument(
        "--architecture", choices=["multi_agent", "single_agent"],
        default="multi_agent", help="Artifact set and isolated judge cache to use.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise SystemExit(
            "Không tìm thấy OPENAI_API_KEY "
            "trong environment hoặc .env."
        )

    args = parse_args()
    configure_architecture(args.architecture)
    print(f"Architecture: {args.architecture}")

    # Nếu truyền --case-id mà --dataset=all, tự suy ra dataset theo prefix.
    # Mxx -> M5, Hxx -> HCMC_BUS.
    if args.case_id:
        args.case_id = args.case_id.strip().upper()

        if not CASE_RE.fullmatch(args.case_id):
            raise SystemExit(
                "--case-id không hợp lệ. "
                "Dùng định dạng M01 hoặc H01."
            )

        inferred_dataset = (
            "M5"
            if args.case_id.startswith("M")
            else "HCMC_BUS"
        )

        if args.dataset == "all":
            args.dataset = inferred_dataset

        elif args.dataset != inferred_dataset:
            raise SystemExit(
                f"{args.case_id} thuộc {inferred_dataset}, "
                f"không thuộc {args.dataset}."
            )

    client = OpenAI(
        api_key=api_key
    )

    if args.target_model == "all":
        target_models = [
            "gemini",
            "gemma",
        ]
    else:
        target_models = [
            args.target_model
        ]

    if args.dataset == "all":
        datasets = [
            "M5",
            "HCMC_BUS",
        ]
    else:
        datasets = [
            args.dataset
        ]

    all_frames = []

    for target_model_key in target_models:
        for dataset in datasets:
            df = evaluate_dataset(
                client=client,
                target_model_key=(
                    target_model_key
                ),
                dataset=dataset,
                judge_model=(
                    args.judge_model
                ),
                resume=args.resume,
                case_id=args.case_id,
                limit=args.limit,
            )

            all_frames.append(df)

    if not all_frames:
        raise SystemExit(
            "Không có kết quả để lưu."
        )

    per_case_df = pd.concat(
        all_frames,
        ignore_index=True,
    )

    summary_df = build_summary(
        per_case_df
    )

    distribution_df = (
        build_score_distribution(
            per_case_df
        )
    )

    # Lưu CSV tổng hợp tách riêng theo target model để tránh
    # Gemini và Gemma ghi đè kết quả của nhau.
    #
    # Ví dụ:
    #   evaluation/results/llm_judge/gemini/
    #   evaluation/results/llm_judge/gemma/
    #
    # Nếu người dùng chủ động đánh giá nhiều model trong cùng một lần chạy
    # (--target-model all), lưu bộ tổng hợp vào thư mục comparison/.
    if len(target_models) == 1:
        result_dir = RESULT_ROOT / target_models[0]
    else:
        result_dir = RESULT_ROOT / "comparison"

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Không để single-case/pilot ghi đè file kết quả full.
    if args.case_id:
        scope_suffix = f"_{args.case_id}"
    elif args.limit is not None:
        scope_suffix = f"_pilot_{args.limit}"
    else:
        scope_suffix = ""

    per_case_path = (
        result_dir
        / f"explanation_per_case{scope_suffix}.csv"
    )

    summary_path = (
        result_dir
        / f"explanation_summary{scope_suffix}.csv"
    )

    distribution_path = (
        result_dir
        / f"explanation_score_distribution{scope_suffix}.csv"
    )

    per_case_df.to_csv(
        per_case_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    distribution_df.to_csv(
        distribution_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("LLM-AS-A-JUDGE SUMMARY")
    print("=" * 72)

    if summary_df.empty:
        print("No summary.")
    else:
        print(
            summary_df.to_string(
                index=False
            )
        )

    print()
    print("Results saved to:")
    print(result_dir)
    print()
    print("Files:")
    print(f"- {per_case_path.name}")
    print(f"- {summary_path.name}")
    print(
        f"- {distribution_path.name}"
    )


if __name__ == "__main__":
    main()
