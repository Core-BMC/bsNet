"""Index healthy control (HC) adults from 7 OpenNeuro datasets.

Queries OpenNeuro GraphQL API to discover participants, then filters
for healthy controls aged ≥18 with resting-state fMRI + T1w.

Datasets (from MoBSE HC collection):
    ds000030  UCLA Consortium for Neuropsychiatric Phenomics
    ds000243  Washington University 120
    ds000258  Multi-echo Cambridge
    ds001386
    ds001747
    ds001771
    ds001796  Bilingualism and the brain

Output: data/hc_adult_index.csv

Usage:
    python src/scripts/index_openneuro_hc.py
    python src/scripts/index_openneuro_hc.py --cache-dir data/openneuro
"""

import argparse
import csv
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

GRAPHQL_URL = "https://openneuro.org/crn/graphql"

DATASET_IDS = [
    "ds000030",
    "ds000243",
    "ds000258",
    "ds001386",
    "ds001747",
    "ds001771",
    "ds001796",
]

# Diagnosis/group terms considered as "healthy control"
# Includes bilingualism groups (MC/EB/LB) — all healthy adults
HC_TERMS = {
    "control", "healthy", "healthy control", "hc", "normal",
    "ctrl", "nor", "healthy_control", "con", "nc",
    # Bilingual study groups (ds001747): all healthy
    "mc", "eb", "lb",
}

# Datasets known to contain only healthy adults (no diagnosis/group column needed)
# When these datasets lack a diagnosis column, all subjects are assumed HC.
ALL_HC_DATASETS = {"ds000243", "ds000258", "ds001771", "ds001796"}


def _graphql_query(query: str, retries: int = 3) -> dict:
    """Send GraphQL query to OpenNeuro API with retry."""
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            logger.warning(f"  GraphQL attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(2 ** attempt)
    return {}


def _fetch_file_url(dataset_id: str, filepath: str) -> str | None:
    """Get download URL for a specific file in a dataset."""
    query = f"""
    {{
      dataset(id: "{dataset_id}") {{
        draft {{
          files(prefix: "{filepath}") {{
            filename
            urls
          }}
        }}
      }}
    }}
    """
    result = _graphql_query(query)
    try:
        files = result["data"]["dataset"]["draft"]["files"]
        for f in files:
            if f["filename"] == filepath:
                urls = f.get("urls", [])
                return urls[0] if urls else None
    except (KeyError, TypeError, IndexError):
        pass
    return None


def _download_text(url: str) -> str | None:
    """Download text content from URL."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        logger.warning(f"  Download failed: {e}")
        return None


def _fetch_participants_tsv(dataset_id: str, cache_dir: Path) -> str | None:
    """Get participants.tsv content, checking local cache first."""
    # Check local cache (multiple possible locations)
    for subpath in [
        cache_dir / dataset_id / "participants.tsv",
        cache_dir / dataset_id / "1.0.0" / "uncompressed" / "participants.tsv",
    ]:
        if subpath.exists():
            logger.info(f"  Found local: {subpath}")
            return subpath.read_text()

    # Try GraphQL → download URL
    logger.info("  Querying GraphQL for participants.tsv ...")
    url = _fetch_file_url(dataset_id, "participants.tsv")
    if url:
        content = _download_text(url)
        if content:
            # Cache locally
            local_path = cache_dir / dataset_id / "participants.tsv"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(content)
            return content

    # Fallback: try S3 direct URL pattern
    s3_url = f"https://s3.amazonaws.com/openneuro.org/{dataset_id}/participants.tsv"
    logger.info(f"  Trying S3 fallback: {s3_url}")
    content = _download_text(s3_url)
    if content:
        local_path = cache_dir / dataset_id / "participants.tsv"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content)
        return content

    return None


def _list_subjects_s3(dataset_id: str) -> list[str]:
    """List subject directories via S3 XML listing (fallback for GraphQL)."""
    import xml.etree.ElementTree as ET

    s3_url = f"https://s3.amazonaws.com/openneuro.org?prefix={dataset_id}/&delimiter=/"
    logger.info(f"  Trying S3 listing: {s3_url}")
    try:
        req = urllib.request.Request(s3_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_text = resp.read().decode("utf-8")
        root = ET.fromstring(xml_text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        subjects = set()
        for prefix_elem in root.findall(".//s3:CommonPrefixes/s3:Prefix", ns):
            text = prefix_elem.text or ""
            # Format: "ds000258/sub-01/" → extract "sub-01"
            parts = text.strip("/").split("/")
            if len(parts) >= 2 and parts[-1].startswith("sub-"):
                subjects.add(parts[-1])
        return sorted(subjects)
    except Exception as e:
        logger.warning(f"  S3 listing failed: {e}")
        return []


def _list_subjects_graphql(dataset_id: str) -> list[str]:
    """List subject directories via GraphQL when participants.tsv is absent."""
    query = f"""
    {{
      dataset(id: "{dataset_id}") {{
        draft {{
          files(prefix: "") {{
            filename
          }}
        }}
      }}
    }}
    """
    result = _graphql_query(query)
    subjects = set()
    try:
        files = result["data"]["dataset"]["draft"]["files"]
        for f in files:
            name = f["filename"]
            if name.startswith("sub-") and "/" in name:
                sub_id = name.split("/")[0]
                subjects.add(sub_id)
    except (KeyError, TypeError):
        pass

    if not subjects:
        # Fallback to S3 listing
        return _list_subjects_s3(dataset_id)

    return sorted(subjects)


def _has_rest_and_t1w(dataset_id: str, sub_id: str) -> dict:
    """Check if subject has rest fMRI and T1w via GraphQL file listing."""
    query = f"""
    {{
      dataset(id: "{dataset_id}") {{
        draft {{
          files(prefix: "{sub_id}/") {{
            filename
          }}
        }}
      }}
    }}
    """
    result = _graphql_query(query)
    has_rest = False
    has_t1w = False
    try:
        files = result["data"]["dataset"]["draft"]["files"]
        for f in files:
            fn = f["filename"].lower()
            if "func" in fn and ("rest" in fn or "restingstate" in fn) and fn.endswith((".nii.gz", ".nii")):
                has_rest = True
            if "anat" in fn and "t1w" in fn and fn.endswith((".nii.gz", ".nii")):
                has_t1w = True
    except (KeyError, TypeError):
        pass
    return {"has_rest": has_rest, "has_t1w": has_t1w}


def _is_hc(diagnosis_val: str | None) -> bool:
    """Check if diagnosis value indicates healthy control."""
    if diagnosis_val is None or str(diagnosis_val).strip() in ("", "n/a", "nan"):
        return False
    return str(diagnosis_val).strip().lower() in HC_TERMS


def _parse_age(age_val: str | None) -> float | None:
    """Parse age from various formats (int, float, range '20-25')."""
    if age_val is None or str(age_val).strip() in ("", "n/a", "nan"):
        return None
    s = str(age_val).strip()
    # Handle range like "20-25"
    if "-" in s and not s.startswith("-"):
        parts = s.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except (ValueError, IndexError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def index_dataset(
    dataset_id: str,
    cache_dir: Path,
) -> list[dict]:
    """Index HC adults from a single dataset.

    Args:
        dataset_id: OpenNeuro dataset ID (e.g. 'ds000030').
        cache_dir: Local cache directory for downloaded metadata.

    Returns:
        List of dicts with keys: dataset_id, participant_id, age, gender,
        has_rest, has_t1w, diagnosis.
    """
    logger.info(f"[{dataset_id}] Indexing ...")
    records = []

    tsv_content = _fetch_participants_tsv(dataset_id, cache_dir)

    if tsv_content:
        lines = tsv_content.strip().split("\n")
        reader = csv.DictReader(lines, delimiter="\t")
        headers = reader.fieldnames or []
        logger.info(f"  participants.tsv columns: {headers}")

        # Identify diagnosis column (various naming conventions)
        diag_col = None
        for col in ["diagnosis", "group", "Group", "dx", "Diagnosis", "condition", "participant_group"]:
            if col in headers:
                diag_col = col
                break

        # Identify age column
        age_col = None
        for col in ["age", "Age", "AGE"]:
            if col in headers:
                age_col = col
                break

        # Identify gender column
        gender_col = None
        for col in ["gender", "sex", "Gender", "Sex"]:
            if col in headers:
                gender_col = col
                break

        # Identify rest/T1w availability columns (ds000030 style)
        rest_col = None
        for col in ["rest", "resting", "task-rest"]:
            if col in headers:
                rest_col = col
                break
        t1w_col = None
        for col in ["T1w", "t1w", "anat"]:
            if col in headers:
                t1w_col = col
                break

        n_total = 0
        n_hc = 0
        n_adult = 0
        for row in reader:
            n_total += 1
            sub_id = row.get("participant_id", "").strip()
            if not sub_id:
                continue

            diag_val = row.get(diag_col) if diag_col else None
            age_val = _parse_age(row.get(age_col) if age_col else None)
            gender_val = row.get(gender_col, "").strip() if gender_col else ""

            # No diagnosis column → assume all are HC (single-group datasets)
            is_healthy = True if diag_col is None else _is_hc(diag_val)

            if not is_healthy:
                continue
            n_hc += 1

            # Age filter: ≥18 (if age available)
            if age_val is not None and age_val < 18:
                continue
            n_adult += 1

            # Check rest + T1w availability
            has_rest = True  # default assume yes
            has_t1w = True
            _neg = ("0", "no", "No", "n/a")
            _pos = ("1", "yes", "Yes", "TRUE", "")
            if rest_col and row.get(rest_col, "").strip() not in _pos and row.get(rest_col, "").strip() in _neg:
                has_rest = False
            if t1w_col and row.get(t1w_col, "").strip() not in _pos and row.get(t1w_col, "").strip() in _neg:
                has_t1w = False

            if has_rest and has_t1w:
                records.append({
                    "dataset_id": dataset_id,
                    "participant_id": sub_id,
                    "age": age_val,
                    "gender": gender_val,
                    "has_rest": has_rest,
                    "has_t1w": has_t1w,
                    "diagnosis": str(diag_val).strip() if diag_val else "assumed_hc",
                })

        logger.info(
            f"  Total={n_total}, HC={n_hc}, Adult(≥18)={n_adult}, "
            f"with rest+T1w={len(records)}"
        )

    else:
        # No participants.tsv — index via file listing
        logger.info("  No participants.tsv found. Indexing via file listing ...")
        subjects = _list_subjects_graphql(dataset_id)
        logger.info(f"  Found {len(subjects)} subject directories")

        if dataset_id in ALL_HC_DATASETS:
            # Known all-HC dataset: assume all subjects have rest + T1w
            logger.info(f"  {dataset_id} is in ALL_HC_DATASETS → assuming all HC with rest+T1w")
            for sub_id in subjects:
                records.append({
                    "dataset_id": dataset_id,
                    "participant_id": sub_id,
                    "age": None,
                    "gender": "",
                    "has_rest": True,
                    "has_t1w": True,
                    "diagnosis": "assumed_hc",
                })
        else:
            for i, sub_id in enumerate(subjects):
                if (i + 1) % 20 == 0:
                    logger.info(f"  Checking files: {i + 1}/{len(subjects)} ...")
                modalities = _has_rest_and_t1w(dataset_id, sub_id)
                records.append({
                    "dataset_id": dataset_id,
                    "participant_id": sub_id,
                    "age": None,
                    "gender": "",
                    "has_rest": modalities["has_rest"],
                    "has_t1w": modalities["has_t1w"],
                    "diagnosis": "unknown_no_tsv",
                })

            # Filter to only those with rest + T1w
            before = len(records)
            records = [r for r in records if r["has_rest"] and r["has_t1w"]]
            logger.info(f"  With rest+T1w: {len(records)}/{before}")

    return records


def main() -> None:
    """Index all 7 datasets and save combined CSV."""
    parser = argparse.ArgumentParser(
        description="Index HC adults from OpenNeuro datasets"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/openneuro",
        help="Local cache directory for OpenNeuro data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/hc_adult_index.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for ds_id in DATASET_IDS:
        try:
            records = index_dataset(ds_id, cache_dir)
            all_records.extend(records)
        except Exception as e:
            logger.error(f"[{ds_id}] Failed: {e}")

    # Write CSV
    if all_records:
        fieldnames = [
            "dataset_id", "participant_id", "age", "gender",
            "has_rest", "has_t1w", "diagnosis",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

        logger.info(f"\nSaved {len(all_records)} HC adults to {output_path}")

        # Summary per dataset
        from collections import Counter
        ds_counts = Counter(r["dataset_id"] for r in all_records)
        logger.info("Per-dataset breakdown:")
        for ds_id in DATASET_IDS:
            logger.info(f"  {ds_id}: {ds_counts.get(ds_id, 0)} subjects")
    else:
        logger.warning("No records collected.")


if __name__ == "__main__":
    main()
