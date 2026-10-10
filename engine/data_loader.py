"""Sunwa-AI — Data Loader.
Handles: loading complaints CSV, landmarks CSV, interventions JSON, actions JSON.
Safely identifies schema, handles missing columns gracefully.
Never overwrites source files with derived data.
"""
import os
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from models.schemas import RawComplaint, InterventionRecord

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

REQUIRED_COMPLAINT_COLS = {"complaint_id", "region", "complaint_text", "submitted_at"}
OPTIONAL_COMPLAINT_COLS = {"location_description"}


def _coerce_datetime(value: str) -> str:
    """Normalize various datetime formats to ISO8601 string, preserving original on failure."""
    if not value or str(value).strip() in ("", "nan", "NaT", "None"):
        return datetime.utcnow().isoformat()
    v = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(v, fmt).isoformat()
        except ValueError:
            pass
    logger.debug("Could not parse datetime '%s', using as-is.", v)
    return v


def load_complaints(csv_path: Optional[Path] = None) -> List[RawComplaint]:
    """Load complaints CSV, yield validated RawComplaint objects.

    Handles: missing optional columns, malformed rows, empty file.
    Does NOT modify source CSV.
    """
    path = csv_path or (DATA_DIR / "complaints.csv")
    if not path.exists():
        logger.warning("Complaints file not found at %s — returning empty list.", path)
        return []

    try:
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig', encoding_errors='replace')
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-16', encoding_errors='replace')
    except Exception as exc:
        logger.error("Failed to read complaints CSV: %s", exc)
        return []

    missing = REQUIRED_COMPLAINT_COLS - set(df.columns)
    if missing:
        logger.error("Complaints CSV missing required columns: %s", missing)
        return []

    complaints: List[RawComplaint] = []
    seen_ids: set[str] = set()
    errors = 0

    for idx, row in df.iterrows():
        try:
            cid = str(row["complaint_id"]).strip()
            region = str(row["region"]).strip()
            text = str(row["complaint_text"]).strip()
            ts = _coerce_datetime(row.get("submitted_at", ""))
            loc_desc = str(row.get("location_description", "")).strip()

            if not cid or cid == "nan":
                logger.warning("Row %d: empty complaint_id — skipping.", idx + 2)
                errors += 1
                continue
            if not text or text == "nan":
                logger.warning("Row %d (id=%s): empty complaint_text — skipping.", idx + 2, cid)
                errors += 1
                continue
            if cid in seen_ids:
                logger.warning("Row %d: duplicate complaint_id '%s' — skipping.", idx + 2, cid)
                errors += 1
                continue

            seen_ids.add(cid)
            complaints.append(
                RawComplaint(
                    complaint_id=cid,
                    region=region,
                    complaint_text=text,
                    submitted_at=ts,
                    location_description=loc_desc if loc_desc not in ("", "nan") else None,
                )
            )
        except Exception as row_exc:
            logger.warning("Row %d: unexpected parse error: %s — skipping.", idx + 2, row_exc)
            errors += 1

    logger.info("Loaded %d complaints; %d skipped.", len(complaints), errors)
    return complaints


def load_landmarks(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load landmarks catalogue CSV. Returns empty DataFrame on failure."""
    path = csv_path or (DATA_DIR / "landmarks.csv")
    if not path.exists():
        logger.warning("Landmarks file not found at %s.", path)
        return pd.DataFrame()
    try:
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig', encoding_errors='replace')
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-16', encoding_errors='replace')
        required = {"landmark_id", "region", "landmark_name", "latitude", "longitude"}
        missing = required - set(df.columns)
        if missing:
            logger.error("Landmarks CSV missing columns: %s", missing)
            return pd.DataFrame()
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        logger.info("Loaded %d landmarks.", len(df))
        return df
    except Exception as exc:
        logger.error("Failed to load landmarks: %s", exc)
        return pd.DataFrame()


def load_interventions(json_path: Optional[Path] = None) -> List[InterventionRecord]:
    """Load interventions.json. Returns empty list on any error."""
    path = json_path or (DATA_DIR / "interventions.json")
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data: List[Dict[str, Any]] = json.load(fh)
        records = []
        for item in data:
            records.append(
                InterventionRecord(
                    intervention_id=item.get("intervention_id", ""),
                    target_cluster_id=item.get("target_cluster_id", ""),
                    cluster_description=item.get("cluster_description", ""),
                    region=item.get("region", ""),
                    action_type=item.get("action_type", ""),
                    department=item.get("department", ""),
                    created_timestamp=item.get("created_timestamp", ""),
                    completion_timestamp=item.get("completion_timestamp"),
                    status=item.get("status", "Unknown"),
                    notes=item.get("notes", ""),
                    is_simulated=item.get("is_simulated", True),
                )
            )
        logger.info("Loaded %d intervention records.", len(records))
        return records
    except Exception as exc:
        logger.error("Failed to load interventions: %s", exc)
        return []


def save_intervention_append(record: InterventionRecord, json_path: Optional[Path] = None) -> bool:
    """Append a new intervention record to interventions.json, persisting across restarts."""
    path = json_path or (DATA_DIR / "interventions.json")
    try:
        data = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                try:
                    data = json.load(fh)
                except Exception:
                    data = []
        data.append(record.to_dict())
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Saved intervention %s to JSON.", record.intervention_id)
        return True
    except Exception as exc:
        logger.error("Failed to save intervention: %s", exc)
        return False


def load_actions_json(json_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load actions.json. Returns parsed list or empty list on error."""
    path = json_path or (DATA_DIR / "actions.json")
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.error("Failed to load actions JSON: %s", exc)
        return []


def save_complaints_append(new_complaint: RawComplaint, csv_path: Optional[Path] = None) -> bool:
    """Append a single new complaint to the complaints CSV without overwriting existing data."""
    path = csv_path or (DATA_DIR / "complaints.csv")
    row = {
        "complaint_id": new_complaint.complaint_id,
        "region": new_complaint.region,
        "complaint_text": new_complaint.complaint_text,
        "submitted_at": new_complaint.submitted_at,
        "location_description": new_complaint.location_description or "",
    }
    try:
        if path.exists():
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            if new_complaint.complaint_id in df["complaint_id"].values:
                logger.warning("Complaint %s already exists — skipping append.", new_complaint.complaint_id)
                return False
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(path, index=False)
        logger.info("Appended complaint %s to CSV.", new_complaint.complaint_id)
        return True
    except Exception as exc:
        logger.error("Failed to append complaint: %s", exc)
        return False
