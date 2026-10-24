"""Authenticated, read-only Kaggle submission inventory and CSV integrity check.

Run through GitHub Actions with the existing KAGGLE secret. Never downloads
training labels, shares trained weights, or submits to competition.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

COMPETITION = "filament-segmentation-2026"
OWNER = "minahussain111"
V5 = f"{OWNER}/solar-filament-research-v5/v3"
REPORT = Path("reports/KAGGLE_READINESS.json")
RLE_SIZE = (2048, 2048)
STEM_RE = re.compile(r"^\d{14}[A-Za-z]h_\d+$")


def _str(value):
    if value is None:
        return None
    return str(value)


def csv_integrity(path):
    """Validate the *actual* original file, including every COCO RLE.

    Decoding and mask.area of an untrusted RLE can fail; report failure rather
    than silently accepting a well-formed CSV with bogus predictions.
    """
    from pycocotools import mask as coco
    totals = set()
    stems = set()
    seen = set()
    areas = []
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for piece in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(piece)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Competition CSV header differs from exactly two expected fields")
        rows = 0
        for row in reader:
            if rows >= 100000:
                raise ValueError("Unreasonably large submission")
            identifier = row.get("filament_id") or ""
            rle = row.get("segmentation_rle") or ""
            if not STEM_RE.fullmatch(identifier) or identifier in seen:
                raise ValueError("Invalid or repeated filament ID")
            seen.add(identifier)
            stem, _ = identifier.rsplit("_", 1)
            if not rle or len(rle) > 7000000 or "\x00" in rle:
                raise ValueError("Invalid COCO RLE counts")
            try:
                area = int(coco.area({"size": list(RLE_SIZE), "counts": rle.encode("ascii")}))
            except Exception as err:
                raise ValueError("Unreadable COCO RLE, refusing to call it a valid model") from err
            if not 0 < area <= RLE_SIZE[0] * RLE_SIZE[1]:
                raise ValueError("Empty/oversized mask")
            stems.add(stem)
            areas.append(area)
            rows += 1
    if not rows or len(stems) > 180:
        raise ValueError("No valid test predictions or more than 180 unique frames")
    return {"sha256": digest.hexdigest(), "rows": rows, "unique_ids": len(seen),
            "unique_stems": len(stems), "min_area": min(areas), "max_area": max(areas),
            "rle_all_rows_area_checked": True,
            "input_inventory_matched": False,
            "note": "Every encoded instance was checked; test file inventory and official score are separate"}


def _status(api, reference):
    try:
        response = api.kernels_status(reference)
        return str(response.status).split(".")[-1].upper()
    except Exception as err:
        return "UNAVAILABLE_" + type(err).__name__


def _submissions(api):
    records = []
    try:
        for page in range(1, 4):
            values = api.competition_submissions(COMPETITION, page_number=page, page_size=100) or []
            for row in values:
                if row is None:
                    continue
                # Scoring can be absent for a pending/failed submission.
                score = getattr(row, "public_score", None)
                if score is None:
                    score = getattr(row, "publicScore", None)
                if score is None:
                    score = getattr(row, "score", None)
                try:
                    score = float(score) if score is not None else None
                    if score is not None and not math.isfinite(score):
                        score = None
                except (TypeError, ValueError):
                    score = None
                records.append({"ref": _str(getattr(row, "ref", None)),
                                "status": _str(getattr(row, "status", None)),
                                "public_score": score,
                                "description": _str(getattr(row,"description",None))})
            if len(values) < 100:
                break
        return {"status": "READ", "items": records[:40],
                "scores_available": sum(r["public_score"] is not None for r in records)}
    except Exception as err:
        return {"status": "UNAVAILABLE_"+type(err).__name__, "items": [], "scores_available": 0}


def _research(api):
    result = {}
    for day in range(1, 5):
        slug = f"{OWNER}/solar-autoresearch-d{day:02}"
        try:
            owned = api.kernels_list(mine=True, search=slug.rsplit("/",1)[1], page_size=100)
            if owned is None:
                result[str(day)] = {"status":"OWNED_INVENTORY_UNAVAILABLE"}
                continue
            found = [item for item in owned
                     if item is not None and str(getattr(item, "ref", "")).lower() == slug.lower()]
            if not found:
                result[str(day)] = {"status": "NOTEBOOK_NOT_FOUND"}
                continue
            state = _status(api, slug)
            result[str(day)] = {"status":state}
            if state != "COMPLETE":
                continue
            with tempfile.TemporaryDirectory() as output:
                api.kernels_output(slug, output, file_pattern=r"auto14_result\.json$")
                paths=list(Path(output).rglob("auto14_result.json"))
                if len(paths)!=1:
                    result[str(day)]["evidence"]="NO_RESULT_MANIFEST"
                    continue
                info=json.loads(paths[0].read_text())
                result[str(day)]["evidence"] = {
                    k: info.get(k) for k in ("status","source_sha","day",
                                            "best_checkpoint_sha256",
                                            "local_tuning_pq",
                                            "split_months_disjoint",
                                            "boundary_weight")}
        except Exception as err:
            result[str(day)] = {"status": "UNAVAILABLE_"+type(err).__name__}
    return result


def audit(api):
    report = {"snapshot_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
              "competition":COMPETITION,"account":OWNER,
              "submission_attempted":False,
              "new_model_official_score_verified":False,
              "previous_v5_notebook":V5,
              "v5_status":_status(api,V5),
              "submissions":_submissions(api),
              "auto14":_research(api)}
    try:
        with tempfile.TemporaryDirectory() as folder:
            api.kernels_output(V5, folder, file_pattern=r"submission\.csv$")
            candidates=list(Path(folder).rglob("submission.csv"))
            if len(candidates)!=1:
                report["v5_csv"]={"status":"MISSING_OR_NONUNIQUE","count":len(candidates)}
            else:
                report["v5_csv"]={"status":"VALID","integrity":csv_integrity(candidates[0])}
    except Exception as err:
        report["v5_csv"]={"status":"UNAVAILABLE_OR_INVALID_"+type(err).__name__}
    report["assessment"]="NO_NEW_SCORED_CHALLENGER" if not any(
        v.get("evidence",{}).get("status")=="COMPLETE"
        for v in report["auto14"].values()) else "NEW_LOCAL_MODEL_NEEDS_INDEPENDENT_CONFIRMATION"
    return report


def main():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("Missing Kaggle GitHub Actions secret; no Kaggle audit")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi()
    api.authenticate()
    report=audit(api)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("Kaggle audit status:",report["assessment"])
    print("V5 CSV:",report["v5_csv"]["status"])
    print("Submission inventory:",report["submissions"]["status"],
          "count",len(report["submissions"]["items"]))
    print("Research notebooks:",{k:v["status"] for k,v in report["auto14"].items()})
    print("No competition submission was made by this audit.")


if __name__=="__main__":
    main()
