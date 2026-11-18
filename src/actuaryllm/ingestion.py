"""Offline, permission-gated UTF-8 ingestion; no model, OCR or network."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from actuaryllm import storage
from actuaryllm.retrieval import parse_date
from actuaryllm.storage import (
    InputError, file_sha256, http_url, json_text, load_sources, local_path,
    parse_json, read_jsonl,
)

_MAX_BYTES = 1_048_576
_BASE_FIELDS = ("title", "source_id", "domain", "kind", "effective_from", "effective_until", "url")
_REVIEW_FIELDS = {"review_status", "reviewed_by", "reviewed_at", "review_evidence"}


def _path(root: Path, value: Any, *, relative: bool = False) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise InputError("Yerel dosya yolu gerekli.")
    raw = Path(value)
    if relative and raw.is_absolute():
        raise InputError("Manifest yolları root-relative olmalı.")
    if ".." in raw.parts:
        raise InputError("Üst dizin geçişi kabul edilmez.")
    try:
        candidate = raw if raw.is_absolute() else root / raw
        if not candidate.is_relative_to(root):
            raise InputError("Dosya yolu proje dışına çıkamaz.")
        # Reject even in-project links: lexical output policy must not be redirected.
        for part in (candidate, *candidate.parents):
            if part == root:
                break
            if part.is_symlink():
                raise InputError("Ingestion dosya yollarında symlink kabul edilmez.")
        return local_path(root, candidate)
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("Yerel dosya yolu çözümlenemedi.") from exc


def _bytes(path: Path, *, maximum: int = _MAX_BYTES) -> bytes:
    try:
        if not path.is_file():
            raise InputError("Girdi düzenli yerel dosya olmalı.")
        with path.open("rb") as stream:
            raw = stream.read(maximum + 1)
        if not raw or len(raw) > maximum:
            raise InputError("Dosya boş veya 1 MiB sınırını aşıyor.")
        return raw
    except OSError as exc:
        raise InputError("Yerel dosya okunamadı.") from exc


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _string(value: Any, field: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise InputError(f"{field}: bounded dolu string gerekli.")
    return value


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise InputError(f"{field}: tam küçük harf SHA-256 gerekli.")
    return value


def _datetime(value: Any) -> None:
    if not isinstance(value, str) or len(value) > 64 or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)", value
    ):
        raise InputError("reviewed_at: timezone içeren ISO datetime gerekli.")
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError
    except ValueError as exc:
        raise InputError("reviewed_at: geçersiz ISO datetime.") from exc


def _chunks(text: str, limit: int) -> list[tuple[int, int]]:
    """Prefer paragraph ends; preserve *every* Unicode character, including CRLF."""
    ends = [match.end() for match in re.finditer(r"(?:\r?\n)[ \t]*(?:\r?\n)", text)]
    ends.append(len(text))
    result: list[tuple[int, int]] = []
    start = 0
    cursor = 0
    while start < len(text):
        stop = min(start + limit, len(text))
        while cursor < len(ends) and ends[cursor] <= start:
            cursor += 1
        next_cursor = cursor
        while next_cursor < len(ends) and ends[next_cursor] <= stop:
            next_cursor += 1
        if next_cursor > cursor:
            stop = ends[next_cursor - 1]
        result.append((start, stop))
        start = stop
    return result


def _prepare(root: Path, manifest_path: str | Path, *, cache: dict | None = None) -> dict:
    # Lazy import avoids evaluation -> retrieval -> ingestion -> evaluation cycles.
    DOMAINS = ['actuarial_foundations', 'insurance_basics', 'life_pensions', 'non_life', 'operations_reinsurance', 'pricing_risk', 'regulation_reporting', 'reserving']

    manifest_file = _path(root, manifest_path)
    raw_manifest = _bytes(manifest_file)
    try:
        manifest = parse_json(raw_manifest.decode("utf-8"), origin="ingestion manifest")
    except (InputError, UnicodeError) as exc:
        # Strict JSON errors can contain untrusted key names; never echo them.
        raise InputError("Manifest geçerli, unique-key UTF-8 JSON olmalı.") from exc
    if not isinstance(manifest, dict):
        raise InputError("Manifest JSON nesnesi olmalı.")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise InputError("Manifest schema_version=1 gerekli.")
    for field in ("document_id", "source_id", "title", "domain", "kind", "version"):
        _string(manifest.get(field), field)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", manifest["document_id"]):
        raise InputError("document_id: 1–128 güvenli ASCII kimlik karakteri gerekli.")
    if manifest["domain"] not in DOMAINS:
        raise InputError("Manifest domain geçersiz.")
    if manifest["kind"] not in {"official", "synthetic_educational"}:
        raise InputError("Manifest kind geçersiz.")
    for field in ("effective_from", "effective_until", "url"):
        if field not in manifest:
            raise InputError(f"{field} anahtarı gerekli; bilinçli null desteklenir.")
    start = parse_date(manifest["effective_from"], "effective_from")
    end = parse_date(manifest["effective_until"], "effective_until")
    if start is not None and end is not None and end <= start:
        raise InputError("Geçerlilik bitişi başlangıçtan sonra olmalı.")
    http_url(manifest["url"])
    limit = manifest.get("chunk_chars", 2000)
    if type(limit) is not int or not 256 <= limit <= 8000:
        raise InputError("chunk_chars 256–8000 tam sayı olmalı.")
    input_file = _path(root, manifest.get("input_path"), relative=True)
    if input_file.suffix not in {".txt", ".md"}:
        raise InputError("Girdi yalnız .txt veya .md olabilir.")
    input_raw = _bytes(input_file)
    input_hash = _sha(input_raw)
    if _hash(manifest.get("input_sha256"), "input_sha256") != input_hash:
        raise InputError("Raw input SHA-256 manifest ile eşleşmiyor.")
    try:
        text = input_raw.decode("utf-8")
    except UnicodeError as exc:
        raise InputError("Girdi geçerli UTF-8 değil.") from exc
    if not text.strip():
        raise InputError("Girdi metni boş olamaz.")

    sources_file = _path(root, "data/sources.json")
    sources_hash = file_sha256(sources_file)
    try:
        sources = load_sources(root)
    except InputError as exc:
        raise InputError("Kaynak envanteri geçersiz veya okunamadı.") from exc
    if file_sha256(sources_file) != sources_hash:
        raise InputError("Kaynak envanteri inceleme sırasında değişti.")
    source = sources.get(manifest["source_id"])
    if source is None:
        raise InputError("Manifest kaynağı envanterde yok.")
    source_hash = _sha(json_text(source, indent=None).encode("utf-8"))
    if manifest["kind"] == "official":
        if source["source_type"] == "project_original":
            raise InputError("project_original resmi doküman olarak sunulamaz.")
        if start is None or manifest["url"] is None:
            raise InputError("official belge URL, sürüm ve effective_from gerektirir.")

    review = manifest.get("material_review")
    if not isinstance(review, dict):
        raise InputError("material_review nesnesi gerekli.")
    status = review.get("status")
    if not isinstance(status, str) or status not in {"pending", "approved", "blocked"}:
        raise InputError("material_review status geçersiz.")
    for field in ("reviewed_by", "reviewed_at", "evidence_path", "evidence_sha256", "no_personal_data"):
        if field not in review:
            raise InputError(f"material_review {field} gerekli.")
    if review["no_personal_data"] is not None and type(review["no_personal_data"]) is not bool:
        raise InputError("no_personal_data true/false/null olmalı.")
    if review["reviewed_by"] is not None:
        _string(review["reviewed_by"], "reviewed_by")
    if review["reviewed_at"] is not None:
        _datetime(review["reviewed_at"])
    evidence_file = None
    evidence_hash = None
    if review["evidence_path"] is not None:
        evidence_file = _path(root, review["evidence_path"], relative=True)
        evidence_raw = _bytes(evidence_file)
        if not evidence_raw.strip():
            raise InputError("Materyal inceleme kanıtı boş olamaz.")
        evidence_hash = _sha(evidence_raw)
        if _hash(review["evidence_sha256"], "evidence_sha256") != evidence_hash:
            raise InputError("Kanıt SHA-256 eşleşmiyor.")
        try:
            if any(evidence_file.samefile(path) for path in (input_file, manifest_file, sources_file)):
                raise InputError("Kanıt ayrı bir dosya olmalı (hardlink alias dahil).")
        except OSError as exc:
            raise InputError("Kanıt dosya kimliği doğrulanamadı.") from exc
    elif review["evidence_sha256"] is not None:
        raise InputError("Kanıt hash'i için evidence_path gerekli.")
    blockers = []
    if source["rag_permission"] != "approved":
        blockers.append("source_rag_permission_not_approved")
    if status != "approved":
        blockers.append("material_review_not_approved")
    if review["reviewed_by"] is None:
        blockers.append("material_reviewer_required")
    if review["reviewed_at"] is None:
        blockers.append("material_review_date_required")
    if evidence_file is None:
        blockers.append("material_evidence_required")
    if review["no_personal_data"] is not True:
        blockers.append("human_privacy_review_required")

    manifest_hash = _sha(raw_manifest)
    relative_manifest = manifest_file.relative_to(root).as_posix()
    cache_key = (str(root), relative_manifest, manifest_hash, input_hash, source_hash, evidence_hash)
    records = cache.get(cache_key) if cache is not None else None
    if records is None:
        records = []
        version_hash = _sha(manifest["version"].encode("utf-8"))
        for index, (char_start, char_end) in enumerate(_chunks(text, limit)):
            records.append({
                **{field: manifest[field] for field in _BASE_FIELDS},
                "id": f"{manifest['document_id']}:{version_hash}:{input_hash}:{index:06d}",
                "text": text[char_start:char_end], "review_status": "draft",
                "ingestion": {
                    "manifest_path": relative_manifest, "manifest_sha256": manifest_hash,
                    "document_id": manifest["document_id"], "version": manifest["version"],
                    "input_sha256": input_hash, "source_sha256": source_hash,
                    "evidence_sha256": evidence_hash,
                    "chunk_index": index, "char_start": char_start, "char_end": char_end,
                },
            })
        if cache is not None:
            cache[cache_key] = records
    # Loader requires nonblank text. Do not normalize or silently drop whitespace
    # to make an otherwise unindexable bounded chunk appear usable.
    if any(not row["text"].strip() for row in records):
        blockers.append("whitespace_only_chunk_not_indexable")
    snapshot = {manifest_file: manifest_hash, input_file: input_hash, sources_file: sources_hash}
    if evidence_file is not None:
        snapshot[evidence_file] = evidence_hash
    _unchanged(root, snapshot)
    return {
        "records": records, "snapshot": snapshot,
        "report": {
            "ready": not blockers, "blockers": blockers, "executed": False,
            "manifest_path": relative_manifest, "manifest_sha256": manifest_hash,
            "input_sha256": input_hash, "source_sha256": source_hash,
            "sources_sha256": sources_hash, "evidence_sha256": evidence_hash,
            "document_count": 1, "chunk_count": len(records), "input_bytes": len(input_raw),
            "input_chars": len(text), "review_status": "draft",
        },
    }


def _unchanged(root: Path, snapshot: dict[Path, str]) -> None:
    for path, digest in snapshot.items():
        if _path(root, path) != path or file_sha256(path) != digest:
            raise InputError("Manifest/girdi/kaynak/kanıt inceleme sırasında değişti.")


def inspect_ingestion(root: Path, *, manifest_path: str | Path) -> dict:
    """Return bounded metadata only; malformed inputs raise controlled InputError."""
    return _prepare(root.resolve(), manifest_path)["report"]


def _output(root: Path, output_path: str | Path, snapshot: dict) -> Path:
    path = _path(root, output_path)
    corpus = _path(root, "data/corpus")
    staging = _path(root, "artifacts/staging")
    if path.suffix != ".jsonl" or not (path.parent == corpus or path.is_relative_to(staging)):
        raise InputError("Çıktı data/corpus doğrudan çocuğu veya artifacts/staging altında .jsonl olmalı.")
    if path in snapshot or path.exists():
        raise InputError("Mevcut dosya veya provenance girdisi üzerine yazılamaz.")
    return path


def _collisions(root: Path, records: list[dict]) -> None:
    ids = {row["id"] for row in records}
    for path in sorted(_path(root, "data/corpus").glob("*.jsonl")):
        path = _path(root, path)
        try:
            for row in read_jsonl(path):
                if not isinstance(row.get("id"), str):
                    raise InputError("Mevcut korpus id alanı geçersiz.")
                if row["id"] in ids:
                    raise InputError("Chunk kimliği mevcut korpus ile çakışıyor.")
        except InputError as exc:
            # A malformed corpus JSON error may quote an untrusted key name.
            raise InputError("Mevcut korpus geçersiz veya chunk kimliği çakışıyor.") from exc


def ingest_document(root: Path, *, manifest_path: str | Path, output_path: str | Path,
                    execute: bool = False) -> dict:
    """Inspect by default; explicit execution creates a new, draft JSONL only."""
    if type(execute) is not bool:
        raise InputError("execute bool olmalı.")
    root = root.resolve()
    prepared = _prepare(root, manifest_path)
    output = _output(root, output_path, prepared["snapshot"])
    _collisions(root, prepared["records"])
    report = {**prepared["report"], "output_path": output.relative_to(root).as_posix()}
    if not execute:
        return report
    if not report["ready"]:
        raise InputError("Ingestion kapıları kapalı: " + ", ".join(report["blockers"]))
    _unchanged(root, prepared["snapshot"])
    _output(root, output_path, prepared["snapshot"])
    storage.write_jsonl(output, prepared["records"], overwrite=False)
    return {**report, "executed": True, "output_sha256": file_sha256(output)}


def validate_ingested_record(root: Path, record: dict, *, cache: dict | None = None) -> None:
    """Fail closed against current approvals and exact deterministic expected chunk.

    Only human content-review fields may differ. Caller owns a load-local cache;
    approvals and raw hashes are always re-read, even if that cache is reused.
    """
    if not isinstance(record, dict) or not isinstance(record.get("ingestion"), dict):
        raise InputError("Ingestion provenance nesnesi gerekli.")
    provenance = record["ingestion"]
    manifest_path = provenance.get("manifest_path")
    if not isinstance(manifest_path, str) or Path(manifest_path).is_absolute():
        raise InputError("Ingestion manifest_path root-relative olmalı.")
    prepared = _prepare(root.resolve(), manifest_path, cache=cache)
    if not prepared["report"]["ready"]:
        raise InputError("Ingestion materyal/kaynak/gizlilik onayı artık geçerli değil.")
    index = provenance.get("chunk_index")
    if type(index) is not int or not 0 <= index < len(prepared["records"]):
        raise InputError("Ingestion chunk_index geçersiz.")
    expected = prepared["records"][index]
    if {key: value for key, value in record.items() if key not in _REVIEW_FIELDS} != {
        key: value for key, value in expected.items() if key not in _REVIEW_FIELDS
    }:
        raise InputError("Korpus chunk içeriği/provenance beklenen kayıtla eşleşmiyor.")
    # dict equality alone accepts True == 1; provenance offsets require actual ints.
    for field in ("chunk_index", "char_start", "char_end"):
        if type(provenance.get(field)) is not int:
            raise InputError("Ingestion offset/index tam sayı olmalı.")
