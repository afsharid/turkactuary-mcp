"""Strict local JSON I/O, provenance and project path boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit


class InputError(ValueError):
    """A user-facing invalid input or unsafe operation."""


def project_root(value: str | Path | None = None) -> Path:
    root = Path(value).expanduser() if value is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    if not root.is_dir():
        raise InputError(f"Proje dizini bulunamadı: {root}")
    return root


def local_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise InputError("Dosya yolu proje dizininin dışına çıkamaz.")
    return path


def _invalid_constant(value: str) -> None:
    raise InputError(f"JSON sonlu olmayan sayı içeremez: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        try:
            key.encode("utf-8")
        except UnicodeError as exc:
            raise InputError("JSON anahtarı geçerli Unicode değil.") from exc
        if key in result:
            raise InputError(f"Yinelenen JSON anahtarı: {key}")
        result[key] = value
    return result


def _unicode_strings(value: Any) -> None:
    # JSON allows escaped lone UTF-16 surrogates; our UTF-8 data/hashes do not.
    # Walk iteratively so validation does not impose another recursion limit.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            item.encode("utf-8")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def parse_json(text: str, *, origin: str = "JSON") -> Any:
    try:
        value = json.loads(text, parse_float=Decimal, parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
        _unicode_strings(value)
        return value
    except (json.JSONDecodeError, ValueError, DecimalException, RecursionError, UnicodeError) as exc:
        raise InputError(f"{origin}: geçersiz JSON: {exc}") from exc


def read_json(path: Path) -> Any:
    try:
        return parse_json(path.read_text(encoding="utf-8"), origin=str(path))
    except UnicodeError as exc:
        raise InputError(f"Dosya geçerli UTF-8 değil: {path.name}") from exc
    except OSError as exc:
        raise InputError(f"Dosya okunamadı: {path.name}: {exc.strerror}") from exc


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if len(line) > 1_000_000:
                    raise InputError(f"{path.name}:{number}: JSONL satırı çok büyük.")
                value = parse_json(line, origin=f"{path.name}:{number}")
                if not isinstance(value, dict):
                    raise InputError(f"{path.name}:{number}: JSON nesnesi bekleniyor.")
                yield value
    except UnicodeError as exc:
        raise InputError(f"Dosya geçerli UTF-8 değil: {path.name}") from exc
    except OSError as exc:
        raise InputError(f"Dosya okunamadı: {path.name}: {exc.strerror}") from exc


def json_text(value: Any, *, indent: int | None = 2) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, Decimal):
            if not item.is_finite():
                raise InputError("Sonlu olmayan Decimal çıktı olamaz.")
            return str(item)
        raise TypeError(f"JSON çıktısı desteklenmiyor: {type(item).__name__}")
    try:
        text = json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False, default=encode)
        text.encode("utf-8")
        return text
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise InputError(f"JSON çıktısı oluşturulamadı: {exc}") from exc


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Publish complete local output atomically; never clobber by default."""
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Linking is an atomic create-if-absent; an existence check + rename
            # would race with another agent publishing the same output name.
            os.link(temporary, path)
    except FileExistsError as exc:
        raise InputError("Çıktı veya üst dizin zaten dosya olarak var; yolu kontrol edin. Mevcut çıktı için açık --overwrite gerekir.") from exc
    except (OSError, UnicodeError) as exc:
        raise InputError(f"Çıktı yazılamadı: {path.name}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # A failed cleanup cannot expose a partial .json/.jsonl file.
                # Scratch .tmp files are never loaded as data or corpus.
                pass


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    _write_text(path, json_text(value) + "\n", overwrite=overwrite)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], *, overwrite: bool = False) -> None:
    lines: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise InputError("JSONL çıktısının her satırı nesne olmalı.")
        line = json_text(record, indent=None)
        if len(line) > 1_000_000:
            raise InputError("JSONL çıktı satırı çok büyük.")
        lines.append(line)
    if not lines:
        raise InputError("JSONL çıktı boş olamaz.")
    _write_text(path, "\n".join(lines) + "\n", overwrite=overwrite)


def file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65_536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise InputError(f"Hash için dosya okunamadı: {path.name}") from exc


def text_field(record: dict[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{record.get('id', 'kayıt')}: {name} dolu bir string olmalı.")
    return value


def http_url(value: Any, *, field: str = "url") -> str | None:
    """Validate public provenance URLs without echoing potential credentials."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field}: http/https URL veya null olmalı.")
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("http/https ve hostname gerekli")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URL kimlik bilgisi içeremez")
        # Accessing port also validates malformed or out-of-range port numbers.
        _ = parsed.port
    except ValueError as exc:
        raise InputError(f"{field}: geçersiz veya kimlik bilgisi içeren URL.") from exc
    return value


def load_sources(root: Path) -> dict[str, dict[str, Any]]:
    records = read_json(local_path(root, "data/sources.json"))
    if not isinstance(records, list) or not records:
        raise InputError("Kaynak envanteri boş olmayan bir JSON listesi olmalı.")
    sources: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise InputError("Kaynak kaydı nesne olmalı.")
        key = text_field(record, "id")
        if key in sources:
            raise InputError(f"Tekrarlanan kaynak id: {key}")
        for field in ("name", "source_type", "license_status", "notes"):
            text_field(record, field)
        for field in ("rag_permission", "training_permission", "publication_permission"):
            if text_field(record, field) not in {"approved", "pending", "blocked"}:
                raise InputError(f"{key}: {field} izin değeri geçersiz.")
        if "url" not in record:
            raise InputError(f"{key}: url alanı gerekli; özgün kaynaktaysa null kullanın.")
        http_url(record["url"], field=f"{key}: url")
        sources[key] = record
    return sources
