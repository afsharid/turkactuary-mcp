"""Seçimli hibrit arama (v5): BM25 (v2) + yerel embedding, mevzuat işareti olmayan sorularda resmî parça kapısı.

Yalnız loopback embedding sunucusu (OpenAI-uyumlu `/v1/embeddings`, ör. LM Studio `text-embedding-qwen3-embedding-4b`).
Belge vektörleri önceden hesaplanmış yerel önbellekten okunur (varsayılan `.cache/dense-embeddings-v2.json`, git dışı;
`scripts/retrieval_dense_experiment.py` üretir). Sunucu ya da vektör yoksa açık `InputError`; sessizce BM25'e düşülmez.
Ölçüm ve ağırlık seçimi: artifacts/reviews/retrieval-dense/REPORT-002.md (dev v1–v4'e göre w=0,6; λ=0,6).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from actuaryllm.storage import InputError, json_text, local_path, parse_json, project_root

# Sorgu vektörü önbelleği (ölçüm için): ayarlıysa v5 sorgu vektörünü bu dosyadan okur, embedding sunucusu gerekmez.
QUERY_CACHE_ENV = "ACTUARYLLM_EMBED_QUERY_CACHE"
INSTRUCT = "Instruct: Türk sigortacılık mevzuatı hakkındaki soruyu yanıtlayan mevzuat parçalarını getir\nQuery: "
# Mevzuat işareti (scripts/retrieval_v4_experiment.py LEGAL ile aynı): yoksa resmî parçaların puanı gate_lambda ile çarpılır.
LEGAL_CUES = re.compile(
    r"\b(kanun\w*|yönetmeli\w*|tebliğ\w*|genel şart\w*|madde\w*|mevzuat\w*|kurum(?:ca|un|a)?\b|seddk|müsteşarl\w*|bakan\w*|"
    r"ruhsat\w*|levha\w*|sicil\w*|izn\w*|izin\w*|yükümlü\w*|zorunlu\w*|ceza\w*|tahkim\w*|komisyon\w*|hakem\w*|acente\w*|"
    r"broker\w*|eksper\w*|dask|örym|güvence hesab\w*|devlet katkı\w*|bireysel emeklilik\w*|aktüerler\w*|sermaye yeterlili\w*|"
    r"teknik karşılık\w*|denetim\w*|uyum\w*|resmî gazete|rg\b|süre(?:si|sinde)?\b|gün içinde|ay içinde|yıl içinde)", re.I)


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


VECTOR_STORE_ENV = "ACTUARYLLM_EMBED_VECTOR_STORE"


def unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def text_key(text: str, model: str) -> str:
    return hashlib.sha256((model + "\x00" + text).encode("utf-8")).hexdigest()


def document_text(document: Any) -> str:
    return document.title + "\n" + document.text


class EmbeddingClient:
    """Yalnız loopback; proxy ve yönlendirme kapalı."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 60):
        from actuaryllm.local_client import _NoRedirect, loopback_url
        self.base_url = loopback_url(base_url)
        if not isinstance(model, str) or not model.strip():
            raise InputError("Embedding model kimliği gerekli.")
        self.model = model
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = json_text({"model": self.model, "input": texts}, indent=None).encode("utf-8")
        request = Request(self.base_url + "/embeddings", data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = response.read(50_000_001)
        except HTTPError as exc:
            raise InputError(f"Embedding sunucusu HTTP hatası: {exc.code}; v5 için {self.model} yüklü olmalı.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise InputError(f"Embedding sunucusuna erişilemedi ({self.base_url}); v5 profili yerel embedding sunucusu "
                             f"gerektirir ({self.model}). Başka profil (ör. v2) seçin ya da sunucuyu başlatın.") from exc
        try:
            value = parse_json(payload.decode("utf-8"), origin="Embedding yanıtı")
            data = sorted(value["data"], key=lambda item: item["index"])
            vectors = [unit([float(x) for x in item["embedding"]]) for item in data]
        except (UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
            raise InputError("Embedding sunucusu beklenen biçimde yanıt vermedi.") from exc
        if len(vectors) != len(texts):
            raise InputError("Embedding yanıtındaki vektör sayısı istekle eşleşmiyor.")
        return vectors


class VectorStore:
    """Belge vektör önbelleği: {"model", "vectors": {sha256(model + NUL + başlık\\nmetin): [..]}}."""

    def __init__(self, vectors: dict[str, list[float]], model: str):
        self.vectors, self.model = vectors, model

    @classmethod
    def load(cls, root: Path, path: str, model: str) -> "VectorStore":
        file = local_path(root, path)
        if not file.is_file():
            raise InputError(f"v5 belge vektörleri yok: {path}. scripts/retrieval_dense_experiment.py ile oluşturun.")
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("model") != model or not isinstance(data.get("vectors"), dict):
            raise InputError(f"{path}: {model} için vektör önbelleği değil.")
        return cls(data["vectors"], model)

    def vector(self, document: Any) -> list[float]:
        vec = self.vectors.get(text_key(document_text(document), self.model))
        if vec is None:
            raise InputError(f"v5: {document.cite_id} için belge vektörü yok (korpus değişmiş olabilir); "
                             "scripts/retrieval_dense_experiment.py ile önbelleği güncelleyin.")
        return vec


class CachedQueryEmbedder:
    """Önceden hesaplanmış sorgu vektörleri: {"model", "vectors": {sha256(model + NUL + INSTRUCT+soru): [..]}}.

    Belge vektörleriyle aynı model ve nicemlemeyle üretilmelidir (scripts/build_query_vector_cache.py). Önbellekte
    olmayan sorgu açık InputError verir; sessizce sunucuya veya BM25'e düşülmez.
    """

    def __init__(self, vectors: dict[str, list[float]], model: str):
        self.vectors, self.model = vectors, model

    @classmethod
    def load(cls, root: Path, path: str, model: str) -> "CachedQueryEmbedder":
        file = local_path(root, path)
        if not file.is_file():
            raise InputError(f"v5 sorgu vektörü önbelleği yok: {path}. scripts/build_query_vector_cache.py ile oluşturun.")
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("model") != model or not isinstance(data.get("vectors"), dict):
            raise InputError(f"{path}: {model} için sorgu vektörü önbelleği değil.")
        return cls(data["vectors"], model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = self.vectors.get(text_key(text, self.model))
            if vec is None:
                raise InputError("v5: sorgu vektörü önbellekte yok (soru önbellek üretildikten sonra değişmiş olabilir); "
                                 "scripts/build_query_vector_cache.py ile güncelleyin ya da önbelleği kapatın.")
            out.append(vec)
        return out


_STORES: dict[tuple[str, str, str], VectorStore] = {}
_QUERY_CACHES: dict[tuple[str, str, str], CachedQueryEmbedder] = {}


def default_embedder(root: Path, h: dict[str, Any]) -> Embedder:
    cache = os.environ.get(QUERY_CACHE_ENV)
    if not cache:
        return EmbeddingClient(h["base_url"], h["embedding_model"])
    key = (str(root), cache, h["embedding_model"])
    if key not in _QUERY_CACHES:
        _QUERY_CACHES[key] = CachedQueryEmbedder.load(root, cache, h["embedding_model"])
    return _QUERY_CACHES[key]


def hybrid_search(index: Any, query: str, profile: dict[str, Any], *, as_of: str | None = None,
                  embedder: Embedder | None = None, store: VectorStore | None = None, root: Path | None = None) -> list[dict[str, Any]]:
    """BM25 ilk `bm25_candidates` + dense ilk `dense_candidates`; min-max normalize ağırlıklı toplam; kapı; ilk `limit`."""
    from actuaryllm.retrieval import parse_date
    h = profile["hybrid"]
    root = root or project_root()
    embedder = embedder or default_embedder(root, h)
    if store is None:
        path = os.environ.get(VECTOR_STORE_ENV) or h["vector_store"]  # deney: madde korpusu vektörleri (11-12)
        key = (str(root), path, h["embedding_model"])
        store = _STORES.get(key) or VectorStore.load(root, path, h["embedding_model"])
        _STORES[key] = store
    when = parse_date(as_of, "as_of")
    documents = [d for d in index.corpus.documents if d.valid_on(when)]
    vectors = [store.vector(d) for d in documents]  # eksik vektör: InputError (sorgu gömülmeden önce)
    bm = index.search(query, limit=h["bm25_candidates"], as_of=as_of, full_text=True)
    by_cite = {d.cite_id: d for d in documents}
    qv = embedder.embed([INSTRUCT + query])[0]
    dense = sorted(((sum(a * b for a, b in zip(qv, v)), d) for v, d in zip(vectors, documents)),
                   key=lambda item: (-item[0], item[1].id))[: h["dense_candidates"]]
    w = h["weight"]
    combined: dict[str, float] = {}
    bmax = max((x["score"] for x in bm), default=1.0) or 1.0
    for x in bm:
        combined[x["id"]] = combined.get(x["id"], 0.0) + (1 - w) * x["score"] / bmax
    if dense:
        dmax, dmin = dense[0][0], dense[-1][0]
        for score, d in dense:
            combined[d.cite_id] = combined.get(d.cite_id, 0.0) + w * (score - dmin) / ((dmax - dmin) or 1.0)
    legal = bool(LEGAL_CUES.search(query))

    def gated(cite: str) -> float:
        doc = by_cite[cite]
        return combined[cite] * (h["gate_lambda"] if (doc.kind == "official" and not legal) else 1.0)

    ranked = sorted(combined, key=lambda c: (-gated(c), by_cite[c].id))[: profile["limit"]]
    return [by_cite[c].evidence(gated(c), query, full_text=profile["full_text"]) for c in ranked]
