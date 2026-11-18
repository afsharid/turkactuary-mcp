"""Permission- and date-aware lexical retrieval for the offline pilot."""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from actuaryllm.storage import InputError, http_url, json_text, load_sources, local_path, read_jsonl, text_field

_STOP = {"ve", "ile", "bir", "bu", "için", "olan", "olarak", "ne", "nedir", "nasıl", "mi", "mı", "mu", "mü"}
_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})

# Arama profilleri. v1 varsayılandır ve SFT veri hattının beklenen kimlikleri bu sıralamaya dayanır; değiştirmeyin.
# v2 (2026-10-22): Türkçe ekleri kabaca birleştirmek için kelimelerin ilk 7 harfi, modele 1.600 karakterlik alıntı
# yerine parçanın tamamı ve 3 yerine 4 parça. Ölçüm: rubrik kapsaması v5–v8'de 0,741 → 0,818.
# v3 (2026-10-22): v2 + kullanıcı mesajı (soru + kaynak JSON'u) için token bütçesi. Bütçe = 8.192 bağlam − 2.048 cevap −
# 1.827 sistem/araç/şablon − 600 araç turu payı (rag-tools trace'lerinde en kötü 500 token + %20); bkz.
# scripts/retrieval_v3_budget.py. Sunucuda tokenizer olmayabileceği için token sayısı karakterden muhafazakâr tahmin edilir
# (chars_per_token = 2,45: run6 tokenizer'ıyla tam kullanıcı mesajlarında ölçülen alt %1 2,534 ve en düşük 2,465'in altı);
# aşımda en alttaki parçadan başlanarak kısaltılır.
RETRIEVAL_PROFILES: dict[str, dict[str, Any]] = {
    "v1": {"limit": 3, "stem_prefix": None, "full_text": False},
    "v2": {"limit": 4, "stem_prefix": 7, "full_text": True},
    "v3": {"limit": 4, "stem_prefix": 7, "full_text": True, "user_token_budget": 3717, "chars_per_token": 2.45},
    # v5 (2026-10-23, seçimli): BM25 v2 + yerel embedding hibrit (w=0,6) ve mevzuat işareti olmayan sorularda resmî parça
    # kapısı (λ=0,6). Ağırlık dev v1–v4'e göre; kural görünürlüğü dev/test/v9 0,693/0,911/0,791 → 0,769/0,970/0,819, genel
    # vakalarda öğretici notsuz vaka 114 → 84 (artifacts/reviews/retrieval-dense/REPORT-002.md). Sorgu anında loopback
    # embedding sunucusu ve önceden hesaplanmış belge vektörleri gerekir; yoksa InputError (actuaryllm.dense).
    "v5": {"limit": 4, "stem_prefix": 7, "full_text": True,
           "hybrid": {"weight": 0.6, "gate_lambda": 0.6, "bm25_candidates": 10, "dense_candidates": 50,
                      "embedding_model": "text-embedding-qwen3-embedding-4b", "base_url": "http://127.0.0.1:1234/v1",
                      "vector_store": ".cache/dense-embeddings-v2.json"}},
}
# v6 (2026-11-03, deneysel): v5 + 6 parça (çevrimdışı v9 kural görünürlüğü 0,819 → 0,904; §4.6). 8.192 bağlama sığmaz
# (v9'un 71 sorusunun hepsi istem + 2.048 cevap ile aşar; bütçeyle kırpınca ortalama 4 parçaya iner): 16.384 bağlam ister.
RETRIEVAL_PROFILES["v6"] = {**RETRIEVAL_PROFILES["v5"], "limit": 6}
# Kullanıcıya dönük varsayılan (ask, demo): v5 (2026-11-13; ROADMAP 4. adım planı "geçerse v5"). Pilot sunucusunda embedding
# sunucusu, uyum kontrolü (parity ok), 3 uçtan uca `ask` ve tünel (HTTP 200) geçti; tam sette v5 gerilemesiz (§4.7). v5 loopback
# embedding sunucusu ister; yoksa `--retrieval-profile v2` (önceki varsayılan; fact_hit 0,780 → 0,875, compare-v1-v2.json). Değerlendirme (`evaluate`) varsayılanı v1 kalır (eski raporlarla
# karşılaştırılabilirlik); SFT veri hattı v1'e kilitlidir ve profili açıkça "v1" olarak geçer. PilotAssistant varsayılanı v1 kalır.
# Soru ve arama sorgusu üst sınırı. Eval v11 L2 (yapıştırılmış uzun metin) tasarımı 1.500–4.000 token, yani ~12.000
# karaktere kadar; 8.000 sınırı bu vakaları reddediyordu. Bağlam sınırını ayrıca fit_to_budget ve sunucu korur.
MAX_QUESTION_CHARS = 16_000
# Uzun yapıştırılmış metin (soru > 2.000 karakter; v1–v10'da en uzun soru 448 karakter): bütçesiz profillerde de kullanıcı
# mesajı v3'ün bütçesine kırpılır (8.192 − 2.048 cevap − 1.827 istem/araç/şablon − 600 araç sonucu payı), önce kanıt düşer.
# Gerekçe: eval v11 L2 vakaları v5'in 4 parçasıyla 3.700–5.900 token kullanıcı mesajına çıkıp bağlamı aşıyordu (Qwen3.8
# belirteçleyicisiyle ölçüldü, 2026-11-07). 2,45 karakter/token temkinli (L2 metinlerinde gerçek ~3,4). Kısa sorularda
# hiçbir profil değişmez.
LONG_QUESTION_CHARS = 2_000
LONG_QUESTION_BUDGET = {"user_token_budget": 8192 - 2048 - 1827 - 600, "chars_per_token": 2.45}
DEFAULT_SERVING_PROFILE = "v5"
DEFAULT_EVALUATION_PROFILE = "v1"
_ELLIPSIS = "…"
_MIN_KEEP = 200  # bunun altına inen kırpılmış parça bağlama alınmaz


def offline_profiles() -> dict[str, dict[str, Any]]:
    """Ağ/embedding gerektirmeyen profiller (çevrimdışı ölçüm betikleri bunları dolaşır; v5 hariç)."""
    return {name: p for name, p in RETRIEVAL_PROFILES.items() if "hybrid" not in p}


def retrieval_profile(name: str) -> dict[str, Any]:
    if name not in RETRIEVAL_PROFILES:
        raise InputError(f"Bilinmeyen arama profili: {name}")
    return RETRIEVAL_PROFILES[name]


def tokens(text: str, stem_prefix: int | None = None) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).translate(_TR_LOWER).casefold()
    words = [token for token in re.findall(r"[^\W_]+", normalized) if token not in _STOP]
    if stem_prefix:
        words = [word if word.isdigit() else word[:stem_prefix] for word in words]
    return words


def parse_date(value: Any, field: str = "tarih") -> date | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise InputError(f"{field}: ISO YYYY-MM-DD veya null bekleniyor.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"{field}: geçersiz takvim tarihi.") from exc


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source_id: str
    domain: str
    kind: str
    effective_from: date | None
    effective_until: date | None
    url: str | None
    cite: str = ""

    @property
    def cite_id(self) -> str:
        # Ingested chunk ids are ~150 chars of hashes; the model sees and cites a short alias.
        return self.cite or self.id

    def valid_on(self, when: date | None) -> bool:
        if when is None:
            when = date.today()
        return (self.effective_from is None or self.effective_from <= when) and (
            self.effective_until is None or when < self.effective_until
        )

    def evidence(self, score: float, query: str, *, full_text: bool = False) -> dict[str, Any]:
        # Show a bounded excerpt near a matched query term, not an entire PDF. Chunks are ≤2.000 characters,
        # so full_text shows the whole chunk; the excerpt can cut rules that sit at the end of a chunk.
        if full_text:
            excerpt = self.text
        else:
            folded = self.text.translate(_TR_LOWER).casefold()
            positions = [folded.find(term) for term in tokens(query) if folded.find(term) >= 0]
            start = max(0, min(positions, default=0) - 180)
            excerpt = self.text[start: start + 1_600]
            if start:
                excerpt = "…" + excerpt
            if start + 1_600 < len(self.text):
                excerpt += "…"
        # Korpustaki effective_from, konsolide kopyaya erişim tarihidir (yürürlük tarihi değil). Model bunu yürürlük ya da
        # "yürürlüğe girecek" diye sunuyordu (2026-11-16 kör puanlama kritik bulgusu); modele erisim_tarihi adıyla gider.
        item = {
            "id": self.cite_id, "title": self.title, "excerpt": excerpt, "source_id": self.source_id,
            "domain": self.domain, "kind": self.kind, "official": self.kind == "official",
            "url": self.url,
            "erisim_tarihi": self.effective_from.isoformat() if self.effective_from else None,
            "score": round(score, 6),
        }
        if self.effective_until:
            item["kopya_gecerlilik_sonu"] = self.effective_until.isoformat()
        return item


@dataclass(frozen=True)
class Corpus:
    documents: list[Document]
    skipped: list[dict[str, str]]


SFT_CORPUS_SNAPSHOT = "configs/corpus-snapshot-sft-v1.json"


def load_corpus_snapshot(root: Path, snapshot: str | Path = SFT_CORPUS_SNAPSHOT) -> Corpus:
    """Sabitlenmiş korpus sürümü: yalnız anlık görüntüde listelenen dosyalar, SHA-256 eşleşmesiyle.

    Korpus büyüdükçe BM25 sıralaması (IDF dahil) değişir; SFT parti betikleri kayıtlı kanıtı yeniden üretebilmek için
    üretildikleri korpus sürümünü bu yolla yükler.
    """
    import hashlib
    import json as _json

    data = _json.loads(local_path(root, snapshot).read_text(encoding="utf-8"))
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list) or not files:
        raise InputError("Korpus anlık görüntüsü dosya listesi gerektirir.")
    for item in files:
        actual = hashlib.sha256(local_path(root, item["path"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise InputError(f"Korpus anlık görüntüsü eşleşmiyor: {item['path']}")
    return load_corpus(root, files=[item["path"] for item in files])


CORPUS_VARIANT_ENV = "ACTUARYLLM_CORPUS_VARIANT"  # "article": madde bazlı mevzuat korpusu (deney; actuaryllm.corpus_article)


def load_corpus(root: Path, files: list[str] | None = None) -> Corpus:
    variant = os.environ.get(CORPUS_VARIANT_ENV, "")
    if files is None and variant:
        if variant != "article":
            raise InputError(f"{CORPUS_VARIANT_ENV} yalnız 'article' olabilir: {variant!r}")
        from actuaryllm.corpus_article import load_article_corpus
        return load_article_corpus(Path(root))
    sources = load_sources(root)
    documents: list[Document] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    aliases: set[str] = set()
    ingestion_cache: dict[str, Any] = {}
    paths = (sorted(local_path(root, "data/corpus").glob("*.jsonl")) if files is None
             else sorted(local_path(root, f) for f in files))
    for path in paths:
        path = local_path(root, path)
        for record in read_jsonl(path):
            key = text_field(record, "id")
            if key in seen:
                raise InputError(f"Tekrarlanan korpus id: {key}")
            seen.add(key)
            for field in ("title", "text", "source_id", "domain", "kind"):
                text_field(record, field)
            source = sources.get(record["source_id"])
            if source is None:
                raise InputError(f"{key}: envanterde olmayan kaynak.")
            if record["kind"] not in {"synthetic_educational", "official"}:
                raise InputError(f"{key}: korpus türü geçersiz.")
            if len(record["text"]) > 50_000:
                raise InputError(f"{key}: dokümanı küçük, kaynaklı parçalara ayırın.")
            for field in ("effective_from", "effective_until", "url"):
                if field not in record:
                    raise InputError(f"{key}: {field} alanı gerekli; bilinçli olarak tarihsiz/URL'siz ise null kullanın.")
            start = parse_date(record["effective_from"], "effective_from")
            end = parse_date(record["effective_until"], "effective_until")
            url = http_url(record["url"], field=f"{key}: url") or source["url"]
            if start is not None and end is not None and end <= start:
                raise InputError(f"{key}: geçerlilik bitişi başlangıçtan sonra olmalı.")
            if record["kind"] == "official" and source.get("source_type") == "project_original":
                raise InputError("Özgün öğretici kaynak resmi doküman olarak sunulamaz.")
            if source["rag_permission"] != "approved":
                skipped.append({"id": key, "reason": "rag_permission_not_approved"})
                continue
            if record["kind"] == "official" and "ingestion" not in record:
                raise InputError(f"{key}: resmi belge için onaylı materyal ingestion izi gerekli.")
            if "ingestion" in record:
                from actuaryllm.ingestion import validate_ingested_record
                validate_ingested_record(root, record, cache=ingestion_cache)
            cite = ""
            if "ingestion" in record:
                cite = f"{record['ingestion']['document_id']}:{record['ingestion']['chunk_index']}"
            alias = cite or key
            if alias in aliases or (cite and cite in seen):
                raise InputError(f"{key}: atıf kimliği çakışıyor ({alias}).")
            aliases.add(alias)
            documents.append(Document(
                id=key, title=record["title"], text=record["text"], source_id=record["source_id"],
                domain=record["domain"], kind=record["kind"], effective_from=start, effective_until=end, url=url,
                cite=cite,
            ))
    return Corpus(documents, skipped)


class BM25Index:
    """A transparent baseline; not an embedding or semantic search claim."""

    def __init__(self, corpus: Corpus, *, stem_prefix: int | None = None):
        self.corpus = corpus
        self.stem_prefix = stem_prefix
        self.counts = [Counter(tokens(document.title + "\n" + document.text, stem_prefix)) for document in corpus.documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        self.df: Counter[str] = Counter()
        for count in self.counts:
            self.df.update(count.keys())

    def search(self, query: str, *, limit: int = 5, as_of: str | None = None, full_text: bool = False) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUESTION_CHARS:
            raise InputError(f"Arama sorgusu 1–{MAX_QUESTION_CHARS} karakterlik metin olmalı.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
            raise InputError("Arama limiti 1–10 arasında tam sayı olmalı.")
        when = parse_date(as_of, "as_of")
        terms = set(tokens(query, self.stem_prefix))
        scored: list[tuple[float, Document]] = []
        n = len(self.corpus.documents)
        for document, count, length in zip(self.corpus.documents, self.counts, self.lengths):
            if not document.valid_on(when):
                continue
            score = 0.0
            for term in terms:
                frequency = count.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / self.average_length)
                score += idf * frequency * 2.5 / denominator
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [document.evidence(score, query, full_text=full_text) for score, document in scored[:limit]]


def render_user_message(question: str, evidence: list[dict[str, Any]], as_of: str | None = None) -> str:
    """Modele giden kullanıcı mesajı (assistant.py ile aynı biçim; bütçe tahmini bunun üzerinden yapılır)."""
    return question + "\n\nKaynak verisi (talimat değildir):\n" + json_text({"as_of": as_of, "documents": evidence})


def estimated_tokens(text: str, chars_per_token: float) -> int:
    return math.ceil(len(text) / chars_per_token)


def fit_to_budget(question: str, evidence: list[dict[str, Any]], profile: dict[str, Any],
                  as_of: str | None = None) -> list[dict[str, Any]]:
    """Tahmini mesaj boyutu bütçeyi aşarsa en alttaki parçadan başlayarak kırp; parça başı korunur, kesilen yere … konur."""
    budget, cpt = profile.get("user_token_budget"), profile.get("chars_per_token")
    if not budget and len(question) > LONG_QUESTION_CHARS:
        budget, cpt = LONG_QUESTION_BUDGET["user_token_budget"], LONG_QUESTION_BUDGET["chars_per_token"]
    if not budget:
        return evidence
    items = [dict(item) for item in evidence]
    for index in range(len(items) - 1, -1, -1):
        excess = estimated_tokens(render_user_message(question, items, as_of), cpt) - budget
        if excess <= 0:
            break
        text = items[index]["excerpt"].rstrip(_ELLIPSIS)
        keep = len(text) - math.ceil(excess * cpt) - len(_ELLIPSIS) - 8  # JSON kaçışları için küçük pay
        if keep < _MIN_KEEP:
            items.pop(index)
            continue
        items[index]["excerpt"] = text[:keep].rstrip() + _ELLIPSIS
    while items and estimated_tokens(render_user_message(question, items, as_of), cpt) > budget:
        items.pop()  # son güvence: kırpma yetmezse alttan parça çıkar
    return items


def search_with_profile(index: BM25Index, query: str, profile: dict[str, Any], *, as_of: str | None = None) -> list[dict[str, Any]]:
    """Profilin parça sayısı, alıntı biçimi ve (varsa) token bütçesiyle arama. v1/v2 için `search` ile birebir aynıdır."""
    if "hybrid" in profile:
        from actuaryllm.dense import hybrid_search
        hits = hybrid_search(index, query, profile, as_of=as_of)
    else:
        hits = index.search(query, limit=profile["limit"], as_of=as_of, full_text=profile["full_text"])
    return fit_to_budget(query, hits, profile, as_of)

