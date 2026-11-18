"""OpenAI-compatible chat, strictly on loopback with proxies/redirects disabled."""

from __future__ import annotations

import ipaddress
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from actuaryllm.storage import InputError, json_text, parse_json


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise InputError("Yerel model çağrısında HTTP yönlendirmesi yasak.")


def loopback_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
        host = parsed.hostname
    except (ValueError, TypeError) as exc:
        raise InputError("Geçersiz yerel model URL'si.") from exc
    if parsed.scheme not in {"http", "https"} or not host:
        raise InputError("Yerel model URL'si http(s) ve loopback adresi kullanmalı.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise InputError("Model URL'sine anahtar, kimlik bilgisi, query veya fragment koymayın.")
    if parsed.path.rstrip("/") not in {"", "/v1"}:
        raise InputError("Yerel model base URL yolu boş veya /v1 olmalı.")
    if host == "localhost":
        host = "127.0.0.1"  # Avoid DNS, proxy configuration and nonlocal localhost remapping.
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise InputError("Bu pilot yalnız loopback IP/localhost kabul eder; uzak API kullanmaz.") from exc
    if not address.is_loopback:
        raise InputError("Uzak model sunucusu bloklandı; yalnız loopback kullanılabilir.")
    netloc = f"[{address}]" if address.version == 6 else str(address)
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme, netloc, "/v1", "", ""))


class LocalChatClient:
    def __init__(self, base_url: str, model: str, *, max_tokens: int = 2048, timeout: float = 60, no_think: bool = False):
        self.base_url = loopback_url(base_url)
        if not isinstance(model, str) or not model.strip() or len(model) > 256:
            raise InputError("Yerel model id gerekli.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
            raise InputError("max_tokens 1–8192 aralığında tam sayı olmalı.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise InputError("timeout 0–300 saniye aralığında olmalı.")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Qwen3 düşünme modunu sunucu ayarı olmadan kapatır (ör. LM Studio): sistem mesajının sonuna "/no_think".
        # vLLM akışında düşünme sunucu tarafında kapatıldığı için orada gerekmez.
        self.no_think = bool(no_think)
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self.no_think:
            messages = [dict(message) for message in messages]
            if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
                messages[0]["content"] = messages[0]["content"].rstrip() + "\n/no_think"
            else:
                messages.insert(0, {"role": "system", "content": "/no_think"})
        body: dict[str, Any] = {
            "model": self.model, "messages": messages, "max_tokens": self.max_tokens, "temperature": 0.2,
        }
        if tools:
            body["tools"] = tools
        try:
            return self._post(body)
        except InputError as exc:
            # Bağlam aşımı (vLLM 400): cevap payı kalan yere sığacak kadar küçültülüp bir kez yeniden denenir. 2026-11-12 FP8 v4:
            # 257 karakterlik soruda 4 tam parça 6.145 istem token'ı tuttu; 2.048 cevap payıyla 8.192 bir token aşıldı.
            # 2026-11-15 FP8 v4: vLLM'in "at least N" sayısı alt sınırdır; ilk hata 6.145, yeniden denemedeki hata 6.162 dedi.
            # Bu yüzden pay 128 token ve en çok iki yeniden deneme; her denemede cevap payı yalnız küçülür.
            current, error = self.max_tokens, exc
            for _ in range(MAX_CONTEXT_RETRIES):
                reduced = _fitting_max_tokens(str(error), current)
                if reduced is None:
                    raise error
                current = reduced
                try:
                    return self._post({**body, "max_tokens": current})
                except InputError as again:
                    error = again
            raise error

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        encoded = json_text(body, indent=None).encode("utf-8")
        if len(encoded) > 150_000:
            raise InputError("Model isteği pilot boyut sınırını aşıyor.")
        headers = {"Content-Type": "application/json"}
        # Optional local service authentication: never read/print a secrets file.
        token = os.environ.get("ACTUARYLLM_LOCAL_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(self.base_url + "/chat/completions", data=encoded, headers=headers, method="POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = response.read(1_000_001)
        except HTTPError as exc:
            detail = _error_detail(exc)
            raise InputError(f"Yerel model HTTP hatası: {exc.code}; model/sunucu ayarlarını kontrol edin."
                             + (f" Sunucu: {detail}" if detail else "")) from exc
        except (URLError, TimeoutError, OSError) as exc:
            # Zaman aşımı ayrı bildirilir: sunucu çalışıyor ama yoğun (2026-11-17 eşzamanlı prova: 60 sn'de 29/40 soru düştü).
            if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
                raise InputError(f"Yerel model {self.timeout:g} saniyede yanıt vermedi (zaman aşımı); sunucu yoğun olabilir, "
                                 "biraz sonra tekrar deneyin.") from exc
            raise InputError("Yerel model sunucusuna erişilemedi; sunucuyu kendiniz başlatın. Model indirilmedi.") from exc
        if len(payload) > 1_000_000:
            raise InputError("Yerel model yanıtı çok büyük.")
        try:
            value = parse_json(payload.decode("utf-8"), origin="Yerel model yanıtı")
        except UnicodeDecodeError as exc:
            raise InputError("Yerel model yanıtı UTF-8 değil.") from exc
        if not isinstance(value, dict) or not isinstance(value.get("choices"), list) or not value["choices"]:
            raise InputError("Yerel sunucu chat-completions biçiminde yanıt vermedi.")
        message = value["choices"][0].get("message") if isinstance(value["choices"][0], dict) else None
        if not isinstance(message, dict):
            raise InputError("Yerel model message nesnesi eksik.")
        return {"message": message, "response_model": value.get("model"), "usage": value.get("usage")}


_CONTEXT_RE = re.compile(r"maximum context length is (\d+) tokens.*?prompt contains at least (\d+) input tokens", re.S)
MIN_RETRY_TOKENS = 512  # bundan az cevap payı kalıyorsa yeniden denemek anlamsız; hata olduğu gibi döner
CONTEXT_MARGIN = 128  # vLLM'in istem sayısı alt sınır olduğu için güvenlik payı
MAX_CONTEXT_RETRIES = 2


def _fitting_max_tokens(message: str, current: int) -> int | None:
    """vLLM bağlam aşımı mesajından sığan cevap payını hesaplar (CONTEXT_MARGIN pay); sığmıyorsa ya da tanınmazsa None."""
    m = _CONTEXT_RE.search(message)
    if not m:
        return None
    room = int(m.group(1)) - int(m.group(2)) - CONTEXT_MARGIN
    return room if MIN_RETRY_TOKENS <= room < current else None


def _error_detail(exc: HTTPError, limit: int = 300) -> str:
    """Hata gövdesinden kısa, tek satırlık açıklama (vLLM {"error": {"message": ...}} ya da düz metin)."""
    try:
        raw = exc.read(4096).decode("utf-8", "replace")
    except Exception:  # gövde okunamazsa yalnız kod raporlanır
        return ""
    text = raw
    try:
        value = parse_json(raw, origin="Yerel model hata gövdesi")
        error = value.get("error") if isinstance(value, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            text = error["message"]
        elif isinstance(value, dict) and isinstance(value.get("message"), str):
            text = value["message"]
    except Exception:
        pass
    text = " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())
    return text[:limit] + ("…" if len(text) > limit else "")
