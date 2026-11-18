"""actuaryllm_mcp: stdio JSON-RPC üzerinden el sıkışma, araç listesi, arama ve hesap (ağsız)."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def session(*messages, env_extra=None, delay=0.0):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "ACTUARYLLM_SEARCH": "v2", **(env_extra or {})}
    if delay:  # ısınma iş parçacığı bitsin diye ilk mesajdan sonra bekle
        proc = subprocess.Popen([sys.executable, "-m", "actuaryllm_mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT)
        proc.stdin.write(json.dumps(messages[0]) + "\n"); proc.stdin.flush()
        time.sleep(delay)
        rest = "\n".join(json.dumps(m) for m in messages[1:]) + "\n"
        stdout, stderr = proc.communicate(rest, timeout=120)
        assert proc.returncode == 0, stderr
        return {r["id"]: r for r in (json.loads(l) for l in stdout.splitlines())}
    stdin = "\n".join(json.dumps(m) for m in messages) + "\n"
    out = subprocess.run([sys.executable, "-m", "actuaryllm_mcp"], input=stdin, capture_output=True, text=True, env=env,
                         timeout=120, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return {r["id"]: r for r in (json.loads(l) for l in out.stdout.splitlines())}


class McpServerTest(unittest.TestCase):
    def test_handshake_list_search_calculate(self):
        r = session(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "mevzuat_ara", "arguments": {"sorgu": "zorunlu deprem sigortası", "adet": 3}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "chain_ladder", "arguments": {"triangle": [["80", "100"], ["120"]], "basis": "paid"}}},
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "present_value", "arguments": {"cashflows": [{"period": 1, "amount": "110"}], "rate": "0.10"}}},
            {"jsonrpc": "2.0", "id": 6, "method": "prompts/get", "params": {"name": "turkactuary"}},
        )
        self.assertEqual(r[1]["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(len(r), 6)  # bildirime cevap yok
        names = {t["name"] for t in r[2]["result"]["tools"]}
        self.assertTrue({"mevzuat_ara", "belge_getir", "chain_ladder", "bornhuetter_ferguson", "poisson_glm"} <= names)
        self.assertEqual(len(names), 16)
        hits = json.loads(r[3]["result"]["content"][0]["text"])["parcalar"]
        self.assertEqual(len(hits), 3)
        self.assertTrue(all(h["id"] and h["excerpt"] for h in hits))
        self.assertEqual(json.loads(r[4]["result"]["content"][0]["text"])["total_reserve"], "30")
        self.assertEqual(json.loads(r[5]["result"]["content"][0]["text"])["present_value"], "100")
        self.assertIn("uzman veya yetkili kurum yerine geçmez", r[6]["result"]["messages"][0]["content"]["text"])

    def test_errors_are_reported_not_crashing(self):
        r = session(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "exec", "arguments": {"code": "x"}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "belge_getir", "arguments": {"kimlik": "yok"}}},
            {"jsonrpc": "2.0", "id": 3, "method": "bilinmeyen"},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "present_value", "arguments": {"cashflows": [], "rate": "NaN"}}},
        )
        self.assertTrue(r[1]["result"]["isError"])
        self.assertTrue(r[2]["result"]["isError"])
        self.assertEqual(r[3]["error"]["code"], -32601)
        self.assertTrue(r[4]["result"]["isError"])

    def test_get_document_by_id(self):
        r = session({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "belge_getir", "arguments": {"kimlik": "edu_risk_pool"}}})
        doc = json.loads(r[1]["result"]["content"][0]["text"])
        self.assertEqual(doc["id"], "edu_risk_pool")
        self.assertIn("sentetik", doc["excerpt"])


class ArticleLabelTest(unittest.TestCase):
    """Her resmî parçada kapsanan madde numarası (küçük modeller atıfta madde uydurmasın; 2026-11-18)."""

    def test_article_map_covers_all_official_chunks(self):
        sys.path.insert(0, str(ROOT / "src"))
        from actuaryllm.retrieval import load_corpus
        from actuaryllm_mcp.server import article_map
        docs = load_corpus(ROOT).documents
        m = article_map(docs)
        self.assertEqual(m["mevzuat-5684_consolidated:25"], ["Madde 23"])  # önceki parçada başlayan madde taşınır
        self.assertEqual(m["mevzuat-5684_consolidated:24"], ["Madde 22", "Madde 23"])
        self.assertIn("Genel Şartlar A.3", m["mevzuat-teb_14982:1"])  # genel şartlarda bölüm numarası
        self.assertIn("Geçici Madde 1", m["mevzuat-teb_20752:17"])  # "GEÇİCİ\n\nMADDE 1" ve doğru Türkçe yazım
        self.assertEqual([d.cite_id for d in docs if d.kind == "official" and not m.get(d.cite_id)], [])
        self.assertNotIn("edu_risk_pool", m)

    def test_search_and_get_include_article_numbers(self):
        r = session(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "belge_getir", "arguments": {"kimlik": "mevzuat-5684_consolidated:25"}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "mevzuat_ara", "arguments": {"sorgu": "sigorta acentelerinin levhaya kaydı", "adet": 3}}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "belge_getir", "arguments": {"kimlik": "edu_risk_pool"}}},
        )
        doc = json.loads(r[1]["result"]["content"][0]["text"])
        self.assertEqual(doc["maddeler"], ["Madde 23"])
        self.assertTrue(doc["excerpt"].startswith("(Bu parça: Madde 23)\n"))
        hits = json.loads(r[2]["result"]["content"][0]["text"])["parcalar"]
        self.assertTrue(all(h["maddeler"] for h in hits if h["id"].startswith("mevzuat-")))
        self.assertNotIn("maddeler", json.loads(r[3]["result"]["content"][0]["text"]))


class FakeEmbeddings(BaseHTTPRequestHandler):
    """Metnin özetinden belirlenimli 8 boyutlu vektör döner (gerçek model değildir)."""
    calls = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeEmbeddings.calls += 1
        data = [{"index": i, "embedding": [b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]]}
                for i, t in enumerate(body["input"])]
        payload = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class McpV5Test(unittest.TestCase):
    SEARCH = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "mevzuat_ara", "arguments": {"sorgu": "zorunlu deprem sigortası", "adet": 3}}}
    INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}

    def test_v5_with_local_embedding_server_and_cache(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEmbeddings)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as cache:
                env = {"ACTUARYLLM_SEARCH": "auto", "ACTUARYLLM_CACHE": cache, "ACTUARYLLM_EMBED_MODEL": "fake-embed",
                       "ACTUARYLLM_EMBED_URL": f"http://127.0.0.1:{server.server_port}/v1"}
                r = session(self.INIT, self.SEARCH, env_extra=env, delay=8)
                out = json.loads(r[2]["result"]["content"][0]["text"])
                self.assertEqual(out["profil"], "v5", out["v5_durumu"])
                self.assertEqual(len(out["parcalar"]), 3)
                self.assertEqual(len(list(Path(cache).glob("vectors-*.json"))), 1)
                before = FakeEmbeddings.calls
                r = session(self.INIT, self.SEARCH, env_extra=env, delay=3)  # önbellekten: belge vektörü yeniden hesaplanmaz
                self.assertEqual(json.loads(r[2]["result"]["content"][0]["text"])["profil"], "v5")
                self.assertEqual(FakeEmbeddings.calls - before, 1)  # yalnız sorgu vektörü
        finally:
            server.shutdown()

    def test_auto_falls_back_to_v2_without_embedding_server(self):
        with tempfile.TemporaryDirectory() as cache:
            env = {"ACTUARYLLM_SEARCH": "auto", "ACTUARYLLM_CACHE": cache, "ACTUARYLLM_EMBED_URL": "http://127.0.0.1:9/v1"}
            r = session(self.INIT, self.SEARCH, env_extra=env, delay=2)
            out = json.loads(r[2]["result"]["content"][0]["text"])
            self.assertEqual(out["profil"], "v2")
            self.assertIn("kullanılamıyor", out["v5_durumu"])


if __name__ == "__main__":
    unittest.main()
