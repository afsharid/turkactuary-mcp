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
        self.assertEqual(len(names), 18)
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


def poll_search(env_extra, done, timeout=60):
    """Sunucuyu başlatır, `done(sonuç)` doğru olana kadar her saniye arar (ısınma zamanlamasına bağımlı olmamak için)."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **env_extra}
    proc = subprocess.Popen([sys.executable, "-m", "actuaryllm_mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT)
    try:
        deadline, i, out = time.time() + timeout, 0, None
        while time.time() < deadline:
            i += 1
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {
                "name": "mevzuat_ara", "arguments": {"sorgu": "zorunlu deprem sigortası", "adet": 3}}}) + "\n")
            proc.stdin.flush()
            out = json.loads(json.loads(proc.stdout.readline())["result"]["content"][0]["text"])
            if done(out):
                break
            time.sleep(1)
        return out
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
        proc.stdout.close()
        proc.stderr.close()


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
                # auto: hazır vektörlerle uyumsuz model (ya da hazır vektör yok) → tam hesap yok, v2 ve nedeni
                before = FakeEmbeddings.calls
                out = poll_search(env, lambda o: o["v5_durumu"].startswith("kapalı"))
                self.assertEqual(out["profil"], "v2")
                self.assertIn("ACTUARYLLM_SEARCH=v5", out["v5_durumu"])
                self.assertLessEqual(FakeEmbeddings.calls - before, 1)  # yalnız 3 parçalık uyum yoklaması
                self.assertEqual(list(Path(cache).glob("vectors-*.json")), [])
                # v5 açıkça istenince bütün parçalar hesaplanır ve önbelleğe yazılır
                out = poll_search({**env, "ACTUARYLLM_SEARCH": "v5"}, lambda o: o.get("profil") == "v5")
                self.assertEqual(out["profil"], "v5", out)
                self.assertEqual(len(out["parcalar"]), 3)
                self.assertEqual(len(list(Path(cache).glob("vectors-*.json"))), 1)
                before = FakeEmbeddings.calls
                out = poll_search(env, lambda o: o["profil"] == "v5")  # auto önbelleği kullanır, yeniden hesaplamaz
                self.assertEqual(out["profil"], "v5")
                self.assertLessEqual(FakeEmbeddings.calls - before, 3)  # yalnız sorgu vektörleri (yoklama sayısı kadar)
        finally:
            server.shutdown()
            server.server_close()

    def test_auto_falls_back_to_v2_without_embedding_server(self):
        with tempfile.TemporaryDirectory() as cache:
            env = {"ACTUARYLLM_SEARCH": "auto", "ACTUARYLLM_CACHE": cache, "ACTUARYLLM_EMBED_URL": "http://127.0.0.1:9/v1"}
            out = poll_search(env, lambda o: "kullanılamıyor" in o["v5_durumu"])
            self.assertEqual(out["profil"], "v2")
            self.assertIn("kullanılamıyor", out["v5_durumu"])


class SurfaceTest(unittest.TestCase):
    def test_all_tools_are_annotated_read_only(self):
        r = session({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = r[1]["result"]["tools"]
        self.assertEqual(len(tools), 18)
        for t in tools:
            self.assertEqual(t["annotations"], {"readOnlyHint": True, "destructiveHint": False,
                                                "idempotentHint": True, "openWorldHint": False}, t["name"])

    def test_no_annotations_for_2024_11_05_clients(self):
        r = session({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual(r[1]["result"]["protocolVersion"], "2024-11-05")
        self.assertFalse(any("annotations" in t for t in r[2]["result"]["tools"]))

    def test_seed_from_other_corpus_is_reported_not_crashing(self):
        import gzip
        sys.path.insert(0, str(ROOT / "src"))
        from actuaryllm_mcp.server import SEED_VECTORS, Server

        class Client:
            calls = 0

            def embed(self, texts):
                Client.calls += 1
                return [[1.0, 0.0] for _ in texts]
        s = Server(ROOT)
        docs = s.index.corpus.documents
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / SEED_VECTORS
            seed.parent.mkdir(parents=True)
            seed.write_bytes(gzip.compress(json.dumps({"model": "m", "vectors": {"baska": [1.0, 0.0]}}).encode()))
            s.root = Path(tmp)
            vectors, note = s._seed(Client(), docs, "m")
        self.assertEqual((vectors, note), ({}, ": hazır vektörler bu korpusla eşleşmiyor"))
        self.assertEqual(Client.calls, 1)  # yalnız erişim yoklaması

    def test_turkish_case_folding_in_search(self):
        sys.path.insert(0, str(ROOT / "src"))
        from actuaryllm.retrieval import tokens
        self.assertEqual(tokens("İSTANBUL IĞDIR SİGORTA ısı"), ["istanbul", "ığdır", "sigorta", "ısı"])
        ask = lambda i, q: {"jsonrpc": "2.0", "id": i, "method": "tools/call",
                            "params": {"name": "mevzuat_ara", "arguments": {"sorgu": q, "adet": 3}}}
        r = session(ask(1, "zorunlu deprem sigortası"), ask(2, "ZORUNLU DEPREM SİGORTASI"))
        ids = [[h["id"] for h in json.loads(r[i]["result"]["content"][0]["text"])["parcalar"]] for i in (1, 2)]
        self.assertEqual(ids[0], ids[1])


def _public_pyproject() -> Path | None:
    """Açık paketin pyproject'i: özel depoda şablon, açık depoda kökteki dosya."""
    for p in (ROOT / "scripts/mcp_public_pyproject.toml", ROOT / "pyproject.toml"):
        if p.is_file() and 'name = "turkactuary-mcp"' in p.read_text(encoding="utf-8"):
            return p
    return None


def _public_readme() -> Path:
    p = ROOT / "scripts/mcp_public_README.md"
    return p if p.is_file() else ROOT / "README.md"


class PackagingTest(unittest.TestCase):
    """Kurulu tekerlek (wheel) düzeni: veri paket içinde, depo yok. Sürüm ve README sayıları tek kaynaktan."""

    @classmethod
    def setUpClass(cls):
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))

    def test_version_has_single_source(self):
        import re
        from actuaryllm_mcp import __version__
        from actuaryllm_mcp.server import SERVER_VERSION
        self.assertEqual(SERVER_VERSION, __version__)
        pyproject = _public_pyproject().read_text(encoding="utf-8")
        self.assertIn('dynamic = ["version"]', pyproject)
        self.assertIn('path = "src/actuaryllm_mcp/__init__.py"', pyproject)
        self.assertIsNone(re.search(r"(?m)^version\s*=", pyproject))

    def test_registry_metadata_matches_package(self):
        from actuaryllm_mcp import __version__
        path = next(p for p in (ROOT / "scripts/mcp_public_server.json", ROOT / "server.json") if p.is_file())
        server = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual({server["version"], *(p["version"] for p in server["packages"])}, {__version__})
        self.assertEqual([(p["registryType"], p["identifier"]) for p in server["packages"]], [("pypi", "turkactuary-mcp")])
        self.assertLessEqual(len(server["description"]), 100)  # MCP Registry sınırı
        self.assertIn(f"<!-- mcp-name: {server['name']} -->", _public_readme().read_text(encoding="utf-8"))

    def test_readme_counts_match_corpus_and_tools(self):
        from actuaryllm.retrieval import load_corpus
        from actuaryllm_mcp.server import ALL_TOOLS, COMPENSATION_TOOLS, Server
        n = len(load_corpus(ROOT, files=Server(ROOT).corpus_files()).documents)
        text = _public_readme().read_text(encoding="utf-8")
        self.assertIn(f"toplam {n} parça", text)
        self.assertIn(f"{n} chunks", text)
        self.assertIn(f"{len(ALL_TOOLS) + len(COMPENSATION_TOOLS)} araç, çağrı sınırı yok", text)

    def test_installed_wheel_layout_runs_search_and_compensation(self):
        import shutil
        import tomllib
        pyproject = _public_pyproject()
        if pyproject is None or pyproject.parent != ROOT:
            self.skipTest("yalnız açık depoda (özel depoda artifacts/ çok büyük; dışa aktarılan kopyada koşar)")
        conf = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["hatch"]["build"]["targets"]["wheel"]
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site-packages"
            for pkg in conf["packages"]:  # hatchling: src/<paket> → <paket>
                shutil.copytree(ROOT / pkg, site / Path(pkg).name, ignore=shutil.ignore_patterns("__pycache__"))
            for src, dest in conf["force-include"].items():
                shutil.copytree(ROOT / src, site / dest)
            msgs = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "mevzuat_ara", "arguments": {"sorgu": "zorunlu deprem sigortası", "adet": 2}}},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "is_goremezlik", "arguments": dict(
                    cinsiyet="erkek", dogum_tarihi="1990-01-01", kaza_tarihi="2024-01-01", hesap_tarihi="2026-01-01",
                    surekli_maluliyet_orani="0.10", son_aylik_net="22000", pasif_aylik_net="22000",
                    gelir_donemleri=[{"baslangic": "2024-01-01", "bitis": "2026-01-01", "aylik_net": "20000"}])}},
                {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "destekten_yoksun_kalma", "arguments": dict(
                    destek_cinsiyet="erkek", destek_dogum_tarihi="1990-01-01", olum_tarihi="2024-01-01", hesap_tarihi="2025-01-01",
                    gelir_donemleri=[{"baslangic": "2024-01-01", "bitis": "2025-01-01", "aylik_net": "20000"}],
                    son_aylik_net="20000", pasif_donem_destegi=False,
                    hak_sahipleri=[{"etiket": "es", "tip": "es", "cinsiyet": "kadin", "dogum_tarihi": "1992-01-01"}])}},
            ]
            env = {k: v for k, v in os.environ.items() if k not in ("ACTUARYLLM_ROOT", "PYTHONPATH")}
            env.update(PYTHONPATH=str(site), ACTUARYLLM_SEARCH="v2", ACTUARYLLM_CACHE=str(Path(tmp) / "cache"))
            out = subprocess.run([sys.executable, "-m", "actuaryllm_mcp"], input="\n".join(map(json.dumps, msgs)) + "\n",
                                 capture_output=True, text=True, env=env, cwd=tmp, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr)
            r = {x["id"]: x for x in map(json.loads, out.stdout.splitlines())}
            for i in (2, 3, 4):
                self.assertFalse(r[i]["result"].get("isError"), r[i]["result"]["content"][0]["text"][:300])
            self.assertEqual(len(json.loads(r[2]["result"]["content"][0]["text"])["parcalar"]), 2)


if __name__ == "__main__":
    unittest.main()
