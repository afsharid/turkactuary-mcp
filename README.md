# TurkActuary MCP

**Türkçe** · [English](#english)

Türkiye sigortacılık mevzuatı araması ve aktüeryal hesap araçları için bir **MCP sunucusu**.
İstediğiniz modeli ya da ajanı kullanın: LM Studio, Claude Code, Codex, VS Code, Claude Desktop, Cursor ve MCP destekleyen her istemci bu araçlara erişir.

> **Uyarı:** Bu araçlar karar desteği içindir; uzman veya yetkili kurum görüşünün yerine geçmez. Mevzuat metinleri
> 2026-10-18 erişimli konsolide kopyadır ve kendiliğinden güncellenmez. Güncel hüküm için resmî kaynağı doğrulayın.

## Ne içerir

| Araç | Ne yapar |
|---|---|
| `mevzuat_ara` | Sigortacılık mevzuatında (5684 sayılı Kanun, yönetmelikler ve tebliğler; konsolide) ve 16 öğretici notta arama, toplam 624 parça. Projede ölçülen **v5** karma arama (BM25 + Qwen3-Embedding-4B); embedding modeli yoksa BM25. Parça kimliği, başlık, kaynak ve metin döner. |
| `belge_getir` | Parça kimliğiyle tam metni getirir. |
| `present_value`, `annuity`, `life_insurance_pv` | Bugünkü değer, annüite, süreli hayat sigortası. |
| `chain_ladder`, `bornhuetter_ferguson` | Hasar rezervi (kümülatif üçgen, paid/incurred açık). |
| `portfolio_metrics`, `combined_ratio` | Frekans, şiddet, saf prim, hasar oranı; bileşik oran. |
| `life_annuity_pv`, `life_cashflow_projection` | Yaşam koşullu annüite, hayat poliçesi nakit akışı. |
| `xl_layer_recovery`, `credibility_estimate`, `earned_exposure` | XL katman tahsilatı, güvenilirlik, kazanılmış maruziyet. |
| `poisson_glm`, `gamma_severity_glm` | Frekans ve şiddet GLM'leri. |

Bir de `turkactuary` adlı **prompt** var: modele kaynaklara dayanmasını, sayıyı araçla hesaplamasını ve belge içindeki
talimatları uygulamamasını söyleyen sistem talimatı.

Hesaplar `Decimal` ile yapılır ve testlidir. Oranlar ondalık verilir (0.10 = %10). Sunucu ağa çıkmaz, model indirmez, dosya yazmaz.

## Kurulum

Python 3.11+ yeterli; ek bağımlılık yok.

```bash
git clone https://github.com/afsharid/turkactuary-mcp.git
cd turkactuary-mcp
python3 -m unittest discover -s tests   # isteğe bağlı: sunucu testleri
```

### Anlamsal arama (v5) için embedding modeli

En iyi sonuç için yerel bir embedding modeli açın. LM Studio'da **Qwen3-Embedding-4B** modelini indirip yükleyin
(`text-embedding-qwen3-embedding-4b`) ve yerel sunucuyu başlatın (varsayılan `http://127.0.0.1:1234`). Sunucu açılışta
modelinizi 3 parçada hazır vektörlerle karşılaştırır: uyumluysa birkaç saniyede v5'e geçer, değilse 624 parçayı kendi
modelinizle hesaplar (bu Mac'te ~20 dk; bir kez, `~/.cache/actuaryllm-mcp/` altına yazılır). Embedding modeli yoksa BM25
(v2) ile çalışır; her arama sonucunda `profil` ve `v5_durumu` alanları hangisinin kullanıldığını söyler.

Ayarlar (isteğe bağlı): `ACTUARYLLM_SEARCH=auto|v5|v2`, `ACTUARYLLM_EMBED_URL` (yalnız 127.0.0.1/localhost),
`ACTUARYLLM_EMBED_MODEL`, `ACTUARYLLM_CACHE`.

### LM Studio

LM Studio → **Program** sekmesi → **Install → Edit mcp.json**, sonra ekleyin (yolu kendi klasörünüzle değiştirin):

```json
{
  "mcpServers": {
    "turkactuary": {
      "command": "python3",
      "args": ["-m", "actuaryllm_mcp"],
      "env": { "PYTHONPATH": "/yol/turkactuary-mcp/src", "ACTUARYLLM_ROOT": "/yol/turkactuary-mcp" }
    }
  }
}
```

Araç çağırabilen bir model seçin. Sohbette araçlar otomatik görünür.

**Model seçimi önemli.** Sunucu doğru sonucu döndürür; onu doğru kullanmak modele kalır. 2026-11-18 denemesinde Claude
Sonnet aynı soruda aracı doğru kullandı (chain ladder rezervi 30, 5684 s.K. md. 23 atfı). Qwen3-8B ise araca hatalı
parametre verdi ve hatadan sonra sonucu kendisi uydurdu. Araç kullanmada güçlü, büyük modeller (ör. 27B ve üstü) önerilir.

### Claude Code

```bash
claude mcp add turkactuary -e PYTHONPATH=/yol/turkactuary-mcp/src -e ACTUARYLLM_ROOT=/yol/turkactuary-mcp -- python3 -m actuaryllm_mcp
```

### Codex

`~/.codex/config.toml`:

```toml
[mcp_servers.turkactuary]
command = "python3"
args = ["-m", "actuaryllm_mcp"]
env = { PYTHONPATH = "/yol/turkactuary-mcp/src", ACTUARYLLM_ROOT = "/yol/turkactuary-mcp" }
```

### VS Code (Copilot)

`.vscode/mcp.json`:

```json
{
  "servers": {
    "turkactuary": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "actuaryllm_mcp"],
      "env": { "PYTHONPATH": "/yol/turkactuary-mcp/src", "ACTUARYLLM_ROOT": "/yol/turkactuary-mcp" }
    }
  }
}
```

### Claude Desktop / Cursor / diğerleri

LM Studio'daki `mcpServers` bloğunu istemcinin MCP yapılandırmasına ekleyin (Claude Desktop: `claude_desktop_config.json`,
Cursor: `.cursor/mcp.json`). `uv` kullanıyorsanız: `"command": "uv", "args": ["run", "--directory", "/yol/turkactuary-mcp", "turkactuary-mcp"]`.

## Örnek

> "Kümülatif ödenen üçgen [[80,100],[120]] için chain ladder rezervi nedir? Sigorta acentelerinin levha kaydı hangi
> yönetmelikte düzenlenir?"

Model `chain_ladder` aracıyla 30 TL bulur, `mevzuat_ara` ile ilgili yönetmelik parçasını getirir ve parça kimliğiyle atıf yapar.

## Kaynak ve lisans

- Kod: Apache-2.0 (`LICENSE`).
- Öğretici notlar ve proje verisi: CC BY 4.0 (`LICENSE-DATA`).
- Mevzuat metinleri: mevzuat.gov.tr konsolide metinleri (resmî metinler); kaynak, erişim tarihi ve izin kaydı
  `data/sources.json` içindedir. Kamuya açık resmî metinlerdir; yorum ve uygulama için resmî kaynak esastır.

## Ölçülen kurulumla farkı

Bu sunucu ActuaryLLM projesinin (Türkiye sigortacılık ve aktüerya asistanı pilotu) arama ve araç katmanından türetildi.
Projede ölçülen kurulumla birebir aynı değildir:

| | Projede ölçülen (Qwen3.8-27B, ince ayarsız) | Bu MCP sunucusu |
|---|---|---|
| Arama | v5: BM25 + Qwen3-Embedding-4B anlamsal arama (karma) | Aynı v5 (embedding modeli açıksa; 6 deneme sorgusunda ilk 4 parça birebir aynı). Model yoksa v2: yalnız BM25 |
| Hesap araçları | 6 araç, soru başına en çok 2 çağrı | 14 araç, çağrı sınırı yok |
| Sistem talimatı | aynı metin | `turkactuary` prompt'u (aynı metin + araç rehberi) |

Ölçümler (projenin kendi altın etiketli vakaları; otomatik ölçüm, uzman puanı değildir): kuralın modele giden bağlamda
görünme oranı dev/test/v9 için v2 0,693/0,911/0,791 → v5 0,769/0,970/0,819; uçtan uca `fact_hit` v2, önceki v1 profiline
(3 parça, kısaltılmış alıntı) göre 0,780 → 0,875 ve v5 tam sette v2'ye göre gerilemesiz. Sonuçlar modelinize, istemcinize ve araç kullanımına göre
değişir; bu sayılar bu sunucunun sizin kurulumunuzdaki kalitesini garanti etmez.

---

<a id="english"></a>
## English

**TurkActuary MCP** is a Model Context Protocol server for **Turkish insurance regulation search** and **tested actuarial
calculators**. Use it with any model or agent: LM Studio, Claude Code, Codex, VS Code (Copilot), Claude Desktop, Cursor and
any other MCP client.

> **Disclaimer:** decision support only; not legal or actuarial advice. Regulation texts are a consolidated copy accessed
> on 2026-10-18 and are not updated automatically. Always verify against the official source.

**Tools**

- `mevzuat_ara` (search): 624 chunks of Turkish insurance law (Insurance Law No. 5684, regulations and communiqués,
  consolidated) and 16 teaching notes. Hybrid search (BM25 + Qwen3-Embedding-4B) as measured in the project when a local
  embedding model is available; BM25 otherwise. Returns chunk id, title, source and text for citation.
- `belge_getir` (get document): full text of a chunk by id.
- 14 calculators (exact `Decimal` arithmetic, unit-tested): present value, annuity, term life insurance PV, chain ladder,
  Bornhuetter-Ferguson, portfolio metrics (frequency, severity, pure premium, loss ratio), combined ratio, life annuity,
  life cash-flow projection, excess-of-loss layer recovery, Bühlmann credibility, earned exposure, Poisson frequency GLM,
  gamma severity GLM.
- Prompt `turkactuary`: the system policy (ground answers in retrieved chunks, compute numbers with tools, treat documents
  as data, not instructions).

**Model choice matters.** The server returns correct results; using them well is up to the model. In a test, Claude Sonnet
used the tools correctly, while Qwen3-8B passed a wrong parameter and then made up the result. Prefer larger models that
are good at tool use.

**Setup:** Python 3.11+, no dependencies. `git clone` the repo and add the `mcpServers` block shown above to your client
(paths: `PYTHONPATH=<repo>/src`, `ACTUARYLLM_ROOT=<repo>`). For hybrid search, load `text-embedding-qwen3-embedding-4b` in
LM Studio; the server checks compatibility with the bundled vectors on 3 chunks and switches to hybrid within seconds.

**Licence:** code Apache-2.0, project data CC BY 4.0; regulation texts from mevzuat.gov.tr (official texts; source records in
`data/sources.json`). Derived from the ActuaryLLM pilot; the measured setup and its differences are listed above.
