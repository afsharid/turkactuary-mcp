"""Yargıtay uygulamasına göre iş göremezlik ve destekten yoksun kalma tazminatı hesapları (Decimal, yalnız stdlib).

Varsayılan değerler Yargıtay kararlarından çıkarıldı (data/caselaw, evidence/caselaw-method-2026-11-18) ve çıktıda karar
kimliğiyle gösterilir. Kararlarda kaynağı bulunamayan değerler zorunlu girdidir. Hesap bilirkişi raporu yerine geçmez.

Yöntem (özet):
- Bakiye ömür: TRH 2010, kaza/ölüm tarihindeki yaşa göre, kesirli yaşta doğrusal ara değer.
- Bilinen dönem (kaza → hesap tarihi): verilen dönem gelirleriyle, artırım ve iskonto olmadan.
- Bilinmeyen dönem (hesap tarihi → sonrası): bilinen son gelir her yıl artış oranıyla artırılıp iskonto oranıyla iskonto
  edilir (progresif rant; %10/%10). Aktif dönem 60 yaşa kadar, pasif dönem 60 yaştan bakiye ömür sonuna kadar.
- İndirim sırası: kusur oranı → hatır taşıması → müterafik kusur (ayrı ayrı, toplanmaz) → rücuya tabi SGK peşin sermaye
  değeri → güncellenmiş önceki ödemeler.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from actuaryllm.storage import InputError

YEAR_DAYS = Decimal("365")
AGE_DAYS = Decimal("365.25")
CENT = Decimal("0.01")
SOURCES = {
    "trh2010": "TRH 2010 (Yargıtay 4. HD: 1106451400, 1229028600; tablo doğrulaması 1172658100)",
    "kaza_yasi": "Bakiye ömür kaza tarihindeki yaşa göre (Yargıtay 4. HD 2024: 1229028600)",
    "progresif": "Progresif rant, %10 artış ve %10 iskonto; AYM 17.07.2020 2019/40 E. 2020/40 K. sonrası %1,8 uygulanmaz (4. HD: 1106451400, 895421400)",
    "bilinen": "Bilinen dönem iskontosuz, bilinen gelirlerle (4. HD: 1106451400; 21. HD: 149785000)",
    "aktif60": "Aktif dönem 60 yaşa kadar, sonra bakiye ömre kadar pasif dönem (4. HD: 1229028600; 21. HD: 149785000)",
    "indirim_sirasi": "Önce hatır taşıması, sonra müterafik kusur indirimi; oranlar toplanmaz (4. HD: 957691200)",
    "pay": "Destek 2, eş 2, her çocuk 1, ana ve baba 1'er pay; çocuksuz eşte %50–%50 (17. HD kararları, data/caselaw)",
    "cocuk_yas": "Çocuk desteği erkekte 18, kızda 22, yüksek öğrenimde 25 yaşa kadar (17. HD kararları, data/caselaw)",
    "ana_baba": "Çocuklar destekten çıkınca payları ana-babaya verilmez; ana-babadan biri çıkınca payı diğerine geçer (17. HD kararları, data/caselaw)",
    "sgk": "Yalnız rücuya tabi SGK gelirlerinin ilk peşin sermaye değeri düşülür (17. HD kararları, data/caselaw)",
}
_TABLE: dict[str, dict[int, Decimal]] | None = None


def _table(root: Path | None = None) -> dict[str, dict[int, Decimal]]:
    global _TABLE
    if _TABLE is None:
        base = root or Path(__file__).resolve().parents[2]
        data = json.loads((base / "data/tables/trh2010_bakiye_omur.json").read_text(encoding="utf-8"))
        _TABLE = {sex: {int(a): Decimal(v) for a, v in data[sex].items()} for sex in ("erkek", "kadin")}
    return _TABLE


def _d(value: Any, name: str, *, lo: str | None = None, hi: str | None = None) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise InputError(f"{name} sayı olmalı.")
    try:
        x = Decimal(str(value))
    except Exception as exc:
        raise InputError(f"{name} geçerli bir sayı değil.") from exc
    if not x.is_finite() or (lo is not None and x < Decimal(lo)) or (hi is not None and x > Decimal(hi)):
        raise InputError(f"{name} {lo}–{hi} aralığında olmalı.")
    return x


def _date(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise InputError(f"{name} YYYY-AA-GG biçiminde tarih olmalı.") from exc


def _sex(value: Any) -> str:
    if value not in ("erkek", "kadin"):
        raise InputError("cinsiyet 'erkek' ya da 'kadin' olmalı.")
    return value


def _age(birth: date, when: date) -> Decimal:
    return Decimal((when - birth).days) / AGE_DAYS


def bakiye_omur(yas: Any, cinsiyet: str) -> Decimal:
    """TRH 2010; kesirli yaşta doğrusal ara değer; 97 yaş üstünde 97 değeri."""
    a = _d(yas, "yas", lo="0", hi="120")
    t = _table()[_sex(cinsiyet)]
    if a >= 97:
        return t[97]
    lo = int(a)
    return t[lo] - (a - lo) * (t[lo] - t[lo + 1])


def _omur(args: dict[str, Any], yas: Decimal, sex: str) -> Decimal:
    """TRH 2010; taleple bağlılık (ör. PMF 1931 talebi) için bakiye_omur_yil elle verilebilir."""
    if args.get("bakiye_omur_yil") is not None:
        return _d(args["bakiye_omur_yil"], "bakiye_omur_yil", lo="0", hi="120")
    return bakiye_omur(yas, sex)


def _plus_years(d: date, years: Decimal) -> date:
    return d + timedelta(days=int((years * AGE_DAYS).to_integral_value(ROUND_HALF_UP)))


def _birthday(birth: date, age: int) -> date:
    try:
        return birth.replace(year=birth.year + age)
    except ValueError:  # 29 Şubat
        return birth.replace(year=birth.year + age, day=28)


def _money(x: Decimal) -> str:
    return str(x.quantize(CENT, ROUND_HALF_UP))


def _known_income(periods: Any, start: date, end: date) -> list[tuple[date, date, Decimal]]:
    """Bilinen dönem gelir dilimleri; [start, end) aralığını boşluksuz kaplamalı."""
    if not isinstance(periods, list) or not periods:
        raise InputError("gelir_donemleri boş olmayan liste olmalı: [{baslangic, bitis, aylik_net}].")
    rows = []
    for p in periods:
        if not isinstance(p, dict):
            raise InputError("gelir_donemleri öğeleri nesne olmalı.")
        rows.append((_date(p.get("baslangic"), "baslangic"), _date(p.get("bitis"), "bitis"),
                     _d(p.get("aylik_net"), "aylik_net", lo="0")))
    rows.sort()
    cur = start
    out = []
    for a, b, w in rows:
        if a > cur:
            raise InputError(f"gelir_donemleri {cur} tarihinden itibaren boşluk bırakıyor.")
        a2, b2 = max(a, cur), min(b, end)
        if b2 > a2:
            out.append((a2, b2, w))
            cur = b2
    if cur < end:
        raise InputError(f"gelir_donemleri {end} tarihine kadar uzanmalı (son: {cur}).")
    return out


def _future_segments(start: date, aktif_end: date, life_end: date, son: Decimal, pasif: Decimal | None,
                     artis: Decimal, iskonto: Decimal) -> list[dict[str, Any]]:
    """Hesap tarihinden ömür sonuna dilimler. t. yıl = hesap tarihinin (t-1). ve t. yıldönümü arası; katsayı ((1+a)/(1+i))^t.
    Aktif dönem sonu yılın içine düşerse o yıl iki dilime bölünür, ikisi de aynı t katsayısını alır."""
    out, ratio, t = [], (1 + artis) / (1 + iskonto), 1
    year_start = start
    while year_start < life_end:
        year_end = min(_birthday(start, t), life_end)
        points = [year_start] + ([aktif_end] if year_start < aktif_end < year_end else []) + [year_end]
        for a, b in zip(points, points[1:]):
            donem = "aktif" if a < aktif_end else "pasif"
            aylik = son if donem == "aktif" else pasif
            if aylik is None:
                continue
            full = a == year_start and b == _birthday(start, t)  # tam yıl dilimi: artık yılda da 1 yıl (Uzman B incelemesi)
            years = Decimal(1) if full else Decimal((b - a).days) / YEAR_DAYS
            out.append({"baslangic": a, "bitis": b, "donem": donem, "yil": years, "aylik": aylik,
                        "katsayi": ratio ** t, "tutar": aylik * 12 * years * ratio ** t})
        year_start, t = year_end, t + 1
    return out


def _reductions(gross: Decimal, kusur: Decimal, hatir: Decimal, muterafik: Decimal, sgk: Decimal, onceki: Decimal) -> list[dict]:
    steps, x = [{"adim": "gerçek zarar", "tutar": gross}], gross
    for name, rate in (("sorumlu tarafın kusur oranı", kusur), ("hatır taşıması indirimi", hatir),
                       ("müterafik kusur indirimi", muterafik)):
        if name.startswith("sorumlu"):
            x = x * rate
        elif rate:
            x = x * (1 - rate)
        steps.append({"adim": f"{name} ({rate})", "tutar": x})
    for name, amount in (("rücuya tabi SGK peşin sermaye değeri", sgk), ("güncellenmiş önceki ödemeler", onceki)):
        if amount:
            x = x - amount
            steps.append({"adim": f"− {name}", "tutar": x})
    if x < 0:
        x = Decimal(0)
        steps.append({"adim": "eksiye düşmez", "tutar": x})
    return steps


def _common(args: dict[str, Any]) -> dict[str, Decimal]:
    return {
        "artis": _d(args.get("artis_orani", "0.10"), "artis_orani", lo="0", hi="1"),
        "iskonto": _d(args.get("iskonto_orani", "0.10"), "iskonto_orani", lo="0", hi="1"),
        "aktif_yas": _d(args.get("aktif_donem_sonu_yasi", 60), "aktif_donem_sonu_yasi", lo="1", hi="100"),
        "kusur": _d(args.get("sorumlu_kusur_orani", 1), "sorumlu_kusur_orani", lo="0", hi="1"),
        "hatir": _d(args.get("hatir_indirimi", 0), "hatir_indirimi", lo="0", hi="1"),
        "muterafik": _d(args.get("muterafik_kusur_indirimi", 0), "muterafik_kusur_indirimi", lo="0", hi="1"),
    }


def is_goremezlik(**args: Any) -> dict[str, Any]:
    """Sürekli (ve isteğe bağlı geçici) iş göremezlik tazminatı."""
    c = _common(args)
    sex = _sex(args.get("cinsiyet"))
    birth = _date(args.get("dogum_tarihi"), "dogum_tarihi")
    kaza = _date(args.get("kaza_tarihi"), "kaza_tarihi")
    hesap = _date(args.get("hesap_tarihi"), "hesap_tarihi")
    if not birth < kaza <= hesap:
        raise InputError("Tarih sırası doğum < kaza ≤ hesap olmalı.")
    oran = _d(args.get("surekli_maluliyet_orani"), "surekli_maluliyet_orani", lo="0", hi="1")
    gecici = int(_d(args.get("gecici_is_goremezlik_gun", 0), "gecici_is_goremezlik_gun", lo="0", hi="36500"))
    son = _d(args.get("son_aylik_net"), "son_aylik_net", lo="0")
    if "pasif_aylik_net" not in args:
        raise InputError("pasif_aylik_net zorunlu (pasif dönem geliri; uygulamada çoğunlukla net asgari ücret).")
    pasif = _d(args.get("pasif_aylik_net"), "pasif_aylik_net", lo="0")
    yas = _age(birth, kaza)
    if yas < 18 and gecici and args.get("kaza_tarihinde_calisiyor") is not True:
        raise InputError("Kaza tarihinde 18 yaşından küçük ve çalışmayan kişi için geçici iş göremezlik zararı doğmaz (4. HD); "
                         "çalışıyorsa kaza_tarihinde_calisiyor=true verin.")
    omur = _omur(args, yas, sex)
    life_end = _plus_years(kaza, omur)
    aktif_end = _birthday(birth, int(c["aktif_yas"]))
    rows = []
    gecici_end = min(kaza + timedelta(days=gecici), hesap)
    for a, b, w in _known_income(args.get("gelir_donemleri"), kaza, hesap):
        for s, e, r, label in ((a, min(b, gecici_end), Decimal(1), "geçici iş göremezlik (%100)"),
                               (max(a, gecici_end), b, oran, "bilinen dönem")):
            if e > s:
                rows.append({"baslangic": s, "bitis": e, "donem": label, "aylik": w, "oran": r,
                             "tutar": w * 12 * Decimal((e - s).days) / YEAR_DAYS * r})
    future = _future_segments(hesap, aktif_end, life_end, son, pasif, c["artis"], c["iskonto"])
    for f in future:
        f["oran"] = oran
        f["tutar"] = f["tutar"] * oran
    gross = sum((r["tutar"] for r in rows + future), Decimal(0))
    sgk_oran = _d(args.get("sgk_mahsup_orani", 1), "sgk_mahsup_orani", lo="0", hi="1")
    steps = _reductions(gross, c["kusur"], c["hatir"], c["muterafik"],
                        _d(args.get("sgk_pesin_sermaye_degeri", 0), "sgk_pesin_sermaye_degeri", lo="0") * sgk_oran,
                        _d(args.get("onceki_odemeler_guncel", 0), "onceki_odemeler_guncel", lo="0"))
    return {
        "yontem": "Yargıtay uygulaması: TRH 2010 + progresif rant (bilirkişi raporu yerine geçmez)",
        "kaza_tarihindeki_yas": str(yas.quantize(CENT)), "bakiye_omur_yil": str(omur.quantize(CENT)),
        "omur_sonu": str(life_end), "aktif_donem_sonu": str(aktif_end),
        "bilinen_donem": [_fmt(r) for r in rows], "bilinmeyen_donem": [_fmt(f) for f in future],
        "ozet": {"bilinen_donem": _money(sum((r["tutar"] for r in rows), Decimal(0))),
                 "bilinmeyen_aktif": _money(sum((f["tutar"] for f in future if f["donem"] == "aktif"), Decimal(0))),
                 "bilinmeyen_pasif": _money(sum((f["tutar"] for f in future if f["donem"] == "pasif"), Decimal(0)))},
        "indirimler": [{"adim": s["adim"], "tutar": _money(s["tutar"])} for s in steps],
        "tazminat": _money(steps[-1]["tutar"]),
        "varsayimlar": _assumptions(c, ["trh2010", "kaza_yasi", "bilinen", "progresif", "aktif60", "indirim_sirasi", "sgk"]),
    }


def _fmt(r: dict[str, Any]) -> dict[str, Any]:
    out = {"baslangic": str(r["baslangic"]), "bitis": str(r["bitis"]), "donem": r["donem"], "aylik_net": str(r["aylik"]),
           "tutar": _money(r["tutar"])}
    if "katsayi" in r:
        out["progresif_katsayi"] = str(r["katsayi"].quantize(Decimal("0.000001")))
    if "oran" in r:
        out["oran"] = str(r["oran"])
    if "pay" in r:
        out["pay"] = str(r["pay"].quantize(Decimal("0.000001")))
    return out


def _assumptions(c: dict[str, Decimal], keys: list[str]) -> list[str]:
    out = [SOURCES[k] for k in keys]
    if c["artis"] != Decimal("0.10") or c["iskonto"] != Decimal("0.10"):
        out.append(f"Artış {c['artis']} ve iskonto {c['iskonto']} varsayılandan (%10/%10) farklı verildi.")
    if c["aktif_yas"] != 60:
        out.append(f"Aktif dönem sonu {c['aktif_yas']} yaş verildi (varsayılan 60).")
    out.append("Yıl 365 gün, yaş 365,25 gün kabul edildi; tutarlar kuruşa yuvarlandı.")
    return out


def destekten_yoksun_kalma(**args: Any) -> dict[str, Any]:
    """Destekten yoksun kalma tazminatı; hak sahibi başına."""
    c = _common(args)
    sex = _sex(args.get("destek_cinsiyet"))
    birth = _date(args.get("destek_dogum_tarihi"), "destek_dogum_tarihi")
    olum = _date(args.get("olum_tarihi"), "olum_tarihi")
    hesap = _date(args.get("hesap_tarihi"), "hesap_tarihi")
    if not birth < olum <= hesap:
        raise InputError("Tarih sırası doğum < ölüm ≤ hesap olmalı.")
    if "pasif_donem_destegi" not in args or not isinstance(args["pasif_donem_destegi"], bool):
        raise InputError("pasif_donem_destegi zorunlu (true/false): kararlarda destekten yoksun kalmada pasif dönem için açık kural bulunamadı.")
    son = _d(args.get("son_aylik_net"), "son_aylik_net", lo="0")
    pasif = _d(args.get("pasif_aylik_net"), "pasif_aylik_net", lo="0") if args["pasif_donem_destegi"] else None
    if args["pasif_donem_destegi"] and "pasif_aylik_net" not in args:
        raise InputError("pasif_donem_destegi true ise pasif_aylik_net zorunlu.")
    yas_olum = _age(birth, olum)
    omur = _omur(args, yas_olum, sex)
    life_end = _plus_years(olum, omur)
    gelir_bas = None
    if yas_olum < 18:
        if "destek_gelir_baslangic_yasi" not in args:
            raise InputError("Destek ölüm tarihinde 18 yaşından küçük: destek_gelir_baslangic_yasi zorunlu (4. HD: kazanç elde edeceği yaş, "
                             "kural olarak 18) ve yetistirme_gideri_indirimi verilebilir (4. HD uygulaması %5).")
        gelir_bas = _birthday(birth, int(_d(args["destek_gelir_baslangic_yasi"], "destek_gelir_baslangic_yasi", lo="0", hi="30")))
    aktif_end = _birthday(birth, int(c["aktif_yas"]))
    people = _beneficiaries(args.get("hak_sahipleri"), olum, life_end)
    income = [{"baslangic": a, "bitis": b, "donem": "bilinen dönem", "aylik": w, "katsayi": Decimal(1),
               "tutar": w * 12 * Decimal((b - a).days) / YEAR_DAYS} for a, b, w in _known_income(args.get("gelir_donemleri"), olum, hesap)]
    income += _future_segments(hesap, aktif_end, life_end, son, pasif, c["artis"], c["iskonto"])
    if gelir_bas:  # küçük destek: kazanç başlangıcından önceki dilimler düşer
        income = [{**x, "baslangic": max(x["baslangic"], gelir_bas),
                   "tutar": x["tutar"] * Decimal((x["bitis"] - max(x["baslangic"], gelir_bas)).days) / Decimal((x["bitis"] - x["baslangic"]).days)}
                  for x in income if x["bitis"] > gelir_bas]
    yetistirme = _d(args.get("yetistirme_gideri_indirimi", 0), "yetistirme_gideri_indirimi", lo="0", hi="1")
    # Ana-baba payı ölüm tarihindeki pay birimiyle sabitlenir; çocuklar çıkınca boşalan pay ana-babaya gitmez.
    # Ana-baba payı: ana-baba birimi / (o ana kadar katılmış herkesin birimi). Çocuk çıkınca payı artmaz (17. HD);
    # varsayımsal eş/çocuk eklendikçe azalır (4. HD: genç desteğin ileride evlenme ve çocuk sahibi olma ihtimali).
    def parents_total(at: date) -> Decimal:
        joined = [p for p in people if p["baslangic"] <= at]
        n_par = sum(1 for p in joined if p["tip"] in ("anne", "baba"))
        return Decimal(n_par) / Decimal(2 + sum(2 if p["tip"] == "es" else 1 for p in joined)) if n_par else Decimal(0)
    totals = {p["etiket"]: Decimal(0) for p in people}
    detail = {p["etiket"]: [] for p in people}
    for seg in _split(income, sorted({p["bitis"] for p in people} | {p["baslangic"] for p in people})):
        mid = seg["baslangic"]
        active = [p for p in people if p["baslangic"] <= mid < p["bitis"]]
        parents = [p for p in active if p["tip"] in ("anne", "baba")]
        pt = parents_total(mid) if parents else Decimal(0)
        rest = Decimal(1) - pt
        units = 2 + sum(2 if p["tip"] == "es" else 1 for p in active if p["tip"] not in ("anne", "baba"))
        for p in active:
            share = (pt / Decimal(len(parents)) if p["tip"] in ("anne", "baba")
                     else rest * Decimal(2 if p["tip"] == "es" else 1) / Decimal(units))
            amount = seg["tutar"] * share
            totals[p["etiket"]] += amount
            detail[p["etiket"]].append({**seg, "pay": share, "tutar": amount})
    sgk = args.get("sgk_pesin_sermaye_degeri") or {}
    onceki = args.get("onceki_odemeler_guncel") or {}
    if not isinstance(sgk, dict) or not isinstance(onceki, dict):
        raise InputError("sgk_pesin_sermaye_degeri ve onceki_odemeler_guncel {etiket: tutar} nesnesi olmalı.")
    results, grand = [], Decimal(0)
    for p in [q for q in people if not q["varsayimsal"]]:
        gross = totals[p["etiket"]]
        evl = p["evlenme"] if p["tip"] == "es" else Decimal(0)
        gross_after = gross * (1 - evl)
        if yetistirme and p["tip"] in ("anne", "baba"):
            gross_after = gross_after * (1 - yetistirme)
        steps = _reductions(gross_after, c["kusur"], c["hatir"], c["muterafik"],
                            _d(sgk.get(p["etiket"], 0), "sgk_pesin_sermaye_degeri", lo="0"),
                            _d(onceki.get(p["etiket"], 0), "onceki_odemeler_guncel", lo="0"))
        if evl or (yetistirme and p["tip"] in ("anne", "baba")):
            label = f"evlenme ihtimali indirimi ({evl})" if evl else f"yetiştirme gideri indirimi ({yetistirme})"
            steps.insert(1, {"adim": label, "tutar": gross_after})
            steps[0] = {"adim": "gerçek zarar (indirimsiz)", "tutar": gross}
        grand += steps[-1]["tutar"]
        results.append({"etiket": p["etiket"], "tip": p["tip"], "destek_bitis": str(p["bitis"]),
                        "dilimler": [_fmt(r) for r in detail[p["etiket"]]],
                        "indirimler": [{"adim": s["adim"], "tutar": _money(s["tutar"])} for s in steps],
                        "tazminat": _money(steps[-1]["tutar"])})
    keys = ["trh2010", "kaza_yasi", "bilinen", "progresif", "aktif60", "pay", "cocuk_yas", "ana_baba", "indirim_sirasi", "sgk"]
    notes = _assumptions(c, keys)
    notes.append("Pasif dönem desteği: " + ("var (girdi)" if args["pasif_donem_destegi"] else "yok (girdi)") + "; kararlarda açık kural bulunamadı.")
    notes.append("Eşin evlenme ihtimali oranı girdi olarak alındı (AYİM tablosu; 4. HD'ye göre rapor tarihindeki yaş, 18 yaş altı her çocuk için %5 indirim).")
    if any(p["varsayimsal"] for p in people):
        notes.append("Varsayımsal hak sahipleri (ileride evlenme/çocuk) paylara katıldı, tazminat almadı (4. HD: genç desteklerde azalan pay).")
    else:
        notes.append("Genç/bekar desteklerde 4. HD ana-baba payının yıllara göre azalmasını arar: varsayımsal eş/çocuk (varsayimsal=true, baslangic_tarihi) ekleyin.")
    return {"yontem": "Yargıtay uygulaması: TRH 2010 + progresif rant + pay sistemi (bilirkişi raporu yerine geçmez)",
            "olum_tarihindeki_yas": str(_age(birth, olum).quantize(CENT)), "bakiye_omur_yil": str(omur.quantize(CENT)),
            "omur_sonu": str(life_end), "aktif_donem_sonu": str(aktif_end), "hak_sahipleri": results,
            "toplam_tazminat": _money(grand), "varsayimlar": notes}


def _beneficiaries(items: Any, olum: date, life_end: date) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise InputError("hak_sahipleri boş olmayan liste olmalı.")
    out, labels = [], set()
    for it in items:
        if not isinstance(it, dict) or it.get("tip") not in ("es", "cocuk", "anne", "baba"):
            raise InputError("Her hak sahibi {etiket, tip: es|cocuk|anne|baba, cinsiyet, dogum_tarihi, ...} olmalı.")
        label = str(it.get("etiket") or f"{it['tip']}{len(out) + 1}")
        if label in labels:
            raise InputError(f"Tekrarlanan etiket: {label}")
        labels.add(label)
        sex = _sex(it.get("cinsiyet"))
        birth = _date(it.get("dogum_tarihi"), "dogum_tarihi")
        ref = _date(it["baslangic_tarihi"], "baslangic_tarihi") if it.get("baslangic_tarihi") else olum
        own_end = _plus_years(olum, bakiye_omur(max(_age(birth, olum), Decimal(0)), sex)) if birth < olum else _plus_years(ref, Decimal(80))
        if it["tip"] == "cocuk":
            default = 25 if it.get("yuksek_ogrenim") else (18 if sex == "erkek" else 22)
            limit = int(_d(it.get("destek_bitis_yasi", default), "destek_bitis_yasi", lo="0", hi="100"))
            end = min(_birthday(birth, limit), life_end)
        else:
            end = min(own_end, life_end)
        evl = _d(it.get("evlenme_ihtimali_orani", 0), "evlenme_ihtimali_orani", lo="0", hi="1")
        hyp = it.get("varsayimsal") is True
        start = _date(it["baslangic_tarihi"], "baslangic_tarihi") if it.get("baslangic_tarihi") else olum
        if hyp and start <= olum:
            raise InputError(f"{label}: varsayımsal hak sahibinin baslangic_tarihi ölüm tarihinden sonra olmalı.")
        if not hyp and start != olum:
            raise InputError(f"{label}: baslangic_tarihi yalnız varsayımsal hak sahiplerinde verilir.")
        out.append({"etiket": label, "tip": it["tip"], "baslangic": start, "bitis": max(end, start), "evlenme": evl, "varsayimsal": hyp})
    if sum(p["tip"] == "es" for p in out) > 1:  # varsayımsal eş de sayılır
        raise InputError("En çok bir eş olabilir.")
    return out


def _split(segments: list[dict[str, Any]], cuts: list[date]) -> list[dict[str, Any]]:
    """Gelir dilimlerini hak sahibi bitiş tarihlerinde böler (tutar gün oranıyla)."""
    out = []
    for s in segments:
        points = [s["baslangic"]] + [c for c in cuts if s["baslangic"] < c < s["bitis"]] + [s["bitis"]]
        total = Decimal((s["bitis"] - s["baslangic"]).days)
        for a, b in zip(points, points[1:]):
            out.append({**s, "baslangic": a, "bitis": b, "tutar": s["tutar"] * Decimal((b - a).days) / total})
    return out
