"""actuaryllm_mcp.tazminat: Yargıtay uygulamasına göre iş göremezlik ve destekten yoksun kalma (elle hesaplanan beklentiler)."""
import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from actuaryllm.storage import InputError  # noqa: E402
from actuaryllm_mcp.tazminat import bakiye_omur, destekten_yoksun_kalma, is_goremezlik  # noqa: E402

D = Decimal


class TableTest(unittest.TestCase):
    def test_trh2010_interpolation_matches_yargitay_report(self):
        # 4. HD 2022/8265 E. (1172658100): 29,86 yaşında kadın, bakiye ömür 49,14 yıl
        self.assertEqual(bakiye_omur("29.86", "kadin").quantize(D("0.01")), D("49.14"))
        self.assertEqual(bakiye_omur(34, "erkek"), D("40.62"))
        self.assertEqual(bakiye_omur(110, "erkek"), D("1.23"))


class IsGoremezlikTest(unittest.TestCase):
    BASE = dict(cinsiyet="erkek", dogum_tarihi="1990-01-01", kaza_tarihi="2024-01-01", hesap_tarihi="2026-01-01",
                surekli_maluliyet_orani="0.10", son_aylik_net="22000", pasif_aylik_net="22000",
                gelir_donemleri=[{"baslangic": "2024-01-01", "bitis": "2026-01-01", "aylik_net": "20000"}])

    def test_periods_match_hand_calculation(self):
        r = is_goremezlik(**self.BASE)
        days = lambda a, b: D((date.fromisoformat(b) - date.fromisoformat(a)).days)
        known = D(20000) * 12 * days("2024-01-01", "2026-01-01") / 365 * D("0.10")
        aktif = D(22000) * 12 * 24 * D("0.10")  # 24 tam yıl (artık yıllar dahil 1 yıl), %10/%10 → katsayı 1
        self.assertEqual(r["bakiye_omur_yil"], "40.62")
        self.assertEqual(r["aktif_donem_sonu"], "2050-01-01")
        self.assertEqual(r["ozet"]["bilinen_donem"], str(known.quantize(D("0.01"))))
        self.assertEqual(r["ozet"]["bilinmeyen_aktif"], str(aktif.quantize(D("0.01"))))
        self.assertTrue(any("1229028600" in v for v in r["varsayimlar"]))

    def test_reductions_are_sequential_not_summed(self):
        r = is_goremezlik(**{**self.BASE, "sorumlu_kusur_orani": "0.5", "hatir_indirimi": "0.2", "muterafik_kusur_indirimi": "0.2"})
        gross = D(r["indirimler"][0]["tutar"])
        self.assertEqual(D(r["tazminat"]), (gross * D("0.5") * D("0.8") * D("0.8")).quantize(D("0.01")))

    def test_temporary_disability_at_full_rate(self):
        r = is_goremezlik(**{**self.BASE, "gecici_is_goremezlik_gun": 90})
        first = r["bilinen_donem"][0]
        self.assertEqual((first["donem"], first["oran"], first["bitis"]), ("geçici iş göremezlik (%100)", "1", "2024-03-31"))

    def test_progressive_factor_when_rates_differ(self):
        r = is_goremezlik(**{**self.BASE, "artis_orani": "0.05", "iskonto_orani": "0.10"})
        k = [D(x["progresif_katsayi"]) for x in r["bilinmeyen_donem"][:2]]
        self.assertEqual(k[0], (D("1.05") / D("1.10")).quantize(D("0.000001")))
        self.assertLess(k[1], k[0])

    def test_manual_life_expectancy_and_sgk_rate(self):
        r = is_goremezlik(**{**self.BASE, "bakiye_omur_yil": "30", "sgk_pesin_sermaye_degeri": "10000", "sgk_mahsup_orani": "0.5"})
        self.assertEqual(r["bakiye_omur_yil"], "30.00")
        self.assertEqual(D(r["indirimler"][-2]["tutar"]) - D(r["indirimler"][-1]["tutar"]), D("5000.00"))

    def test_minor_not_working_no_temporary_loss(self):
        args = {**self.BASE, "dogum_tarihi": "2010-01-01", "gecici_is_goremezlik_gun": 30}
        with self.assertRaises(InputError):
            is_goremezlik(**args)
        is_goremezlik(**{**args, "kaza_tarihinde_calisiyor": True})

    def test_invalid_inputs(self):
        bad = {k: v for k, v in self.BASE.items() if k != "pasif_aylik_net"}
        for args in (bad, {**self.BASE, "surekli_maluliyet_orani": "1.5"}, {**self.BASE, "hesap_tarihi": "2023-01-01"},
                     {**self.BASE, "gelir_donemleri": [{"baslangic": "2024-02-01", "bitis": "2026-01-01", "aylik_net": "1"}]}):
            with self.assertRaises(InputError):
                is_goremezlik(**args)


class DestekTest(unittest.TestCase):
    BASE = dict(destek_cinsiyet="erkek", destek_dogum_tarihi="1990-01-01", olum_tarihi="2024-01-01", hesap_tarihi="2025-01-01",
                gelir_donemleri=[{"baslangic": "2024-01-01", "bitis": "2025-01-01", "aylik_net": "20000"}],
                son_aylik_net="20000", pasif_donem_destegi=False,
                hak_sahipleri=[{"etiket": "es", "tip": "es", "cinsiyet": "kadin", "dogum_tarihi": "1992-01-01"},
                               {"etiket": "cocuk", "tip": "cocuk", "cinsiyet": "erkek", "dogum_tarihi": "2020-01-01"}])

    def test_shares_change_when_child_leaves(self):
        r = destekten_yoksun_kalma(**self.BASE)
        es = {p["etiket"]: p for p in r["hak_sahipleri"]}
        self.assertEqual(es["cocuk"]["destek_bitis"], "2038-01-01")  # erkek çocuk 18 yaş
        shares = {(x["baslangic"] < "2038-01-01", D(x["pay"])) for x in es["es"]["dilimler"]}
        self.assertEqual(shares, {(True, D("0.4")), (False, D("0.5"))})  # 2/5 sonra 2/4
        self.assertTrue(all(D(x["pay"]) == D("0.2") for x in es["cocuk"]["dilimler"]))
        self.assertTrue(all(x["bitis"] <= r["aktif_donem_sonu"] for x in es["es"]["dilimler"]))  # pasif destek yok

    def test_parent_share_frozen_and_girl_until_22(self):
        args = {**self.BASE, "hak_sahipleri": self.BASE["hak_sahipleri"][:1] + [
            {"etiket": "kiz", "tip": "cocuk", "cinsiyet": "kadin", "dogum_tarihi": "2010-01-01"},
            {"etiket": "anne", "tip": "anne", "cinsiyet": "kadin", "dogum_tarihi": "1965-01-01"}]}
        r = destekten_yoksun_kalma(**args)
        p = {x["etiket"]: x for x in r["hak_sahipleri"]}
        self.assertEqual(p["kiz"]["destek_bitis"], "2032-01-01")
        self.assertEqual({x["pay"] for x in p["anne"]["dilimler"]}, {str((D(1) / 6).quantize(D("0.000001")))})  # 6 birim, sabit
        es_after = {x["pay"] for x in p["es"]["dilimler"] if "2032-01-01" <= x["baslangic"] < p["anne"]["destek_bitis"]}
        self.assertEqual(es_after, {str(((1 - D(1) / 6) * 2 / 4).quantize(D("0.000001")))})  # kız çıkınca pay ana-babaya gitmez

    def test_parent_share_passes_to_other_parent(self):
        args = {**self.BASE, "hak_sahipleri": self.BASE["hak_sahipleri"][:1] + [
            {"etiket": "anne", "tip": "anne", "cinsiyet": "kadin", "dogum_tarihi": "1960-01-01"},
            {"etiket": "baba", "tip": "baba", "cinsiyet": "erkek", "dogum_tarihi": "1955-01-01"}]}
        r = destekten_yoksun_kalma(**args)
        p = {x["etiket"]: x for x in r["hak_sahipleri"]}
        before = {x["pay"] for x in p["anne"]["dilimler"] if x["baslangic"] < p["baba"]["destek_bitis"]}
        after = {x["pay"] for x in p["anne"]["dilimler"] if x["baslangic"] >= p["baba"]["destek_bitis"]}
        self.assertEqual(before, {str((D(1) / 6).quantize(D("0.000001")))})
        self.assertEqual(after, {str((D(2) / 6).quantize(D("0.000001")))})  # babanın payı anneye geçer

    def test_hypothetical_spouse_reduces_parent_share(self):
        # 4. HD: genç/bekar desteğin ileride evlenme ihtimali → ana-baba payı yıllara göre azalır
        args = {**self.BASE, "destek_dogum_tarihi": "2000-01-01", "hak_sahipleri": [
            {"etiket": "anne", "tip": "anne", "cinsiyet": "kadin", "dogum_tarihi": "1975-01-01"},
            {"etiket": "es_v", "tip": "es", "cinsiyet": "kadin", "dogum_tarihi": "2002-01-01", "varsayimsal": True, "baslangic_tarihi": "2027-01-01"}]}
        r = destekten_yoksun_kalma(**args)
        self.assertEqual([p["etiket"] for p in r["hak_sahipleri"]], ["anne"])  # varsayımsal eş tazminat almaz
        pays = {(x["baslangic"] < "2027-01-01", x["pay"]) for x in r["hak_sahipleri"][0]["dilimler"]}
        self.assertEqual(pays, {(True, str((D(1) / 3).quantize(D("0.000001")))), (False, str((D(1) / 5).quantize(D("0.000001"))))})

    def test_minor_deceased_requires_income_start(self):
        args = {**self.BASE, "destek_dogum_tarihi": "2019-01-01", "hak_sahipleri": [
            {"etiket": "anne", "tip": "anne", "cinsiyet": "kadin", "dogum_tarihi": "1990-01-01"}]}
        with self.assertRaises(InputError):
            destekten_yoksun_kalma(**args)
        r = destekten_yoksun_kalma(**{**args, "destek_gelir_baslangic_yasi": 18, "yetistirme_gideri_indirimi": "0.05"})
        self.assertTrue(all(x["baslangic"] >= "2037-01-01" for x in r["hak_sahipleri"][0]["dilimler"]))
        self.assertIn("yetiştirme gideri", r["hak_sahipleri"][0]["indirimler"][1]["adim"])

    def test_marriage_reduction_and_required_inputs(self):
        r = destekten_yoksun_kalma(**{**self.BASE, "hak_sahipleri": [{**self.BASE["hak_sahipleri"][0], "evlenme_ihtimali_orani": "0.2"}]})
        steps = r["hak_sahipleri"][0]["indirimler"]
        self.assertLessEqual(abs(D(steps[1]["tutar"]) - D(steps[0]["tutar"]) * D("0.8")), D("0.01"))
        with self.assertRaises(InputError):
            destekten_yoksun_kalma(**{k: v for k, v in self.BASE.items() if k != "pasif_donem_destegi"})
        with self.assertRaises(InputError):
            destekten_yoksun_kalma(**{**self.BASE, "pasif_donem_destegi": True})  # pasif_aylik_net eksik


if __name__ == "__main__":
    unittest.main()
