"""Deterministic actuarial tools. Outputs preserve decimal precision.

Assumptions are explicit; these educational estimators are not a certified
valuation engine. The triangle pilot accepts nonnegative, nondecreasing,
square cumulative triangles, not incremental or recovery-heavy data.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from decimal import Decimal, DecimalException, Underflow, localcontext
from functools import wraps
from typing import Any, Callable

from actuaryllm.storage import InputError

_DECIMAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def number(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise InputError(f"{name}: sayı veya ondalık string bekleniyor; bool kabul edilmez.")
    text = str(value).strip()
    if len(text) > 100 or not _DECIMAL.fullmatch(text):
        raise InputError(f"{name}: sonlu, noktalı ondalık sayı bekleniyor (ör. 0.10).")
    try:
        result = Decimal(text)
    except DecimalException as exc:
        raise InputError(f"{name}: geçersiz ondalık sayı.") from exc
    if not result.is_finite() or abs(result.as_tuple().exponent) > 100 or len(result.as_tuple().digits) > 60:
        raise InputError(f"{name}: sayı desteklenen hassasiyet aralığının dışında.")
    return result


def integer(value: Any, name: str, minimum: int = 0, maximum: int = 10_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise InputError(f"{name}: {minimum}–{maximum} arasında tam sayı bekleniyor.")
    return value


def _rate(value: Any) -> Decimal:
    rate = number(value, "rate")
    if rate <= -1:
        raise InputError("Dönemlik iskonto oranı -1'den büyük olmalı.")
    return rate


def _string(value: Decimal) -> str:
    if not value.is_finite():
        raise InputError("Hesap sonucu sonlu değil.")
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _math(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            with localcontext() as context:
                context.prec = 36
                context.Emax = 9999
                context.Emin = -9999
                context.traps[Underflow] = True
                return function(*args, **kwargs)
        except DecimalException as exc:
            raise InputError("Girdiler desteklenen hesaplama aralığının dışında.") from exc
    return wrapped


@_math
def present_value(cashflows: list[dict[str, Any]], rate: Any) -> dict[str, Any]:
    """Net present value; periods are integers and rate is per period."""
    r = _rate(rate)
    if not isinstance(cashflows, list) or not 1 <= len(cashflows) <= 10_000:
        raise InputError("cashflows: boş olmayan, en fazla 10000 öğeli liste bekleniyor.")
    total = Decimal(0)
    for flow in cashflows:
        if not isinstance(flow, dict) or set(flow) != {"period", "amount"}:
            raise InputError("Her nakit akışı yalnız period ve amount içermeli.")
        period = integer(flow["period"], "period")
        total += number(flow["amount"], "amount") / ((1 + r) ** period)
    return {"present_value": _string(total), "period_rate": _string(r)}


@_math
def annuity(payment: Any, rate: Any, periods: int, timing: str = "immediate") -> dict[str, Any]:
    """Immediate: payments at 1..n; due: payments at 0..n-1."""
    n = integer(periods, "periods", 1)
    if timing not in {"immediate", "due"}:
        raise InputError("timing yalnız immediate veya due olabilir.")
    amount = number(payment, "payment")
    r = _rate(rate)
    start = 1 if timing == "immediate" else 0
    # One discounting rule also avoids cancellation for near-zero nonzero rates.
    value = present_value(
        [{"period": period, "amount": amount} for period in range(start, start + n)], r,
    )
    return {**value, "periods": n, "timing": timing}


@_math
def life_insurance_pv(qx: list[Any], benefit: Any, rate: Any) -> dict[str, Any]:
    """Term assurance benefit paid at the end of the death period.

    qx is supplied by the caller; no mortality table or demographic calibration
    is inferred. Terminal survival is not paid as an endowment.
    """
    if not isinstance(qx, list) or not 1 <= len(qx) <= 200:
        raise InputError("qx: 1–200 dönemlik olasılık listesi bekleniyor.")
    amount = number(benefit, "benefit")
    if amount < 0:
        raise InputError("benefit negatif olamaz.")
    r = _rate(rate)
    survival = Decimal(1)
    value = Decimal(0)
    for period, item in enumerate(qx, 1):
        probability = number(item, f"qx[{period - 1}]")
        if not 0 <= probability <= 1:
            raise InputError("qx değerleri 0–1 aralığında olmalı (yüzde değil olasılık).")
        value += survival * probability * amount / ((1 + r) ** period)
        survival *= 1 - probability
    return {
        "present_value": _string(value),
        "terminal_survival_probability": _string(survival),
        "benefit_timing": "death_period_end",
    }


def _triangle(value: Any) -> list[list[Decimal]]:
    if not isinstance(value, list) or not 2 <= len(value) <= 100:
        raise InputError("triangle: 2–100 satırlı kümülatif üçgen bekleniyor.")
    n = len(value)
    triangle: list[list[Decimal]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) not in {n - index, n}:
            raise InputError("Üçgen kare veya kısaltılmış satır biçiminde olmalı.")
        expected = n - index
        if len(raw) == n and any(item is not None for item in raw[expected:]):
            raise InputError("Üçgenin gözlenmeyen hücreleri yalnız null olabilir.")
        observed = [number(item, f"triangle[{index}]") for item in raw[:expected]]
        if any(item < 0 for item in observed):
            raise InputError("Pilot üçgen aracı negatif kümülatif hasarı desteklemiyor.")
        if any(right < left for left, right in zip(observed, observed[1:])):
            raise InputError("Pilot kümülatif üçgen azalamaz; recovery/düzeltme verisi ayrıca incelenmeli.")
        triangle.append(observed)
    return triangle


def _development(triangle: list[list[Decimal]], tail_factor: Any) -> tuple[list[Decimal], list[Decimal]]:
    n = len(triangle)
    tail = number(tail_factor, "tail_factor")
    if tail < 1:
        raise InputError("tail_factor en az 1 olmalı; 1 son yaş sonrası gelişim yok varsayımıdır.")
    factors = []
    for age in range(n - 1):
        rows = triangle[: n - age - 1]
        denominator = sum((row[age] for row in rows), Decimal(0))
        if denominator == 0:
            raise InputError(f"Gelişim yaşı {age}: faktör paydası sıfır; otomatik tahmin yapılmadı.")
        factors.append(sum((row[age + 1] for row in rows), Decimal(0)) / denominator)
    cdf = [tail for _ in range(n)]
    for age in range(n - 2, -1, -1):
        cdf[age] = factors[age] * cdf[age + 1]
    return factors, cdf


def _basis(value: str) -> str:
    if value not in {"paid", "incurred"}:
        raise InputError("basis açıkça paid veya incurred olmalı.")
    return "total_unpaid" if value == "paid" else "development_above_reported"


@_math
def chain_ladder(triangle: Any, basis: str, tail_factor: Any = "1") -> dict[str, Any]:
    """Volume-weighted deterministic cumulative Chain Ladder."""
    interpretation = _basis(basis)
    rows = _triangle(triangle)
    factors, cdf = _development(rows, tail_factor)
    ultimates = [row[-1] * cdf[len(row) - 1] for row in rows]
    reserves = [ultimate - row[-1] for ultimate, row in zip(ultimates, rows)]
    return {
        "method": "chain_ladder",
        "basis": basis,
        "reserve_interpretation": interpretation,
        "development_factors": [_string(item) for item in factors],
        "age_to_ultimate": [_string(item) for item in cdf],
        "ultimate_by_origin": [_string(item) for item in ultimates],
        "reserve_by_origin": [_string(item) for item in reserves],
        "total_ultimate": _string(sum(ultimates, Decimal(0))),
        "total_reserve": _string(sum(reserves, Decimal(0))),
        "tail_factor": _string(number(tail_factor, "tail_factor")),
    }


@_math
def bornhuetter_ferguson(
    triangle: Any,
    basis: str,
    earned_premiums: list[Any],
    expected_loss_ratios: list[Any],
    tail_factor: Any = "1",
) -> dict[str, Any]:
    """BF with explicitly supplied earned premiums and a-priori loss ratios."""
    interpretation = _basis(basis)
    rows = _triangle(triangle)
    factors, cdf = _development(rows, tail_factor)
    n = len(rows)
    if not isinstance(earned_premiums, list) or not isinstance(expected_loss_ratios, list):
        raise InputError("earned_premiums ve expected_loss_ratios liste olmalı.")
    if len(earned_premiums) != n or len(expected_loss_ratios) != n:
        raise InputError("Her origin satırı için kazanılmış prim ve beklenen hasar oranı gerekli.")
    premiums = [number(value, "earned_premiums") for value in earned_premiums]
    ratios = [number(value, "expected_loss_ratios") for value in expected_loss_ratios]
    if any(value < 0 for value in premiums + ratios):
        raise InputError("Kazanılmış prim/beklenen hasar oranı negatif olamaz.")
    expected = [premium * ratio for premium, ratio in zip(premiums, ratios)]
    development = [cdf[len(row) - 1] for row in rows]
    unemerged = [factor - 1 for factor in development]
    fractions = [numerator / factor for numerator, factor in zip(unemerged, development)]
    # Multiply before division: avoid cancellation and rounding an intermediate fraction.
    reserves = [ultimate * numerator / factor for ultimate, numerator, factor in zip(expected, unemerged, development)]
    ultimates = [row[-1] + reserve for row, reserve in zip(rows, reserves)]
    return {
        "method": "bornhuetter_ferguson",
        "basis": basis,
        "reserve_interpretation": interpretation,
        "development_factors": [_string(item) for item in factors],
        "age_to_ultimate": [_string(item) for item in cdf],
        "expected_ultimate_by_origin": [_string(item) for item in expected],
        "unemerged_fraction_by_origin": [_string(item) for item in fractions],
        "ultimate_by_origin": [_string(item) for item in ultimates],
        "reserve_by_origin": [_string(item) for item in reserves],
        "total_ultimate": _string(sum(ultimates, Decimal(0))),
        "total_reserve": _string(sum(reserves, Decimal(0))),
        "tail_factor": _string(number(tail_factor, "tail_factor")),
    }


@_math
def portfolio_metrics(exposure: Any, claim_count: int, total_loss: Any, earned_premium: Any) -> dict[str, Any]:
    """Closed-cohort loss metrics. Exposure units must be consistent."""
    e = number(exposure, "exposure")
    count = integer(claim_count, "claim_count", maximum=1_000_000_000)
    loss = number(total_loss, "total_loss")
    premium = number(earned_premium, "earned_premium")
    if e <= 0 or premium <= 0 or loss < 0:
        raise InputError("Exposure ve kazanılmış prim pozitif; toplam hasar negatif olmayan tutar olmalı.")
    if count == 0 and loss != 0:
        raise InputError("Kapalı kohortta sıfır hasar adediyle pozitif hasar tutarı çelişiyor.")
    return {
        "frequency": _string(Decimal(count) / e),
        "average_severity": _string(loss / count) if count else None,
        "pure_premium": _string(loss / e),
        "loss_ratio": _string(loss / premium),
    }


@_math
def life_annuity_pv(qx: list[Any], payment: Any, rate: Any, timing: str = "immediate") -> dict[str, Any]:
    """Temporary life annuity on caller-supplied qx: a payment is made only while the annuitant is alive.

    immediate: payment at the end of period t if alive at t (t=1..n); due: at the start of period t if alive
    at t-1 (t=1..n, first payment certain). No mortality table, expenses or guarantee period are inferred.
    """
    if not isinstance(qx, list) or not 1 <= len(qx) <= 200:
        raise InputError("qx: 1–200 dönemlik olasılık listesi bekleniyor.")
    if timing not in {"immediate", "due"}:
        raise InputError("timing yalnız immediate veya due olabilir.")
    amount = number(payment, "payment")
    r = _rate(rate)
    survival, value, expected_payments = Decimal(1), Decimal(0), Decimal(0)
    for period, item in enumerate(qx, 1):
        probability = number(item, f"qx[{period - 1}]")
        if not 0 <= probability <= 1:
            raise InputError("qx değerleri 0–1 aralığında olmalı (yüzde değil olasılık).")
        if timing == "due":
            value += survival * amount / ((1 + r) ** (period - 1))
            expected_payments += survival
        survival *= 1 - probability
        if timing == "immediate":
            value += survival * amount / ((1 + r) ** period)
            expected_payments += survival
    return {"present_value": _string(value), "expected_payment_count": _string(expected_payments),
            "terminal_survival_probability": _string(survival), "timing": timing}


@_math
def xl_layer_recovery(losses: list[Any], retention: Any, limit: Any, reinstatements: int | None = None) -> dict[str, Any]:
    """Per-occurrence excess-of-loss layer: recovery = min(max(loss - retention, 0), limit) per loss.

    With ``reinstatements`` the annual aggregate cover is limit * (1 + reinstatements); reinstatement premium,
    event definitions and hours clauses are not modelled. Losses must be ground-up per event, gross of this layer.
    """
    if not isinstance(losses, list) or not 1 <= len(losses) <= 10_000:
        raise InputError("losses: 1–10000 olay tutarı listesi bekleniyor.")
    lower, width = number(retention, "retention"), number(limit, "limit")
    if lower < 0 or width <= 0:
        raise InputError("retention negatif olamaz; limit pozitif olmalı.")
    aggregate = None if reinstatements is None else width * (1 + integer(reinstatements, "reinstatements", 0, 100))
    per_loss, remaining = [], aggregate
    for index, item in enumerate(losses):
        loss = number(item, f"losses[{index}]")
        if loss < 0:
            raise InputError("Olay tutarları negatif olamaz.")
        recovery = min(max(loss - lower, Decimal(0)), width)
        if remaining is not None:
            recovery = min(recovery, remaining)
            remaining -= recovery
        per_loss.append(recovery)
    gross = sum((number(x, "losses") for x in losses), Decimal(0))
    ceded = sum(per_loss, Decimal(0))
    return {"recovery_by_loss": [_string(x) for x in per_loss], "total_recovery": _string(ceded),
            "gross_losses": _string(gross), "net_retained": _string(gross - ceded),
            "aggregate_cover": _string(aggregate) if aggregate is not None else None,
            "aggregate_remaining": _string(remaining) if remaining is not None else None}


@_math
def credibility_estimate(observed_mean: Any, exposure_units: Any, prior_mean: Any, k: Any) -> dict[str, Any]:
    """Bühlmann-style credibility: Z = n / (n + k); estimate = Z * observed + (1 - Z) * prior.

    k (expected process variance / variance of hypothetical means) is supplied by the caller; it is not
    estimated here. Units of n and k must match (e.g. policy-years).
    """
    n, kk = number(exposure_units, "exposure_units"), number(k, "k")
    if n < 0 or kk <= 0:
        raise InputError("exposure_units negatif olamaz; k pozitif olmalı.")
    observed, prior = number(observed_mean, "observed_mean"), number(prior_mean, "prior_mean")
    z = n / (n + kk)
    return {"credibility_factor": _string(z), "estimate": _string(z * observed + (1 - z) * prior),
            "observed_mean": _string(observed), "prior_mean": _string(prior)}


@_math
def earned_exposure(policies: list[dict[str, Any]], period_start: str, period_end: str, days_per_year: int) -> dict[str, Any]:
    """Exposure earned inside [period_start, period_end) from policy [start, end) ISO dates, in years of days_per_year.

    Each policy contributes the days it overlaps the period; cancellations are expressed by an earlier end date.
    """
    from datetime import date

    def day(value: Any, name: str) -> date:
        if not isinstance(value, str):
            raise InputError(f"{name}: YYYY-MM-DD biçiminde tarih bekleniyor.")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise InputError(f"{name}: geçersiz tarih.") from exc

    begin, finish = day(period_start, "period_start"), day(period_end, "period_end")
    if finish <= begin:
        raise InputError("period_end, period_start'tan sonra olmalı.")
    base = integer(days_per_year, "days_per_year", 360, 366)
    if not isinstance(policies, list) or not 1 <= len(policies) <= 100_000:
        raise InputError("policies: 1–100000 poliçe listesi bekleniyor.")
    total, per_policy = 0, []
    for index, item in enumerate(policies):
        if not isinstance(item, dict) or set(item) != {"start", "end"}:
            raise InputError("Her poliçe yalnız start ve end içermeli.")
        start, end = day(item["start"], f"policies[{index}].start"), day(item["end"], f"policies[{index}].end")
        if end <= start:
            raise InputError(f"policies[{index}]: end, start'tan sonra olmalı.")
        days = max((min(end, finish) - max(start, begin)).days, 0)
        per_policy.append(days)
        total += days
    return {"earned_exposure_years": _string(Decimal(total) / base), "earned_days": total,
            "days_by_policy": per_policy, "days_per_year": base}


@_math
def combined_ratio(incurred_losses: Any, loss_adjustment_expenses: Any, acquisition_costs: Any,
                   other_expenses: Any, earned_premium: Any) -> dict[str, Any]:
    """Loss, expense and combined ratios on one consistent earned basis; components must share scope and period."""
    premium = number(earned_premium, "earned_premium")
    if premium <= 0:
        raise InputError("earned_premium pozitif olmalı.")
    parts = {name: number(value, name) for name, value in (
        ("incurred_losses", incurred_losses), ("loss_adjustment_expenses", loss_adjustment_expenses),
        ("acquisition_costs", acquisition_costs), ("other_expenses", other_expenses))}
    if any(value < 0 for value in parts.values()):
        raise InputError("Hasar ve gider kalemleri negatif olamaz.")
    loss_ratio = (parts["incurred_losses"] + parts["loss_adjustment_expenses"]) / premium
    expense_ratio = (parts["acquisition_costs"] + parts["other_expenses"]) / premium
    return {"loss_ratio_incl_lae": _string(loss_ratio), "expense_ratio": _string(expense_ratio),
            "combined_ratio": _string(loss_ratio + expense_ratio)}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: Callable[..., dict[str, Any]]
    parameters: dict[str, Any]

    def api_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}


@_math
def life_cashflow_projection(qx: list[Any], rate: Any, premium: Any, death_benefit: Any, lapse: list[Any] | None = None,
                             maturity_benefit: Any = "0", expense_per_policy: Any = "0",
                             surrender_value: list[Any] | None = None) -> dict[str, Any]:
    """Tek poliçe için dönemlik beklenen nakit akışı projeksiyonu (ölüm + iptal çıkışlı) ve bugünkü değerler.

    Zamanlama: prim ve poliçe başı gider dönem başında yürürlükteki poliçe için; ölüm teminatı dönem sonunda; iptal
    (lapse) ölümlerden sonra dönem sonunda, varsa o dönemin `surrender_value` tutarıyla; vade teminatı son dönem sonunda
    kalanlara. qx ve lapse çağıran tarafından verilir (tablo çıkarılmaz). net_present_value = BD(çıkışlar) − BD(primler):
    pozitifse sözleşme başlangıçta prim yetersizliği (ihtiyat) gösterir. Vergi, komisyon yapısı ve yatırım getirisi
    ayrıştırması modellenmez.
    """
    if not isinstance(qx, list) or not 1 <= len(qx) <= 200:
        raise InputError("qx: 1–200 dönemlik olasılık listesi bekleniyor.")
    n = len(qx)
    def series(values: list[Any] | None, name: str) -> list[Decimal]:
        if values is None:
            return [Decimal(0)] * n
        if not isinstance(values, list) or len(values) != n:
            raise InputError(f"{name}: qx ile aynı uzunlukta liste bekleniyor.")
        return [number(v, f"{name}[{i}]") for i, v in enumerate(values)]
    q, w, sv = series(qx, "qx"), series(lapse, "lapse"), series(surrender_value, "surrender_value")
    if any(not 0 <= v <= 1 for v in q + w):
        raise InputError("qx ve lapse değerleri 0–1 aralığında olasılık olmalı (yüzde değil).")
    if any(v < 0 for v in sv):
        raise InputError("surrender_value negatif olamaz.")
    r = _rate(rate)
    amounts = {k: number(v, k) for k, v in (("premium", premium), ("death_benefit", death_benefit),
                                            ("maturity_benefit", maturity_benefit), ("expense_per_policy", expense_per_policy))}
    if any(v < 0 for v in amounts.values()):
        raise InputError("premium, death_benefit, maturity_benefit ve expense_per_policy negatif olamaz.")
    in_force = Decimal(1)
    pv = {"premiums": Decimal(0), "expenses": Decimal(0), "death_benefits": Decimal(0), "surrenders": Decimal(0),
          "maturity": Decimal(0)}
    table = []
    for t in range(n):
        v_start, v_end = (1 + r) ** -t, (1 + r) ** -(t + 1)
        deaths = in_force * q[t]
        lapses = (in_force - deaths) * w[t]
        row = {"period": t + 1, "in_force_start": in_force, "premium": in_force * amounts["premium"],
               "expense": in_force * amounts["expense_per_policy"], "deaths": deaths,
               "death_outgo": deaths * amounts["death_benefit"], "lapses": lapses, "surrender_outgo": lapses * sv[t]}
        pv["premiums"] += row["premium"] * v_start
        pv["expenses"] += row["expense"] * v_start
        pv["death_benefits"] += row["death_outgo"] * v_end
        pv["surrenders"] += row["surrender_outgo"] * v_end
        in_force = in_force - deaths - lapses
        if t == n - 1:
            row["maturity_outgo"] = in_force * amounts["maturity_benefit"]
            pv["maturity"] += row["maturity_outgo"] * v_end
        table.append({k: (_string(v) if isinstance(v, Decimal) else v) for k, v in row.items()})
    outgo = pv["expenses"] + pv["death_benefits"] + pv["surrenders"] + pv["maturity"]
    return {
        "periods": table,
        "present_values": {k: _string(v) for k, v in pv.items()},
        "net_present_value": _string(outgo - pv["premiums"]),
        "terminal_in_force": _string(in_force),
        "timing": "premium/expense period start; death, lapse, maturity period end",
    }


_GLM_MAX_ROWS, _GLM_MAX_PARAMS, _GLM_MAX_ITER = 2_000, 30, 50
_FACTOR_NAME = re.compile(r"^[A-Za-z0-9_çğıöşüÇĞİÖŞÜ.-]{1,40}$")


def _sig(value: Decimal, digits: int = 12) -> str:
    """Anlamlı hane sayısıyla yuvarlar; çok küçük pozitif değerler 0'a düşmez."""
    if value == 0:
        return "0"
    return _string(value.quantize(Decimal(1).scaleb(value.adjusted() - digits + 1)))


def _solve(a: list[list[Decimal]], b: list[Decimal]) -> list[Decimal]:
    """Gauss eliminasyonu (kısmi pivotlama). Tekil matris: tasarım matrisi tam ranklı değil."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < Decimal("1e-30"):
            raise InputError("GLM: tasarım matrisi tekil (bir seviye tüm satırlarda aynı ya da faktörler çakışıyor).")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f:
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    x = [Decimal(0)] * n
    for i in reversed(range(n)):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


@_math
def poisson_glm(rows: list[dict[str, Any]], factors: list[str], base_levels: dict[str, str] | None = None) -> dict[str, Any]:
    """Log bağlantılı Poisson frekans GLM'i, log(maruziyet) ofsetiyle; kategorik faktörler, IRLS.

    Her satır: {"levels": {faktör: seviye}, "exposure": >0, "claims": tam sayı ≥ 0}. Taban seviye verilmezse alfabetik
    ilk seviyedir. Çıktı: katsayılar, göreli katsayılar (exp β; taban = 1), taban frekans, sapma (deviance) ve serbestlik
    derecesi. Hasarı olmayan seviye tahmin edilemez (β → −∞); birleştirilmesi istenir. Eğitim amaçlı tahmincidir;
    standart hata, etkileşim ve aşırı yayılım (overdispersion) düzeltmesi yoktur.
    """
    if not isinstance(rows, list) or not 1 <= len(rows) <= _GLM_MAX_ROWS:
        raise InputError(f"rows: 1–{_GLM_MAX_ROWS} satırlık liste bekleniyor.")
    if not isinstance(factors, list) or not 1 <= len(factors) <= 5 \
            or any(not isinstance(f, str) or not _FACTOR_NAME.fullmatch(f) for f in factors) or len(set(factors)) != len(factors):
        raise InputError("factors: 1–5 farklı faktör adı bekleniyor.")
    base_levels = {} if base_levels is None else base_levels
    if not isinstance(base_levels, dict) or set(base_levels) - set(factors):
        raise InputError("base_levels yalnız factors içindeki faktörleri içerebilir.")
    data = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"levels", "exposure", "claims"} or not isinstance(row["levels"], dict):
            raise InputError(f"rows[{i}]: levels, exposure, claims alanları bekleniyor.")
        if missing := [f for f in factors if f not in row["levels"]]:
            raise InputError(f"rows[{i}]: levels içinde eksik faktör: {', '.join(missing)} (fazlası yok sayılır).")
        levels = {}
        for f in factors:
            v = row["levels"][f]
            if not isinstance(v, str) or not _FACTOR_NAME.fullmatch(v):
                raise InputError(f"rows[{i}].levels.{f}: 1–40 karakterlik seviye adı bekleniyor.")
            levels[f] = v
        e = number(row["exposure"], f"rows[{i}].exposure")
        if e <= 0:
            raise InputError(f"rows[{i}].exposure pozitif olmalı.")
        data.append((levels, e, Decimal(integer(row["claims"], f"rows[{i}].claims", maximum=1_000_000_000))))
    columns, bases = ["(intercept)"], {}
    for f in factors:
        seen = sorted({lv[f] for lv, _, _ in data})
        if len(seen) < 2:
            raise InputError(f"{f}: en az iki seviye gerekli.")
        base = base_levels.get(f, seen[0])
        if base not in seen:
            raise InputError(f"{f}: taban seviye '{base}' verilerde yok.")
        bases[f] = base
        for level in seen:
            claims = sum(y for lv, _, y in data if lv[f] == level)
            if claims == 0:
                raise InputError(f"{f}={level}: hasar yok; katsayı tahmin edilemez, seviyeyi komşusuyla birleştirin.")
            if level != base:
                columns.append(f"{f}={level}")
    p = len(columns)
    if p > _GLM_MAX_PARAMS or p >= len(data) + 1:
        raise InputError(f"GLM: parametre sayısı ({p}) satır sayısından küçük ve en çok {_GLM_MAX_PARAMS} olmalı.")
    x = [[Decimal(1)] + [Decimal(int(lv[c.split("=", 1)[0]] == c.split("=", 1)[1])) for c in columns[1:]] for lv, _, _ in data]
    offset = [e.ln() for _, e, _ in data]
    y = [claims for _, _, claims in data]
    beta = [(sum(y) / sum(e for _, e, _ in data)).ln()] + [Decimal(0)] * (p - 1)
    converged, iterations = False, 0
    for iterations in range(1, _GLM_MAX_ITER + 1):
        eta = [sum(xi * b for xi, b in zip(row, beta)) + off for row, off in zip(x, offset)]
        mu = [v.exp() for v in eta]
        z = [et - off + (yi - m) / m for et, off, yi, m in zip(eta, offset, y, mu)]
        xtwx = [[sum(x[k][i] * mu[k] * x[k][j] for k in range(len(x))) for j in range(p)] for i in range(p)]
        xtwz = [sum(x[k][i] * mu[k] * z[k] for k in range(len(x))) for i in range(p)]
        new = _solve(xtwx, xtwz)
        if max(abs(a - b) for a, b in zip(new, beta)) < Decimal("1e-12"):
            beta, converged = new, True
            break
        beta = new
    if not converged:
        raise InputError(f"GLM: {_GLM_MAX_ITER} yinelemede yakınsamadı.")
    mu = [(sum(xi * b for xi, b in zip(row, beta)) + off).exp() for row, off in zip(x, offset)]
    deviance = 2 * sum((yi * (yi / m).ln() if yi else Decimal(0)) - (yi - m) for yi, m in zip(y, mu))
    q = Decimal("1e-10")
    return {
        "base_levels": bases,
        "coefficients": {c: _string(b.quantize(q)) for c, b in zip(columns, beta)},
        "relativities": {c: _sig(b.exp()) for c, b in zip(columns[1:], beta[1:])},
        "base_frequency": _sig(beta[0].exp()),
        "deviance": _string(deviance.quantize(q)) if abs(deviance) >= q else "0",
        "degrees_of_freedom": len(data) - p,
        "iterations": iterations,
        "notes": "Log bağlantılı Poisson, log(maruziyet) ofseti; standart hata ve aşırı yayılım düzeltmesi yok.",
    }


@_math
def gamma_severity_glm(rows: list[dict[str, Any]], factors: list[str],
                       base_levels: dict[str, str] | None = None) -> dict[str, Any]:
    """Log bağlantılı gamma şiddet (ortalama hasar) GLM'i; hasar adediyle ağırlıklı, kategorik faktörler, IRLS.

    Her satır: {"levels": {faktör: seviye}, "claims": tam sayı ≥ 1, "total_loss": > 0}; yanıt ortalama hasar
    (total_loss / claims), ağırlık hasar adedi. Çıktı Poisson GLM'iyle aynı biçimde: katsayılar, göreli katsayılar,
    taban şiddet, sapma ve serbestlik derecesi. Saf prim göreli katsayısı = frekans × şiddet göreli katsayıları (aynı
    faktör ve taban seviyelerle). Standart hata ve dağılım (dispersion) parametresi tahmini yok.
    """
    if not isinstance(rows, list) or not 1 <= len(rows) <= _GLM_MAX_ROWS:
        raise InputError(f"rows: 1–{_GLM_MAX_ROWS} satırlık liste bekleniyor.")
    if not isinstance(factors, list) or not 1 <= len(factors) <= 5 \
            or any(not isinstance(f, str) or not _FACTOR_NAME.fullmatch(f) for f in factors) or len(set(factors)) != len(factors):
        raise InputError("factors: 1–5 farklı faktör adı bekleniyor.")
    base_levels = {} if base_levels is None else base_levels
    if not isinstance(base_levels, dict) or set(base_levels) - set(factors):
        raise InputError("base_levels yalnız factors içindeki faktörleri içerebilir.")
    data = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"levels", "claims", "total_loss"} or not isinstance(row["levels"], dict):
            raise InputError(f"rows[{i}]: levels, claims, total_loss alanları bekleniyor.")
        if missing := [f for f in factors if f not in row["levels"]]:
            raise InputError(f"rows[{i}]: levels içinde eksik faktör: {', '.join(missing)} (fazlası yok sayılır).")
        levels = {}
        for f in factors:
            v = row["levels"][f]
            if not isinstance(v, str) or not _FACTOR_NAME.fullmatch(v):
                raise InputError(f"rows[{i}].levels.{f}: 1–40 karakterlik seviye adı bekleniyor.")
            levels[f] = v
        n = Decimal(integer(row["claims"], f"rows[{i}].claims", minimum=1, maximum=1_000_000_000))
        loss = number(row["total_loss"], f"rows[{i}].total_loss")
        if loss <= 0:
            raise InputError(f"rows[{i}].total_loss pozitif olmalı (hasarsız satırlar şiddet modeline girmez).")
        data.append((levels, n, loss / n))
    columns, bases = ["(intercept)"], {}
    for f in factors:
        seen = sorted({lv[f] for lv, _, _ in data})
        if len(seen) < 2:
            raise InputError(f"{f}: en az iki seviye gerekli.")
        base = base_levels.get(f, seen[0])
        if base not in seen:
            raise InputError(f"{f}: taban seviye '{base}' verilerde yok.")
        bases[f] = base
        columns += [f"{f}={level}" for level in seen if level != base]
    p = len(columns)
    if p > _GLM_MAX_PARAMS or p > len(data):
        raise InputError(f"GLM: parametre sayısı ({p}) satır sayısını aşamaz ve en çok {_GLM_MAX_PARAMS} olmalı.")
    x = [[Decimal(1)] + [Decimal(int(lv[c.split("=", 1)[0]] == c.split("=", 1)[1])) for c in columns[1:]] for lv, _, _ in data]
    w = [n for _, n, _ in data]
    y = [avg for _, _, avg in data]
    beta = [(sum(wi * yi for wi, yi in zip(w, y)) / sum(w)).ln()] + [Decimal(0)] * (p - 1)
    converged, iterations = False, 0
    for iterations in range(1, _GLM_MAX_ITER + 1):
        eta = [sum(xi * b for xi, b in zip(row, beta)) for row in x]
        mu = [v.exp() for v in eta]
        z = [et + (yi - m) / m for et, yi, m in zip(eta, y, mu)]
        xtwx = [[sum(x[k][i] * w[k] * x[k][j] for k in range(len(x))) for j in range(p)] for i in range(p)]
        xtwz = [sum(x[k][i] * w[k] * z[k] for k in range(len(x))) for i in range(p)]
        new = _solve(xtwx, xtwz)
        if max(abs(a - b) for a, b in zip(new, beta)) < Decimal("1e-12"):
            beta, converged = new, True
            break
        beta = new
    if not converged:
        raise InputError(f"GLM: {_GLM_MAX_ITER} yinelemede yakınsamadı.")
    mu = [sum(xi * b for xi, b in zip(row, beta)).exp() for row in x]
    deviance = 2 * sum(wi * (-(yi / m).ln() + (yi - m) / m) for wi, yi, m in zip(w, y, mu))
    q = Decimal("1e-10")
    return {
        "base_levels": bases,
        "coefficients": {c: _string(b.quantize(q)) for c, b in zip(columns, beta)},
        "relativities": {c: _sig(b.exp()) for c, b in zip(columns[1:], beta[1:])},
        "base_severity": _sig(beta[0].exp()),
        "deviance": _string(deviance.quantize(q)) if abs(deviance) >= q else "0",
        "degrees_of_freedom": len(data) - p,
        "iterations": iterations,
        "notes": "Log bağlantılı gamma, hasar adediyle ağırlıklı; standart hata ve dağılım parametresi yok.",
    }


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_NUM = {"type": ["number", "string"], "description": "Sonlu ondalık; oranlar yüzde değil (0.10 = %10)."}
_TRIANGLE = {"type": "array", "minItems": 2, "maxItems": 100,
             "items": {"type": "array", "items": {"type": ["number", "string", "null"]}},
             "description": "Kümülatif üçgen; artımlı veri değildir."}
_BASIS = {"type": "string", "enum": ["paid", "incurred"]}
_LIST_NUM = {"type": "array", "items": _NUM}

TOOLS = {tool.name: tool for tool in [
    Tool("present_value", "Açık dönemlik oranla net bugünkü değer.", present_value,
         _schema({"cashflows": {"type": "array", "maxItems": 10_000, "items": _schema({
             "period": {"type": "integer", "minimum": 0, "maximum": 10_000}, "amount": _NUM,
         }, ["period", "amount"])}, "rate": _NUM}, ["cashflows", "rate"])),
    Tool("annuity", "Dönem sonu veya başı sabit ödeme annüitesi.", annuity,
         _schema({"payment": _NUM, "rate": _NUM, "periods": {"type": "integer", "minimum": 1, "maximum": 10_000},
                  "timing": {"type": "string", "enum": ["immediate", "due"]}}, ["payment", "rate", "periods"])),
    Tool("life_insurance_pv", "Verilen qx ile ölüm döneminin sonunda ödenen süreli hayat teminatı.", life_insurance_pv,
         _schema({"qx": {"type": "array", "items": _NUM, "minItems": 1, "maxItems": 200},
                  "benefit": _NUM, "rate": _NUM}, ["qx", "benefit", "rate"])),
    Tool("chain_ladder", "Açık paid/incurred bazında kümülatif Chain Ladder; saf IBNR varsayılmaz.", chain_ladder,
         _schema({"triangle": _TRIANGLE, "basis": _BASIS, "tail_factor": _NUM}, ["triangle", "basis"])),
    Tool("bornhuetter_ferguson", "Kazanılmış prim ve a-priori hasar oranlarıyla BF tahmini.", bornhuetter_ferguson,
         _schema({"triangle": _TRIANGLE, "basis": _BASIS, "earned_premiums": _LIST_NUM,
                  "expected_loss_ratios": _LIST_NUM, "tail_factor": _NUM},
                 ["triangle", "basis", "earned_premiums", "expected_loss_ratios"])),
    Tool("portfolio_metrics", "Tutarlı kapalı kohort için frekans, şiddet, saf prim ve hasar oranı.", portfolio_metrics,
         _schema({"exposure": _NUM, "claim_count": {"type": "integer", "minimum": 0},
                  "total_loss": _NUM, "earned_premium": _NUM},
                 ["exposure", "claim_count", "total_loss", "earned_premium"])),
]}


# Extended tools: available to people through the CLI/demo, NOT offered to the model. The model's tool list is part
# of the prompt that baselines and SFT runs were measured with; exposing these needs a new training/eval decision.
_DATE = {"type": "string", "description": "YYYY-MM-DD"}
EXTENDED_TOOLS = {tool.name: tool for tool in [
    Tool("life_annuity_pv", "Verilen qx ile yaşam koşullu süreli annüite (dönem sonu veya başı).", life_annuity_pv,
         _schema({"qx": {"type": "array", "items": _NUM, "minItems": 1, "maxItems": 200}, "payment": _NUM, "rate": _NUM,
                  "timing": {"type": "string", "enum": ["immediate", "due"]}}, ["qx", "payment", "rate"])),
    Tool("xl_layer_recovery", "Olay başı hasar fazlası katmanından tahsilat; isteğe bağlı reinstatement ile agregat sınır.",
         xl_layer_recovery, _schema({"losses": _LIST_NUM, "retention": _NUM, "limit": _NUM,
                                     "reinstatements": {"type": "integer", "minimum": 0, "maximum": 100}},
                                    ["losses", "retention", "limit"])),
    Tool("credibility_estimate", "Bühlmann tipi güvenilirlik ağırlığı (k açıkça verilir).", credibility_estimate,
         _schema({"observed_mean": _NUM, "exposure_units": _NUM, "prior_mean": _NUM, "k": _NUM},
                 ["observed_mean", "exposure_units", "prior_mean", "k"])),
    Tool("earned_exposure", "Poliçe tarihlerinden dönem içinde kazanılmış maruziyet (yıl).", earned_exposure,
         _schema({"policies": {"type": "array", "items": _schema({"start": _DATE, "end": _DATE}, ["start", "end"])},
                  "period_start": _DATE, "period_end": _DATE,
                  "days_per_year": {"type": "integer", "minimum": 360, "maximum": 366}},
                 ["policies", "period_start", "period_end", "days_per_year"])),
    Tool("poisson_glm", "Maruziyet ofsetli log bağlantılı Poisson frekans GLM'i (kategorik faktörler, göreli katsayılar).",
         poisson_glm, _schema({"rows": {"type": "array", "maxItems": _GLM_MAX_ROWS, "items": _schema({
             "levels": {"type": "object", "additionalProperties": {"type": "string"}}, "exposure": _NUM,
             "claims": {"type": "integer", "minimum": 0, "maximum": 1_000_000_000}}, ["levels", "exposure", "claims"])},
             "factors": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
             "base_levels": {"type": "object", "additionalProperties": {"type": "string"}}}, ["rows", "factors"])),
    Tool("gamma_severity_glm", "Hasar adediyle ağırlıklı log bağlantılı gamma şiddet GLM'i (kategorik faktörler).",
         gamma_severity_glm, _schema({"rows": {"type": "array", "maxItems": _GLM_MAX_ROWS, "items": _schema({
             "levels": {"type": "object", "additionalProperties": {"type": "string"}},
             "claims": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000}, "total_loss": _NUM},
             ["levels", "claims", "total_loss"])},
             "factors": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
             "base_levels": {"type": "object", "additionalProperties": {"type": "string"}}}, ["rows", "factors"])),
    Tool("life_cashflow_projection", "Ölüm ve iptal çıkışlı dönemlik hayat poliçesi nakit akışı ve bugünkü değerler.",
         life_cashflow_projection, _schema({
             "qx": {"type": "array", "items": _NUM, "minItems": 1, "maxItems": 200}, "rate": _NUM, "premium": _NUM,
             "death_benefit": _NUM, "lapse": {"type": "array", "items": _NUM, "maxItems": 200},
             "maturity_benefit": _NUM, "expense_per_policy": _NUM,
             "surrender_value": {"type": "array", "items": _NUM, "maxItems": 200}},
             ["qx", "rate", "premium", "death_benefit"])),
    Tool("combined_ratio", "Aynı kazanılmış esasta hasar (LAE dahil), gider ve bileşik oran.", combined_ratio,
         _schema({"incurred_losses": _NUM, "loss_adjustment_expenses": _NUM, "acquisition_costs": _NUM,
                  "other_expenses": _NUM, "earned_premium": _NUM},
                 ["incurred_losses", "loss_adjustment_expenses", "acquisition_costs", "other_expenses", "earned_premium"])),
]}
ALL_TOOLS = {**TOOLS, **EXTENDED_TOOLS}


MAX_TOOL_CALLS = 2  # Per question; the assistant stops offering tools after this many results.


def visible_tool_schemas(prior_results: int) -> list[dict[str, Any]] | None:
    """Tool schemas offered to the model given how many tool results it has already seen.

    Shared by inference (PilotAssistant) and SFT encoding so both render identical prompts.
    """
    return [tool.api_schema() for tool in TOOLS.values()] if prior_results < MAX_TOOL_CALLS else None


def run_tool_call(name: Any, arguments: Any) -> dict[str, Any]:
    """The tool-result record returned to the model; shared by inference and SFT validation."""
    try:
        return {"tool": name, "arguments": arguments, "output": invoke_tool(name, arguments), "ok": True}
    except InputError as exc:
        return {"tool": name, "arguments": arguments, "ok": False, "error": str(exc)}


def invoke_tool(name: str, arguments: Any, *, include_extended: bool = False) -> dict[str, Any]:
    """Run a registered tool. The model path never sets include_extended; people (CLI/demo) may."""
    if not isinstance(name, str):
        raise InputError("Araç adı string olmalı.")
    tool = (ALL_TOOLS if include_extended else TOOLS).get(name)
    if tool is None:
        raise InputError(f"İzinli olmayan araç: {name}")
    if not isinstance(arguments, dict):
        raise InputError("Araç girdisi JSON nesnesi olmalı.")
    try:
        inspect.signature(tool.function).bind(**arguments)
    except TypeError as exc:
        raise InputError(f"{name}: eksik/bilinmeyen araç parametresi: {exc}") from exc
    try:
        return tool.function(**arguments)
    except TypeError as exc:
        raise InputError(f"{name}: geçersiz girdi türü.") from exc
