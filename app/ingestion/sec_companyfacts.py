"""Fetch and normalize quarterly SEC Company Facts for supported issuers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx

SEC_COMPANIES = {
    "AAPL": ("0000320193", "Apple Inc."),
    "MSFT": ("0000789019", "Microsoft Corporation"),
    "GOOGL": ("0001652044", "Alphabet Inc."),
}

_SEC_API_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
_DURATION_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A"}
_QUARTERLY_FORMS = {"10-Q", "10-Q/A"}
_ANNUAL_FORMS = {"10-K", "10-K/A"}
_REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
)
_OPERATING_INCOME_TAGS = ("OperatingIncomeLoss",)
_NET_INCOME_TAGS = ("NetIncomeLoss", "ProfitLoss")
_MILLION = 1_000_000


class SECCompanyFactsError(ValueError):
    """Raised when SEC facts cannot support a requested fiscal period."""


@dataclass(frozen=True)
class FactObservation:
    tag: str
    value: float
    unit: str
    start: date
    end: date
    accession: str
    filed: date
    form: str
    fiscal_year: int
    fiscal_period: str


@dataclass(frozen=True)
class MetricValue:
    value: float
    unit: str
    source_tags: tuple[str, ...]
    source_accessions: tuple[str, ...]
    calculation: str | None = None


@dataclass(frozen=True)
class SECQuarter:
    ticker: str
    company_name: str
    cik: str
    fiscal_year: int
    fiscal_quarter: int
    period_start: date
    period_end: date
    source_form: str
    source_accessions: tuple[str, ...]
    metrics: dict[str, MetricValue]
    source_url: str


def companyfacts_url(cik: str) -> str:
    return f"{_SEC_API_BASE}/CIK{cik}.json"


async def fetch_companyfacts(ticker: str, user_agent: str) -> dict[str, Any]:
    """Fetch one issuer's SEC Company Facts JSON using the caller's User-Agent."""
    normalized_ticker = ticker.upper()
    company = SEC_COMPANIES.get(normalized_ticker)
    if company is None:
        supported = ", ".join(sorted(SEC_COMPANIES))
        raise ValueError(f"Unsupported ticker {ticker!r}. Supported tickers: {supported}.")
    if not user_agent.strip():
        raise ValueError("A descriptive SEC User-Agent is required.")

    url = companyfacts_url(company[0])
    async with httpx.AsyncClient(
        headers={"User-Agent": user_agent},
        timeout=httpx.Timeout(30.0),
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), dict):
        raise SECCompanyFactsError(
            f"SEC returned an unexpected Company Facts payload for {ticker}."
        )
    return payload


def build_quarterly_metrics(
    payload: dict[str, Any],
    ticker: str,
    fiscal_year: int,
) -> list[SECQuarter]:
    """Build reported Q1–Q3 metrics and derive Q4 from the annual filing."""
    normalized_ticker = ticker.upper()
    company = SEC_COMPANIES.get(normalized_ticker)
    if company is None:
        supported = ", ".join(sorted(SEC_COMPANIES))
        raise ValueError(f"Unsupported ticker {ticker!r}. Supported tickers: {supported}.")

    facts = payload.get("facts", {}).get("us-gaap", {})
    if not isinstance(facts, dict):
        raise SECCompanyFactsError(f"SEC Company Facts has no us-gaap facts for {ticker}.")

    revenue_tag, revenue_observations, annual_revenue = _find_annual_fact(
        facts, _REVENUE_TAGS, fiscal_year
    )
    operating_income_tag, operating_income_observations, annual_operating_income = (
        _find_matching_annual_fact(
            facts, _OPERATING_INCOME_TAGS, fiscal_year, annual_revenue, ticker, "operating income"
        )
    )
    net_income_tag, net_income_observations, annual_net_income = _find_matching_annual_fact(
        facts, _NET_INCOME_TAGS, fiscal_year, annual_revenue, ticker, "net income"
    )

    annual_facts = {
        "revenue": (revenue_tag, annual_revenue),
        "operating_income": (operating_income_tag, annual_operating_income),
        "net_income": (net_income_tag, annual_net_income),
    }
    quarter_facts = {
        "revenue": (revenue_tag, revenue_observations),
        "operating_income": (operating_income_tag, operating_income_observations),
        "net_income": (net_income_tag, net_income_observations),
    }

    raw_quarters: list[dict[str, Any]] = []
    previous_end: date | None = None
    for quarter in (1, 2, 3):
        period_observation = _find_quarter_period(
            revenue_observations, fiscal_year, quarter, annual_revenue
        )
        period_end = period_observation.end
        period_start = period_observation.start
        if quarter > 1 and period_start == annual_revenue.start:
            if previous_end is None:
                raise SECCompanyFactsError(
                    f"Missing prior-quarter end date for {ticker} FY{fiscal_year} Q{quarter}."
                )
            period_start = previous_end + timedelta(days=1)

        metric_facts: dict[str, tuple[float, tuple[FactObservation, ...], str | None]] = {}
        for metric_name, (tag, observations) in quarter_facts.items():
            amount, sources, calculation = _quarter_amount(
                observations,
                fiscal_year=fiscal_year,
                quarter=quarter,
                period_start=period_start,
                period_end=period_end,
                fiscal_year_start=annual_revenue.start,
            )
            metric_facts[metric_name] = (amount, sources, calculation)

        raw_quarters.append(
            {
                "quarter": quarter,
                "period_start": period_start,
                "period_end": period_end,
                "form": period_observation.form,
                "metrics": metric_facts,
            }
        )
        previous_end = period_end

    q4_metrics: dict[str, tuple[float, tuple[FactObservation, ...], str]] = {}
    for metric_name, (_, annual_observation) in annual_facts.items():
        quarter_values = [
            raw_quarter["metrics"][metric_name][0] for raw_quarter in raw_quarters
        ]
        quarter_sources = [
            source
            for raw_quarter in raw_quarters
            for source in raw_quarter["metrics"][metric_name][1]
        ]
        sources = _unique_observations((annual_observation, *quarter_sources))
        q4_metrics[metric_name] = (
            annual_observation.value - sum(quarter_values),
            sources,
            "Q4 value calculated as the SEC FY amount minus reported Q1–Q3 amounts.",
        )

    quarters: list[SECQuarter] = []
    for raw_quarter in raw_quarters:
        quarters.append(
            _make_quarter(
                payload=payload,
                ticker=normalized_ticker,
                company=company,
                fiscal_year=fiscal_year,
                quarter=raw_quarter["quarter"],
                period_start=raw_quarter["period_start"],
                period_end=raw_quarter["period_end"],
                source_form=raw_quarter["form"],
                metric_facts=raw_quarter["metrics"],
            )
        )

    q4_start = (previous_end or annual_revenue.end) + timedelta(days=1)
    quarters.append(
        _make_quarter(
            payload=payload,
            ticker=normalized_ticker,
            company=company,
            fiscal_year=fiscal_year,
            quarter=4,
            period_start=q4_start,
            period_end=annual_revenue.end,
            source_form=annual_revenue.form,
            metric_facts=q4_metrics,
        )
    )
    return quarters


def _find_annual_fact(
    facts: dict[str, Any],
    tags: tuple[str, ...],
    fiscal_year: int,
) -> tuple[str, list[FactObservation], FactObservation]:
    for tag in tags:
        observations = _load_observations(facts.get(tag), tag, "USD")
        candidates = [
            observation
            for observation in observations
            if observation.fiscal_year == fiscal_year
            and observation.fiscal_period == "FY"
            and observation.form in _ANNUAL_FORMS
            and observation.end.year >= fiscal_year
            and 330 <= (observation.end - observation.start).days <= 400
        ]
        if candidates:
            latest_end = max(observation.end for observation in candidates)
            candidates = [
                observation
                for observation in candidates
                if observation.end == latest_end
            ]
            annual = _latest_fact(candidates, f"FY{fiscal_year} {tag}")
            return tag, observations, annual
    raise SECCompanyFactsError(
        f"No annual SEC XBRL fact found for FY{fiscal_year}; tried tags {tags}."
    )


def _find_matching_annual_fact(
    facts: dict[str, Any],
    tags: tuple[str, ...],
    fiscal_year: int,
    annual_revenue: FactObservation,
    ticker: str,
    label: str,
) -> tuple[str, list[FactObservation], FactObservation]:
    for tag in tags:
        observations = _load_observations(facts.get(tag), tag, "USD")
        candidates = [
            observation
            for observation in observations
            if observation.fiscal_year == fiscal_year
            and observation.fiscal_period == "FY"
            and observation.form in _ANNUAL_FORMS
            and observation.start == annual_revenue.start
            and observation.end == annual_revenue.end
        ]
        if candidates:
            annual = _latest_fact(candidates, f"{ticker} FY{fiscal_year} {label}")
            return tag, observations, annual
    raise SECCompanyFactsError(
        f"No FY{fiscal_year} SEC XBRL {label} fact matched the revenue period for {ticker}."
    )


def _load_observations(
    fact: Any,
    tag: str,
    unit: str,
) -> list[FactObservation]:
    if not isinstance(fact, dict):
        return []
    units = fact.get("units")
    entries = units.get(unit, []) if isinstance(units, dict) else []
    observations: list[FactObservation] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        required = ("start", "end", "accn", "filed", "form", "fy", "fp", "val")
        if any(key not in entry for key in required):
            continue
        if entry["form"] not in _DURATION_FORMS:
            continue
        observations.append(
            FactObservation(
                tag=tag,
                value=float(entry["val"]),
                unit=unit,
                start=date.fromisoformat(entry["start"]),
                end=date.fromisoformat(entry["end"]),
                accession=entry["accn"],
                filed=date.fromisoformat(entry["filed"]),
                form=entry["form"],
                fiscal_year=int(entry["fy"]),
                fiscal_period=str(entry["fp"]),
            )
        )
    return observations


def _find_quarter_period(
    observations: list[FactObservation],
    fiscal_year: int,
    quarter: int,
    annual: FactObservation,
) -> FactObservation:
    period = f"Q{quarter}"
    in_fiscal_year = [
        observation
        for observation in observations
        if observation.fiscal_year == fiscal_year
        and observation.fiscal_period == period
        and observation.form in _QUARTERLY_FORMS
        and annual.start <= observation.start < observation.end <= annual.end
    ]
    standalone = [
        observation
        for observation in in_fiscal_year
        if 70 <= (observation.end - observation.start).days <= 105
    ]
    if standalone:
        latest_end = max(observation.end for observation in standalone)
        standalone = [observation for observation in standalone if observation.end == latest_end]
        return _latest_fact(standalone, f"FY{fiscal_year} {period} revenue")

    year_to_date = [
        observation
        for observation in in_fiscal_year
        if observation.start == annual.start
        and 70 <= (observation.end - observation.start).days <= 310
    ]
    if year_to_date:
        latest_end = max(observation.end for observation in year_to_date)
        year_to_date = [
            observation for observation in year_to_date if observation.end == latest_end
        ]
        return _latest_fact(year_to_date, f"FY{fiscal_year} {period} YTD revenue")
    raise SECCompanyFactsError(
        f"Missing SEC revenue period for FY{fiscal_year} {period}."
    )


def _quarter_amount(
    observations: list[FactObservation],
    fiscal_year: int,
    quarter: int,
    period_start: date,
    period_end: date,
    fiscal_year_start: date,
) -> tuple[float, tuple[FactObservation, ...], str | None]:
    period = f"Q{quarter}"
    direct = [
        observation
        for observation in observations
        if observation.fiscal_year == fiscal_year
        and observation.fiscal_period == period
        and observation.form in _QUARTERLY_FORMS
        and observation.start == period_start
        and observation.end == period_end
    ]
    if direct:
        observation = _latest_fact(direct, f"FY{fiscal_year} {period} {direct[0].tag}")
        return observation.value, (observation,), None

    current_ytd = [
        observation
        for observation in observations
        if observation.fiscal_year == fiscal_year
        and observation.fiscal_period == period
        and observation.form in _QUARTERLY_FORMS
        and observation.start == fiscal_year_start
        and observation.end == period_end
    ]
    if not current_ytd:
        raise SECCompanyFactsError(
            f"Missing SEC XBRL value for FY{fiscal_year} {period} {observations[0].tag}."
        )
    current = _latest_fact(current_ytd, f"FY{fiscal_year} {period} YTD {current_ytd[0].tag}")
    if quarter == 1:
        return current.value, (current,), None

    prior_ytd = [
        observation
        for observation in observations
        if observation.fiscal_year == fiscal_year
        and observation.fiscal_period == f"Q{quarter - 1}"
        and observation.form in _QUARTERLY_FORMS
        and observation.start == fiscal_year_start
        and observation.end < period_end
    ]
    if not prior_ytd:
        raise SECCompanyFactsError(
            f"Cannot derive FY{fiscal_year} {period} without prior-year-to-date XBRL data."
        )
    previous = _latest_fact(
        prior_ytd,
        f"FY{fiscal_year} Q{quarter - 1} YTD {prior_ytd[0].tag}",
    )
    return (
        current.value - previous.value,
        (current, previous),
        f"Quarter amount calculated as Q{quarter} YTD minus Q{quarter - 1} YTD.",
    )


def _latest_fact(observations: list[FactObservation], label: str) -> FactObservation:
    latest_filed = max(observation.filed for observation in observations)
    latest = [observation for observation in observations if observation.filed == latest_filed]
    values = {observation.value for observation in latest}
    if len(values) > 1:
        raise SECCompanyFactsError(f"Conflicting latest SEC XBRL facts for {label}.")
    return max(latest, key=lambda observation: observation.accession)


def _unique_observations(
    observations: tuple[FactObservation, ...] | list[FactObservation],
) -> tuple[FactObservation, ...]:
    unique: dict[tuple[str, str], FactObservation] = {}
    for observation in observations:
        unique[(observation.accession, observation.tag)] = observation
    return tuple(unique.values())


def _make_quarter(
    payload: dict[str, Any],
    ticker: str,
    company: tuple[str, str],
    fiscal_year: int,
    quarter: int,
    period_start: date,
    period_end: date,
    source_form: str,
    metric_facts: dict[str, tuple[float, tuple[FactObservation, ...], str | None]],
) -> SECQuarter:
    cik, fallback_name = company
    company_name = str(payload.get("entityName") or fallback_name)
    source_url = companyfacts_url(cik)
    metrics: dict[str, MetricValue] = {}

    for metric_name in ("revenue", "operating_income", "net_income"):
        amount, sources, calculation = metric_facts[metric_name]
        metrics[metric_name] = MetricValue(
            value=round(amount / _MILLION, 6),
            unit="USD_millions",
            source_tags=tuple(dict.fromkeys(source.tag for source in sources)),
            source_accessions=tuple(dict.fromkeys(source.accession for source in sources)),
            calculation=calculation,
        )

    revenue = metric_facts["revenue"][0]
    operating_income = metric_facts["operating_income"][0]
    if revenue == 0:
        raise SECCompanyFactsError(
            f"Cannot calculate operating margin for {ticker} FY{fiscal_year} "
            f"Q{quarter}: revenue is zero."
        )
    margin_calculation = "Operating margin = operating income / revenue × 100."
    if quarter == 4:
        margin_calculation = (
            "Q4 line items are FY amounts minus Q1–Q3; operating margin is "
            "calculated as Q4 operating income / Q4 revenue × 100."
        )
    metrics["operating_margin"] = MetricValue(
        value=round(operating_income / revenue * 100, 4),
        unit="percent",
        source_tags=tuple(
            dict.fromkeys(
                source.tag
                for metric_name in ("operating_income", "revenue")
                for source in metric_facts[metric_name][1]
            )
        ),
        source_accessions=tuple(
            dict.fromkeys(
                source.accession
                for metric_name in ("operating_income", "revenue")
                for source in metric_facts[metric_name][1]
            )
        ),
        calculation=margin_calculation,
    )

    all_sources = _unique_observations(
        [
            source
            for _, sources, _ in metric_facts.values()
            for source in sources
        ]
    )
    accessions = tuple(dict.fromkeys(source.accession for source in all_sources))
    return SECQuarter(
        ticker=ticker,
        company_name=company_name,
        cik=cik,
        fiscal_year=fiscal_year,
        fiscal_quarter=quarter,
        period_start=period_start,
        period_end=period_end,
        source_form=source_form,
        source_accessions=accessions,
        metrics=metrics,
        source_url=source_url,
    )
