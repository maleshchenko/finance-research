from copy import deepcopy

import pytest

from app.ingestion.sec_companyfacts import (
    SECCompanyFactsError,
    build_quarterly_metrics,
)


def _entry(
    value: int,
    start: str,
    end: str,
    accession: str,
    form: str,
    fiscal_period: str,
    filed: str,
) -> dict:
    return {
        "val": value,
        "start": start,
        "end": end,
        "accn": accession,
        "form": form,
        "fy": 2023,
        "fp": fiscal_period,
        "filed": filed,
    }


def _companyfacts_fixture() -> dict:
    periods = {
        "revenue": {
            "annual": (383_285_000_000, 394_328_000_000),
            "q1": 117_154_000_000,
            "q2": 94_836_000_000,
            "q2_ytd": 211_990_000_000,
            "q3": 81_797_000_000,
            "q3_ytd": 293_787_000_000,
            "prior_q2": 97_278_000_000,
        },
        "operating": {
            "annual": (114_301_000_000, 119_437_000_000),
            "q1": 36_016_000_000,
            "q2": 28_318_000_000,
            "q2_ytd": 64_334_000_000,
            "q3": 22_998_000_000,
            "q3_ytd": 87_332_000_000,
            "prior_q2": 29_979_000_000,
        },
        "net": {
            "annual": (96_995_000_000, 99_803_000_000),
            "q1": 29_998_000_000,
            "q2": 24_160_000_000,
            "q2_ytd": 54_158_000_000,
            "q3": 19_881_000_000,
            "q3_ytd": 74_039_000_000,
            "prior_q2": 25_010_000_000,
        },
    }
    accession_by_quarter = {
        "q1": "0000320193-23-000006",
        "q2": "0000320193-23-000064",
        "q3": "0000320193-23-000077",
    }
    facts = {}
    for metric, values in periods.items():
        entries = [
            _entry(
                values["annual"][1],
                "2021-09-26",
                "2022-09-24",
                "0000320193-23-000106",
                "10-K",
                "FY",
                "2023-11-03",
            ),
            _entry(
                values["annual"][0],
                "2022-09-25",
                "2023-09-30",
                "0000320193-23-000106",
                "10-K",
                "FY",
                "2023-11-03",
            ),
            _entry(
                values["q1"],
                "2022-09-25",
                "2022-12-31",
                accession_by_quarter["q1"],
                "10-Q",
                "Q1",
                "2023-02-03",
            ),
            _entry(
                values["prior_q2"],
                "2021-12-26",
                "2022-03-26",
                accession_by_quarter["q2"],
                "10-Q",
                "Q2",
                "2023-05-05",
            ),
            _entry(
                values["q2_ytd"],
                "2022-09-25",
                "2023-04-01",
                accession_by_quarter["q2"],
                "10-Q",
                "Q2",
                "2023-05-05",
            ),
            _entry(
                values["q2"],
                "2023-01-01",
                "2023-04-01",
                accession_by_quarter["q2"],
                "10-Q",
                "Q2",
                "2023-05-05",
            ),
            _entry(
                values["q3_ytd"],
                "2022-09-25",
                "2023-07-01",
                accession_by_quarter["q3"],
                "10-Q",
                "Q3",
                "2023-08-04",
            ),
            _entry(
                values["q3"],
                "2023-04-02",
                "2023-07-01",
                accession_by_quarter["q3"],
                "10-Q",
                "Q3",
                "2023-08-04",
            ),
        ]
        tag = {
            "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "operating": "OperatingIncomeLoss",
            "net": "NetIncomeLoss",
        }[metric]
        facts[tag] = {
            "units": {
                "USD": entries,
            }
        }

    return {
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": facts,
        },
    }


def test_builds_current_quarters_and_derives_q4_from_sec_facts():
    quarters = build_quarterly_metrics(_companyfacts_fixture(), "AAPL", 2023)

    assert [quarter.period_end.isoformat() for quarter in quarters] == [
        "2022-12-31",
        "2023-04-01",
        "2023-07-01",
        "2023-09-30",
    ]
    assert quarters[0].metrics["revenue"].value == 117154.0
    assert quarters[1].metrics["revenue"].value == 94836.0
    assert quarters[1].metrics["operating_income"].value == 28318.0
    assert quarters[1].metrics["net_income"].value == 24160.0
    assert quarters[3].metrics["revenue"].value == 89498.0
    assert quarters[3].metrics["operating_income"].value == 26969.0
    assert quarters[3].metrics["net_income"].value == 22956.0
    assert quarters[3].metrics["operating_margin"].value == pytest.approx(30.1336)
    assert quarters[0].metrics["operating_margin"].calculation == (
        "Operating margin = operating income / revenue × 100."
    )
    assert quarters[3].metrics["operating_margin"].calculation == (
        "Q4 line items are FY amounts minus Q1–Q3; operating margin is "
        "calculated as Q4 operating income / Q4 revenue × 100."
    )
    assert quarters[3].metrics["revenue"].calculation == (
        "Q4 value calculated as the SEC FY amount minus reported Q1–Q3 amounts."
    )
    assert quarters[3].source_accessions == (
        "0000320193-23-000106",
        "0000320193-23-000006",
        "0000320193-23-000064",
        "0000320193-23-000077",
    )


def test_uses_ytd_differences_when_quarter_only_fact_is_missing():
    payload = _companyfacts_fixture()
    payload = deepcopy(payload)
    for fact in payload["facts"]["us-gaap"].values():
        fact["units"]["USD"] = [
            entry
            for entry in fact["units"]["USD"]
            if not (
                entry["start"] == "2023-01-01"
                and entry["end"] == "2023-04-01"
            )
        ]

    quarters = build_quarterly_metrics(payload, "AAPL", 2023)

    assert quarters[1].metrics["revenue"].value == 94836.0
    assert "Q2 YTD minus Q1 YTD" in quarters[1].metrics["revenue"].calculation


def test_does_not_use_prior_year_comparative_as_target_annual_fact():
    payload = _companyfacts_fixture()
    for fact in payload["facts"]["us-gaap"].values():
        fact["units"]["USD"] = [
            entry
            for entry in fact["units"]["USD"]
            if entry["end"] != "2023-09-30"
        ]

    with pytest.raises(SECCompanyFactsError, match="No annual SEC XBRL fact"):
        build_quarterly_metrics(payload, "AAPL", 2023)
