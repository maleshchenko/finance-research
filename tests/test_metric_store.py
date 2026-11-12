from app.retrieval.metric_store import MetricRecord, _prefer_sec_records


def _record(quarter: int, provider: str | None, value: float) -> MetricRecord:
    return MetricRecord(
        company_ticker="AAPL",
        fiscal_year=2023,
        fiscal_quarter=quarter,
        period_end_date=None,
        metric_name="operating_margin",
        value=value,
        unit="percent",
        segment="Total",
        is_adjusted=False,
        document_id=quarter,
        source_page=1,
        source_provider=provider,
    )


def test_sec_record_replaces_synthetic_metric_for_same_period():
    records = [
        _record(1, None, 32.0),
        _record(1, "sec_companyfacts", 30.7424),
        _record(2, None, 31.6),
        _record(2, "sec_companyfacts", 29.86),
    ]

    selected = _prefer_sec_records(records, {("AAPL", 2023, 1), ("AAPL", 2023, 2)})

    assert [(record.fiscal_quarter, record.value) for record in selected] == [
        (1, 30.7424),
        (2, 29.86),
    ]


def test_sec_period_does_not_fall_back_to_synthetic_metric_when_sec_value_missing():
    records = [_record(4, None, 1.88)]

    assert _prefer_sec_records(records, {("AAPL", 2023, 4)}) == []
