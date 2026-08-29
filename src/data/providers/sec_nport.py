"""SEC N-PORT holdings bulk ingest: ZIP parsing and normalization."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final

import polars as pl

from src.data.schema import TS_DTYPE, Dataset, spec_for

_PROVIDER: Final[str] = "sec"
_SOURCE: Final[str] = "sec_nport"
_PLACEHOLDER_CUSIPS: Final[frozenset[str]] = frozenset({"000000000", "00000000", "999999999"})


def _normalize_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _find_column(frame: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in frame.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _parse_raw_tables(content: bytes) -> dict[str, pl.DataFrame]:
    tables: dict[str, pl.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            raw = archive.read(member)
            if not raw:
                continue
            name = member.rsplit("/", 1)[-1]
            base = name.rsplit(".", 1)[0]
            key = base.upper()
            text = raw.decode("utf-8", errors="replace")
            df: pl.DataFrame | None = None
            read_kwargs = {
                "infer_schema_length": 10_000,
                "null_values": ["N/A", "n/a", "NA", "null", ""],
                "ignore_errors": True,
            }
            for sep in ("\t", ",", "|"):
                try:
                    candidate = pl.read_csv(
                        io.StringIO(text),
                        separator=sep,
                        try_parse_dates=True,
                        **read_kwargs,  # type: ignore[arg-type]
                    )
                    if candidate.width >= 2:
                        df = candidate
                        break
                except Exception:  # noqa: S112
                    continue
            if df is None:
                try:
                    df = pl.read_csv(io.StringIO(text), try_parse_dates=True, **read_kwargs)  # type: ignore[arg-type]
                except Exception:  # noqa: S112
                    continue
            tables[key] = df
    return tables


def _to_date_series(s: pl.Series) -> pl.Series:  # noqa: ARG001
    if s.dtype == pl.Date:
        return s
    if isinstance(s.dtype, pl.Datetime):
        return s.dt.date()
    try:
        return s.cast(pl.String).str.to_date(strict=False)
    except Exception:  # noqa: S110
        pass
    try:
        return s.cast(pl.String).str.strptime(pl.Date, strict=False)
    except Exception:  # noqa: S110
        return s.cast(pl.String).str.to_date(strict=False)


def _to_datetime_utc(s: pl.Series) -> pl.Series:
    if isinstance(s.dtype, pl.Datetime):
        if s.dtype.time_zone is None:
            return s.dt.replace_time_zone("UTC")
        return s.dt.convert_time_zone("UTC")
    # string
    return s.cast(pl.String).str.to_datetime(time_zone="UTC", strict=False)


def _parse_filing_datetime(column: str) -> pl.Expr:
    as_str = pl.col(column).cast(pl.String)
    compact = as_str.str.slice(0, 10)
    return pl.coalesce(
        as_str.str.to_date("%d-%b-%Y", strict=False).cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
        compact.str.to_date("%Y-%m-%d", strict=False).cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
    ).cast(TS_DTYPE)


def _parse_report_date(column: str) -> pl.Expr:
    as_str = pl.col(column).cast(pl.String)
    compact = as_str.str.slice(0, 10)
    return pl.coalesce(
        as_str.str.to_date("%d-%b-%Y", strict=False),
        compact.str.to_date("%Y-%m-%d", strict=False),
    )


def _target_accessions(
    series_map: Mapping[str, str],
    info: pl.DataFrame | None,
    *,
    i_series: str | None,
    i_accession: str | None,
) -> set[str]:
    """Resolve SEC accession numbers covered by the series map."""
    accessions: set[str] = set()
    for key in series_map:
        if "-" in key:
            accessions.add(key)
            continue
        if info is None or i_series is None or i_accession is None:
            continue
        matches = (
            info.filter(pl.col(i_series).cast(pl.String) == str(key))
            .select(pl.col(i_accession).cast(pl.String))
            .drop_nulls()
            .unique()
        )
        accessions.update(matches.to_series().to_list())
    return accessions


def normalize_nport_holdings(
    raw_tables: Mapping[str, pl.DataFrame],
    *,
    series_map: Mapping[str, str],
    retrieved_at: datetime,
) -> pl.DataFrame:
    """Join FUND_REPORTED_HOLDING + IDENTIFIERS + FUND_REPORTED_INFO into etf_holdings."""
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")
    retrieved_at = retrieved_at.astimezone(UTC)

    # locate tables (case-insensitive)
    normed: dict[str, pl.DataFrame] = {k.upper(): v for k, v in raw_tables.items()}
    # find holding table
    holding_key = next((k for k in normed if "HOLDING" in k and "INFO" not in k), None)
    info_key = next((k for k in normed if "INFO" in k and "HOLDING" not in k), None)
    id_key = next((k for k in normed if "IDENTIFIER" in k), None)
    submission_key = next((k for k in normed if k == "SUBMISSION"), None)
    if holding_key is None:
        raise ValueError("raw_tables missing FUND_REPORTED_HOLDING")
    holding = normed[holding_key]
    info = normed.get(info_key) if info_key else None
    identifiers = normed.get(id_key) if id_key else None

    # Normalize column mapping helpers
    def col(df: pl.DataFrame, *cands: str) -> str | None:
        return _find_column(df, tuple(cands))

    # Extract holdings columns
    h_holding_id = col(holding, "holding_id", "holdingid", "id", "identifier")
    h_series = col(holding, "series_id", "seriesid", "series", "series_name")
    h_report = col(holding, "report_date", "reportdate", "period_end", "report_date_str")
    h_value = col(holding, "value_usd", "valueusd", "val_usd", "value", "mkt_value", "currency_value")
    h_weight = col(holding, "weight_pct", "weightpct", "pct_val", "pctval", "percent_val", "weight", "percentage")
    h_issuer = col(holding, "issuer_name", "issuername", "issuer", "name", "title", "issuer_title")
    h_cusip = col(holding, "cusip", "issuer_cusip")
    h_isin = col(holding, "isin")
    h_lei = col(holding, "lei", "issuer_lei")
    h_accession = col(holding, "accession_number", "accessionnumber", "accession")
    h_filing = col(holding, "filing_date", "filingdate", "filing_dt", "acceptance_datetime")

    if h_holding_id is None:
        raise ValueError("FUND_REPORTED_HOLDING missing required holding_id column")
    if h_series is None and h_accession is None:
        raise ValueError("FUND_REPORTED_HOLDING missing series_id and accession_number columns")

    # Info columns
    i_series = col(info, "series_id", "seriesid", "series") if info is not None else None
    i_filing = col(info, "filing_date", "filingdate", "filing_dt", "acceptance_datetime") if info is not None else None
    i_report = col(info, "report_date", "reportdate", "period_end") if info is not None else None
    i_accession = col(info, "accession_number", "accessionnumber", "accession") if info is not None else None

    # Identifiers columns
    id_holding = col(identifiers, "holding_id", "holdingid", "id", "identifier") if identifiers is not None else None
    id_cusip = col(identifiers, "cusip", "issuer_cusip") if identifiers is not None else None
    id_isin = col(identifiers, "isin", "identifier_isin") if identifiers is not None else None
    id_lei = col(identifiers, "lei") if identifiers is not None else None
    id_title = col(identifiers, "issuer_name", "issuername", "title", "name") if identifiers is not None else None

    # Build base frame from holdings
    select_exprs: list[pl.Expr] = []
    select_exprs.append(pl.col(h_holding_id).cast(pl.String).alias("holding_id"))
    if h_series is not None:
        select_exprs.append(pl.col(h_series).cast(pl.String).alias("_series_id"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.String).alias("_series_id"))
    if h_report is not None:
        select_exprs.append(pl.col(h_report).alias("_report_raw"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.String).alias("_report_raw"))
    if h_value is not None:
        select_exprs.append(pl.col(h_value).cast(pl.Float64, strict=False).alias("value_usd"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Float64).alias("value_usd"))
    if h_weight is not None:
        select_exprs.append(pl.col(h_weight).cast(pl.Float64, strict=False).alias("weight_pct"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Float64).alias("weight_pct"))
    if h_issuer is not None:
        select_exprs.append(pl.col(h_issuer).cast(pl.String).alias("_issuer_hold"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.String).alias("_issuer_hold"))
    if h_cusip is not None:
        select_exprs.append(pl.col(h_cusip).cast(pl.String).alias("_cusip_hold"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.String).alias("_cusip_hold"))
    if h_isin is not None:
        select_exprs.append(pl.col(h_isin).cast(pl.String).alias("_isin_hold"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.String).alias("_isin_hold"))
    if h_lei is not None:
        select_exprs.append(pl.col(h_lei).cast(pl.String).alias("_lei_hold"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.String).alias("_lei_hold"))
    if h_accession is not None:
        select_exprs.append(pl.col(h_accession).cast(pl.String).alias("_accession"))
    if h_filing is not None:
        select_exprs.append(pl.col(h_filing).alias("_filing_hold_raw"))

    if series_map and h_accession is not None:
        target_accessions = _target_accessions(
            series_map,
            info,
            i_series=i_series,
            i_accession=i_accession,
        )
        if target_accessions:
            holding = holding.filter(pl.col(h_accession).cast(pl.String).is_in(list(target_accessions)))

    base = holding.select(select_exprs)

    # Join identifiers
    if identifiers is not None and id_holding is not None:
        id_select: list[pl.Expr] = [pl.col(id_holding).cast(pl.String).alias("holding_id")]
        if id_cusip is not None:
            id_select.append(pl.col(id_cusip).cast(pl.String).alias("_cusip_id"))
        if id_isin is not None:
            id_select.append(pl.col(id_isin).cast(pl.String).alias("_isin_id"))
        if id_lei is not None:
            id_select.append(pl.col(id_lei).cast(pl.String).alias("_lei_id"))
        if id_title is not None:
            id_select.append(pl.col(id_title).cast(pl.String).alias("_issuer_id"))
        id_frame = identifiers.select(id_select).unique(subset=["holding_id"])
        base = base.join(id_frame, on="holding_id", how="left")

    # Join info/submission for filing_date and report_date; preserve every amendment filing.
    joined_bulk = False
    if info is not None and i_accession is not None and h_accession is not None and "_accession" in base.columns:
        info_select: list[pl.Expr] = [pl.col(i_accession).cast(pl.String).alias("_accession")]
        if i_series is not None:
            info_select.append(pl.col(i_series).cast(pl.String).alias("_series_id_info"))
        info_frame = info.select(info_select).unique(subset=["_accession"])
        base = base.join(info_frame, on="_accession", how="left")
        if "_series_id_info" in base.columns:
            base = base.with_columns(pl.coalesce(["_series_id", "_series_id_info"]).alias("_series_id")).drop(
                "_series_id_info"
            )
        if submission_key is not None:
            submission = normed[submission_key]
            s_acc = col(submission, "accession_number", "accessionnumber", "accession")
            s_filing = col(submission, "filing_date", "filingdate", "filing_dt", "acceptance_datetime")
            s_report = col(submission, "report_date", "reportdate", "period_end", "report_ending_period")
            if s_acc is not None and s_filing is not None:
                sub_select: list[pl.Expr] = [
                    pl.col(s_acc).cast(pl.String).alias("_accession"),
                    pl.col(s_filing).alias("_filing_raw"),
                ]
                if s_report is not None:
                    sub_select.append(pl.col(s_report).alias("_report_info"))
                sub_frame = submission.select(sub_select).unique(subset=["_accession", "_filing_raw"])
                base = base.join(sub_frame, on="_accession", how="left")
                joined_bulk = True

    if not joined_bulk and info is not None and i_series is not None and i_filing is not None:
        info_select = [pl.col(i_series).cast(pl.String).alias("_series_id")]
        info_select.append(pl.col(i_filing).alias("_filing_raw"))
        if i_report is not None:
            info_select.append(pl.col(i_report).alias("_report_info"))
        if i_accession is not None:
            info_select.append(pl.col(i_accession).cast(pl.String).alias("_accession"))
        info_subset = ["_series_id", "_filing_raw"]
        if i_report is not None:
            info_subset.append("_report_info")
        if i_accession is not None:
            info_subset.append("_accession")
        info_frame = info.select(info_select).unique(subset=info_subset)
        if i_accession is not None and "_accession" in base.columns:
            base = base.join(info_frame, on=["_series_id", "_accession"], how="left")
        elif h_filing is not None:
            base = base.with_columns(pl.col("_filing_hold_raw").alias("_filing_raw"))
            info_for_filing = info_frame.with_columns(pl.col("_filing_raw").alias("_filing_hold_raw"))
            base = base.join(info_for_filing, on=["_series_id", "_filing_hold_raw"], how="left", suffix="_info")
            if "_filing_raw_info" in base.columns:
                base = base.with_columns(pl.coalesce(["_filing_raw", "_filing_raw_info"]).alias("_filing_raw")).drop(
                    "_filing_raw_info"
                )
        else:
            singleton_series = (
                info_frame.group_by("_series_id").len().filter(pl.col("len") == 1).select("_series_id")
            )
            base = base.join(
                info_frame.join(singleton_series, on="_series_id", how="inner"),
                on="_series_id",
                how="left",
            )

    # Prepare cusip/isin/lei/issuer coalesce
    if "_cusip_id" in base.columns and "_cusip_hold" in base.columns:
        base = base.with_columns(pl.coalesce(["_cusip_id", "_cusip_hold"]).alias("cusip"))
    elif "_cusip_id" in base.columns:
        base = base.with_columns(pl.col("_cusip_id").alias("cusip"))
    elif "_cusip_hold" in base.columns:
        base = base.with_columns(pl.col("_cusip_hold").alias("cusip"))
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.String).alias("cusip"))

    if "_isin_id" in base.columns and "_isin_hold" in base.columns:
        base = base.with_columns(pl.coalesce(["_isin_id", "_isin_hold"]).alias("isin"))
    elif "_isin_id" in base.columns:
        base = base.with_columns(pl.col("_isin_id").alias("isin"))
    elif "_isin_hold" in base.columns:
        base = base.with_columns(pl.col("_isin_hold").alias("isin"))
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.String).alias("isin"))

    if "_lei_id" in base.columns and "_lei_hold" in base.columns:
        base = base.with_columns(pl.coalesce(["_lei_id", "_lei_hold"]).alias("lei"))
    elif "_lei_id" in base.columns:
        base = base.with_columns(pl.col("_lei_id").alias("lei"))
    elif "_lei_hold" in base.columns:
        base = base.with_columns(pl.col("_lei_hold").alias("lei"))
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.String).alias("lei"))

    if "_issuer_id" in base.columns and "_issuer_hold" in base.columns:
        base = base.with_columns(pl.coalesce(["_issuer_id", "_issuer_hold"]).alias("issuer_name"))
    elif "_issuer_id" in base.columns:
        base = base.with_columns(pl.col("_issuer_id").alias("issuer_name"))
    elif "_issuer_hold" in base.columns:
        base = base.with_columns(pl.col("_issuer_hold").alias("issuer_name"))
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.String).alias("issuer_name"))

    # filing_date handling
    if "_filing_raw" in base.columns:
        base = base.with_columns(_parse_filing_datetime("_filing_raw").alias("filing_date"))
    elif "_filing_hold_raw" in base.columns:
        base = base.with_columns(_parse_filing_datetime("_filing_hold_raw").alias("filing_date"))
    else:
        base = base.with_columns(pl.lit(None, dtype=TS_DTYPE).alias("filing_date"))

    if "_report_info" in base.columns and "_report_raw" in base.columns:
        base = base.with_columns(
            _parse_report_date("_report_info").alias("_r1"),
            _parse_report_date("_report_raw").alias("_r2"),
        )
        base = base.with_columns(pl.coalesce(["_r1", "_r2"]).alias("report_date"))
    elif "_report_info" in base.columns:
        base = base.with_columns(_parse_report_date("_report_info").alias("report_date"))
    elif "_report_raw" in base.columns:
        base = base.with_columns(_parse_report_date("_report_raw").alias("report_date"))
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.Date).alias("report_date"))

    # Filter to series_map tickers only
    # Map series_id -> ticker
    # Build mapping expr
    if series_map:
        map_frame = pl.DataFrame({"_map_key": list(series_map.keys()), "etf_ticker": list(series_map.values())}).with_columns(
            pl.col("_map_key").cast(pl.String),
            pl.col("etf_ticker").cast(pl.String),
        )
        if "_accession" in base.columns:
            base = base.with_columns(pl.coalesce(["_series_id", "_accession"]).alias("_map_key"))
        else:
            base = base.with_columns(pl.col("_series_id").alias("_map_key"))
        base = base.join(map_frame, on="_map_key", how="inner")
    else:
        # No filtering, use series_id as ticker uppercased
        base = base.with_columns(pl.col("_series_id").cast(pl.String).str.to_uppercase().alias("etf_ticker"))
        # But then need to filter? No filter if empty map - keep all
    # Drop rows without required fields
    # Ensure filing_date not null, report_date not null
    base = base.filter(pl.col("filing_date").is_not_null() & pl.col("report_date").is_not_null())

    # Ensure weight_pct finite and in [0,100]
    base = base.with_columns(pl.col("weight_pct").cast(pl.Float64))
    base = base.filter(pl.col("weight_pct").is_not_null() & pl.col("weight_pct").is_finite() & (pl.col("weight_pct") >= 0) & (pl.col("weight_pct") <= 100))

    base = base.with_columns(
        pl.when(pl.col("cusip").is_in(list(_PLACEHOLDER_CUSIPS)))
        .then(None)
        .otherwise(pl.col("cusip"))
        .alias("cusip")
    )

    # Fill missing identifiers to null
    # Build final frame
    spec = spec_for(Dataset.ETF_HOLDINGS)
    out = base.select(
        pl.col("etf_ticker").cast(pl.String),
        pl.col("report_date").cast(pl.Date),
        pl.col("filing_date").cast(TS_DTYPE),
        pl.col("holding_id").cast(pl.String),
        pl.col("issuer_name").cast(pl.String),
        pl.col("cusip").cast(pl.String),
        pl.col("isin").cast(pl.String),
        pl.col("lei").cast(pl.String),
        pl.col("weight_pct").cast(pl.Float64),
        pl.col("value_usd").cast(pl.Float64),
        pl.lit(_SOURCE, dtype=pl.String).alias("source"),
        pl.lit(retrieved_at, dtype=TS_DTYPE).alias("retrieved_at"),
    )
    # Cast to spec schema
    out = out.cast(pl.Schema(dict(spec.columns)))
    # dedup key? Keep as is - pipeline will handle but we should dedup if duplicate key?
    return out


class SecNportClient:
    """SEC N-PORT bulk ZIP client (no live HTTP here; parsing only)."""

    @staticmethod
    def parse_quarter_zip(content: bytes, *, filing_quarter: str) -> pl.DataFrame:
        """Parse quarterly ZIP bytes into normalized holdings frame."""
        _ = filing_quarter  # quarter label retained for lineage; not used for parsing
        raw_tables = _parse_raw_tables(content)
        # Load default series map if exists else empty
        from pathlib import Path

        default_map_path = Path("configs/etf_metadata/nport_series_map.json")
        series_map: Mapping[str, str] = {}
        if default_map_path.is_file():
            import json

            try:
                doc = json.loads(default_map_path.read_text(encoding="utf-8"))
                if isinstance(doc, dict):
                    series_map = {str(k): str(v) for k, v in doc.items()}
            except Exception:
                series_map = {}
        retrieved_at = datetime.now(UTC)
        return normalize_nport_holdings(raw_tables, series_map=series_map, retrieved_at=retrieved_at)
