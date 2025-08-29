"""
title: Stock Market Helper
description: A comprehensive stock analysis tool that gathers data from Finnhub API and compiles a detailed report.
author: Sergii Zabigailo
author_url: https://github.com/sergezab/
github: https://github.com/sergezab/openwebui_tools/
funding_url: https://github.com/open-webui
version: 0.3.0
license: MIT
requirements: finnhub-python,pytz
"""

from __future__ import annotations

import os
import json
import time
import random
import logging
import shelve
from functools import lru_cache, wraps
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable, Union

import pytz
import finnhub
import yfinance as yf

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------
# Logging (safe, single init)
# ---------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:  # avoid duplicate handlers on reload
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_file = os.path.join(os.path.expanduser("~"), "stock_reporter.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

# ---------------------------
# Timezone / Market helpers
# ---------------------------

EST = pytz.timezone("US/Eastern")


def now_est() -> datetime:
    return datetime.now(EST)


def format_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def last_trading_day(ts: Optional[datetime] = None) -> datetime:
    cur = ts.astimezone(EST) if ts else now_est()
    # If before open, consider yesterday
    m_open = cur.replace(hour=9, minute=30, second=0, microsecond=0)
    day = cur - timedelta(days=1) if cur < m_open else cur
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def is_market_open(ts: Optional[datetime] = None) -> Tuple[bool, str]:
    now = ts.astimezone(EST) if ts else now_est()
    if now.weekday() >= 5:
        return False, "Market is closed (Weekend)"
    m_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    m_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now < m_open:
        return False, f"Opens 9:30 AM EST (now {now.strftime('%I:%M %p')} EST)"
    if now > m_close:
        return False, f"Closed 4:00 PM EST (now {now.strftime('%I:%M %p')} EST)"
    return True, "Market is open"


# ---------------------------
# Retry with backoff
# ---------------------------


def retry_with_backoff(max_retries=3, initial_backoff=1.0, max_backoff=15.0):
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            backoff = initial_backoff
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    msg = str(e)
                    retriable = any(
                        s in msg.lower()
                        for s in ["429", "limit", "timeout", "connection", "504", "502"]
                    )
                    if not retriable or attempt == max_retries:
                        raise
                    sleep_s = min(
                        backoff * (3 if "429" in msg else 1), max_backoff
                    ) + random.uniform(0, 1)
                    logger.warning(
                        f"{fn.__name__} failed (attempt {attempt}/{max_retries}): {msg}. Retrying in {sleep_s:.1f}s"
                    )
                    time.sleep(sleep_s)
                    backoff = min(backoff * 2, max_backoff)

        return wrapper

    return decorator


# ---------------------------
# Finnhub calls (wrapped)
# ---------------------------


@retry_with_backoff()
def fh_profile(client: finnhub.Client, ticker: str) -> Dict[str, Any]:
    return client.company_profile2(symbol=ticker, timeout=10)


@retry_with_backoff()
def fh_financials(client: finnhub.Client, ticker: str) -> Dict[str, Any]:
    return client.company_basic_financials(ticker, "all")


@retry_with_backoff()
def fh_peers(client: finnhub.Client, ticker: str) -> List[str]:
    return client.company_peers(ticker)


@retry_with_backoff()
def fh_quote(client: finnhub.Client, ticker: str) -> Dict[str, Any]:
    return client.quote(ticker, timeout=10)


@retry_with_backoff()
def fh_news(
    client: finnhub.Client, ticker: str, start: str, end: str
) -> List[Dict[str, Any]]:
    return client.company_news(ticker, start, end, timeout=10)


# ---------------------------
# Sentiment model (cached)
# ---------------------------


@lru_cache(maxsize=1)
def get_sentiment_model():
    name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    device = 0 if torch.cuda.is_available() else -1
    return tok, mdl, device


# ---------------------------
# Utilities
# ---------------------------


def is_html_response(resp: Any) -> bool:
    return isinstance(resp, str) and (
        "<html" in resp.lower() or "<!doctype" in resp.lower()
    )


def is_similar(a: str, b: str, threshold: float = 0.9) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    # lightweight similarity (fast) – no difflib at runtime
    if not a or not b:
        return False
    if a == b:
        return True
    # fallback: prefix or containment quick check
    return a.startswith(b[:20]) or b.startswith(a[:20]) or (a in b) or (b in a)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_num_str(v: Any, fmt: str = ".2f", default: str = "N/A") -> str:
    try:
        return f"{float(v):{fmt}}"
    except Exception:
        return default


def format_pct(v: Any) -> str:
    try:
        x = float(v)
        return f"{x:.2f}%" if abs(x) > 1 else f"{x*100:.2f}%"
    except Exception:
        return "N/A"


def format_currency(v: Any, suffix: bool = True) -> str:
    try:
        x = float(v)
        if suffix:
            if x >= 1_000_000_000:
                return f"${x/1_000_000_000:.2f}B"
            if x >= 1_000_000:
                return f"${x/1_000_000:.2f}M"
        return f"${x:,.2f}"
    except Exception:
        return "N/A"


def format_unix_date(ts: Any) -> str:
    try:
        val = int(ts)
        return datetime.fromtimestamp(val).strftime("%B %d, %Y")
    except Exception:
        return "N/A"


# ---------------------------
# Cache freshness
# ---------------------------


def cache_is_today(bucket: Dict[str, Any], key: str) -> bool:
    if not bucket or key not in bucket:
        return False
    try:
        ts = datetime.fromisoformat(bucket[key]["timestamp"])
        if ts.tzinfo is None:
            ts = EST.localize(ts)
        return ts.date() == now_est().date()
    except Exception:
        return False


def price_cache_stale(ts: datetime) -> bool:
    if ts.tzinfo is None:
        ts = EST.localize(ts)
    now = now_est()
    last_td = last_trading_day(now)
    if ts.date() < last_td.date():
        return True
    m_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    m_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    during_hours = (m_open <= now <= m_close) and now.weekday() < 5
    if during_hours:
        return (now - ts).total_seconds() > 60 * 60
    if ts.date() == now.date():
        return False
    # If from last trading day, ensure after close
    last_close = ts.replace(hour=16, minute=0, second=0, microsecond=0)
    return ts < last_close


# ---------------------------
# Classification helpers
# ---------------------------

COMMON_ETFS = {
    "spy",
    "qqq",
    "voo",
    "dia",
    "iwm",
    "vti",
    "gld",
    "xlf",
    "xle",
    "xlu",
    "vnq",
    "kweb",
    "ihi",
    "ewz",
    "bnd",
    "vusxx",
    "vwo",
    "botz",
    "snsxx",
}


def classify_security(profile: Dict[str, Any]) -> Tuple[bool, str]:
    """Returns (is_fund, type_str)"""
    if not profile:
        return False, "Stock"
    name = (profile.get("name") or "").lower()
    ticker = (profile.get("ticker") or "").lower()
    stype = (profile.get("type") or "").lower()

    if any(
        k in name for k in ["etf", "exchange traded fund", "index fund", "index trust"]
    ) or ticker.endswith("etf"):
        return True, "ETF"
    if any(
        k in name
        for k in ["money market", "cash reserves", "liquidity fund", "treasury", "govt"]
    ) or ticker.endswith("mm"):
        return True, "Money Market Fund"
    if stype and ("etf" in stype or "fund" in stype):
        return True, "ETF"
    if ticker in COMMON_ETFS:
        return (True, "Money Market Fund") if "snsxx" in ticker else (True, "ETF")
    return False, "Stock"


# ---------------------------
# Finnhub data shaping
# ---------------------------


def get_basic_info(client: finnhub.Client, ticker: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "profile": {"ticker": ticker, "name": ticker},
        "basic_financials": {"metric": {}},
        "peers": [],
    }
    try:
        prof = fh_profile(client, ticker)
        if isinstance(prof, dict) and not is_html_response(prof):
            prof.setdefault("ticker", ticker)
            prof.setdefault("name", ticker)
            out["profile"] = prof
        fin = fh_financials(client, ticker)
        if isinstance(fin, dict) and "metric" in fin:
            out["basic_financials"] = fin
        peers = fh_peers(client, ticker)
        if isinstance(peers, list):
            out["peers"] = peers
    except Exception as e:
        logger.warning(f"basic info fallback for {ticker}: {e}")
    return out


def get_current_price(client: finnhub.Client, ticker: str) -> Optional[Dict[str, Any]]:
    try:
        q = fh_quote(client, ticker)
        c, dp, pc = q.get("c", 0.0), q.get("dp", 0.0), q.get("pc", 0.0)
        suspicious = abs(dp) > 1000 or pc < 0.10 or (pc and c > pc * 50)
        if suspicious:
            logger.warning(f"Suspicious quote for {ticker}: c={c}, pc={pc}, dp={dp}")
            return {
                "current_price": c,
                "change": 0.0,
                "change_amount": 0.0,
                "high": q.get("h", 0.0),
                "low": q.get("l", 0.0),
                "open": q.get("o", 0.0),
                "previous_close": pc,
                "data_warning": "Potentially stale/delisted",
            }
        return {
            "current_price": c,
            "change": dp,
            "change_amount": q.get("d", 0.0),
            "high": q.get("h", 0.0),
            "low": q.get("l", 0.0),
            "open": q.get("o", 0.0),
            "previous_close": pc,
        }
    except Exception as e:
        logger.error(f"price error {ticker}: {e}")
        return None


def get_company_news(client: finnhub.Client, ticker: str) -> List[Dict[str, str]]:
    try:
        end = now_est()
        start = end - timedelta(days=7)
        items = fh_news(client, ticker, format_date(start), format_date(end))[:10]
        out: List[Dict[str, str]] = []
        for it in items:
            url = it.get("url") or ""
            title = it.get("headline") or ""
            summary = it.get("summary") or ""
            out.append({"url": url, "title": title, "summary": summary})
        return out
    except Exception as e:
        logger.warning(f"news error {ticker}: {e}")
        return []


# ---------------------------
# Sentiment (async-friendly)
# ---------------------------


def analyze_sentiment(summary: str, title: str) -> Dict[str, Any]:
    try:
        tok, mdl, device = get_sentiment_model()
        # keyword prior
        text = f"Headline: {title}\nSummary: {summary}"
        lower = text.lower()
        neg_words = [
            "weak",
            "concern",
            "risk",
            "overvalued",
            "bearish",
            "sell",
            "short",
            "downgrade",
            "warning",
            "decline",
            "drop",
            "plunge",
            "miss",
            "failed",
            "disappointing",
            "caution",
        ]
        pos_words = [
            "strong",
            "buy",
            "bullish",
            "undervalued",
            "upgrade",
            "outperform",
            "beat",
            "growth",
            "positive",
            "rally",
            "surge",
            "breakthrough",
            "leader",
        ]
        neg = sum(w in lower for w in neg_words)
        pos = sum(w in lower for w in pos_words)
        base = (
            ("negative", min(0.9, neg / max(1, neg + pos)))
            if neg > pos
            else (
                ("positive", min(0.9, pos / max(1, neg + pos)))
                if pos > neg
                else ("neutral", 0.5)
            )
        )

        inputs = tok(lower, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = mdl(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[
            0
        ]  # [neutral, positive, negative] for this model
        labels = getattr(
            mdl.config, "id2label", {0: "neutral", 1: "positive", 2: "negative"}
        )
        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        model_sent = labels[pred_idx].lower()
        model_conf = float(max(probs))

        # blend
        if base[0] == model_sent:
            sent, conf = model_sent, (base[1] + model_conf) / 2
        else:
            sent, conf = base if base[1] > model_conf else (model_sent, model_conf)
            conf *= 0.8  # penalty on disagreement

        return {
            "sentiment": sent,
            "confidence": float(conf),
            "probabilities": [[round(p, 5) for p in probs]],
        }
    except Exception as e:
        logger.warning(f"sentiment fallback: {e}")
        return {"sentiment": "neutral", "confidence": 0.0, "probabilities": []}


# ---------------------------
# ETF helpers (yfinance)
# ---------------------------


def load_etf_extras(ticker: str) -> Dict[str, Any]:
    try:
        yf_t = yf.Ticker(ticker)
        info = getattr(yf_t, "info", {}) or {}
        return {
            "expense_ratio": info.get("expenseRatio"),
            "category": info.get("category"),
            "aum": info.get("totalAssets"),
            "nav": info.get("navPrice"),
            "inception_date": info.get("fundInceptionDate"),
            "yield": info.get("yield"),
            "ytd_return": info.get("ytdReturn"),
            "three_year_return": info.get("threeYearAverageReturn"),
            "five_year_return": info.get("fiveYearAverageReturn"),
            "seven_day_yield": info.get("sevenDayYield"),
            "weighted_avg_maturity": info.get("weightedAverageMaturity"),
        }
    except Exception as e:
        logger.warning(f"yfinance etf extras error {ticker}: {e}")
        return {}


# ---------------------------
# Data gather (per ticker)
# ---------------------------


async def gather_stock_data(
    client: finnhub.Client, ticker: str, cache: shelve.Shelf
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "basic_info": {
            "profile": {"ticker": ticker, "name": ticker},
            "basic_financials": {"metric": {}},
            "peers": [],
        },
        "current_price": {
            "current_price": 0.0,
            "change": 0.0,
            "change_amount": 0.0,
            "high": 0.0,
            "low": 0.0,
            "open": 0.0,
            "previous_close": 0.0,
        },
        "sentiments": [],
        "is_etf": False,
        "fund_type": "Stock",
        "etf_data": {},
    }

    tcache = cache.get(ticker, {})

    # basic info
    if not cache_is_today(tcache, "basic_info"):
        logger.info(f"Fetching basic info for {ticker}")
        basic = get_basic_info(client, ticker)
        tcache["basic_info"] = {"data": basic, "timestamp": now_est().isoformat()}
    else:
        basic = tcache["basic_info"]["data"]
        logger.info(f"Using cached basic info for {ticker}")
    result["basic_info"] = basic

    # classify
    is_fund, fund_type = classify_security(basic.get("profile", {}))
    result["is_etf"] = is_fund
    result["fund_type"] = fund_type

    # etf extras (cache)
    if is_fund:
        if cache_is_today(tcache, "etf_data"):
            result["etf_data"] = tcache["etf_data"]["data"]
        else:
            extras = load_etf_extras(ticker)
            tcache["etf_data"] = {"data": extras, "timestamp": now_est().isoformat()}
            result["etf_data"] = extras

    # price
    use_cached = False
    if "current_price" in tcache:
        ts = datetime.fromisoformat(tcache["current_price"]["timestamp"])
        ts = EST.localize(ts) if ts.tzinfo is None else ts
        use_cached = not price_cache_stale(ts)
        if use_cached:
            logger.info(f"Using cached price for {ticker}")
    if use_cached:
        price = tcache["current_price"]["data"]
    else:
        logger.info(f"Fetching price for {ticker}")
        price = get_current_price(client, ticker)
        if price is None and is_fund:
            # yfinance fallback for funds
            try:
                yf_t = yf.Ticker(ticker)
                hist = yf_t.history(period="2d")
                if not hist.empty:
                    last = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else last
                    price = {
                        "current_price": float(last["Close"]),
                        "change": (
                            float((last["Close"] / prev["Close"] - 1) * 100)
                            if float(prev["Close"])
                            else 0.0
                        ),
                        "change_amount": float(last["Close"] - prev["Close"]),
                        "high": float(last["High"]),
                        "low": float(last["Low"]),
                        "open": float(last["Open"]),
                        "previous_close": float(prev["Close"]),
                    }
            except Exception as e:
                logger.warning(f"yfinance price fallback error {ticker}: {e}")
        if price is None:
            price = {
                "current_price": 0.0,
                "change": 0.0,
                "change_amount": 0.0,
                "high": 0.0,
                "low": 0.0,
                "open": 0.0,
                "previous_close": 0.0,
            }
        else:
            tcache["current_price"] = {
                "data": price,
                "timestamp": now_est().isoformat(),
            }
    result["current_price"] = price

    # news + sentiments
    if not cache_is_today(tcache, "sentiments"):
        news = get_company_news(client, ticker)
        news = [n for n in news if n.get("title")]
        items: List[Dict[str, Any]] = []
        for n in news:
            title, summary = n.get("title", ""), n.get("summary", "")
            effective_summary = "" if is_similar(title, summary) else summary
            sent = analyze_sentiment(effective_summary, title)
            items.append(
                {
                    "url": n.get("url", ""),
                    "title": title or "No title",
                    "summary": effective_summary,
                    "sentiment": sent.get("sentiment", "neutral"),
                    "confidence": sent.get("confidence", 0.0),
                    "probabilities": sent.get("probabilities", []),
                }
            )
        tcache["sentiments"] = {"data": items, "timestamp": now_est().isoformat()}
        result["sentiments"] = items
    else:
        result["sentiments"] = tcache["sentiments"]["data"]

    cache[ticker] = tcache
    return result


# ---------------------------
# Analysis / Reports
# ---------------------------


def assess_financial_health(
    metrics: Dict[str, Any], industry: Optional[str] = None
) -> str:
    try:
        ind = (industry or "").lower()
        kind = (
            "retail"
            if "retail" in ind
            else (
                "tech"
                if "technology" in ind
                else (
                    "utilities"
                    if "utilities" in ind
                    else (
                        "healthcare"
                        if ("healthcare" in ind or "health" in ind)
                        else (
                            "energy"
                            if any(k in ind for k in ["energy", "oil", "gas"])
                            else (
                                "financial"
                                if any(k in ind for k in ["financial", "bank"])
                                else "default"
                            )
                        )
                    )
                )
            )
        )

        th = {
            "retail": {
                "profit_margin": (3, 2),
                "current_ratio": (1.2, 0.8),
                "inventory_turnover": (10, 6),
                "debt_equity": (0.5, 1.0),
            },
            "tech": {
                "profit_margin": (20, 15),
                "current_ratio": (1.5, 1.2),
                "debt_equity": (0.3, 0.6),
            },
            "utilities": {
                "profit_margin": (12, 8),
                "current_ratio": (1.0, 0.8),
                "debt_equity": (1.2, 1.5),
                "interest_coverage": (3, 2),
            },
            "healthcare": {
                "profit_margin": (15, 10),
                "current_ratio": (1.5, 1.2),
                "debt_equity": (0.4, 0.8),
            },
            "energy": {
                "profit_margin": (10, 6),
                "current_ratio": (1.2, 1.0),
                "debt_equity": (0.6, 1.0),
            },
            "financial": {"roe": (15, 10), "debt_equity": (3.0, 4.0)},
            "default": {
                "profit_margin": (15, 10),
                "current_ratio": (1.5, 1.2),
                "debt_equity": (0.5, 1.0),
            },
        }[kind]

        # normalize inputs
        roe = safe_float(metrics.get("roeTTM"))
        cur = safe_float(metrics.get("currentRatioQuarterly"))
        pm = safe_float(metrics.get("netProfitMarginTTM"))
        de = safe_float(metrics.get("totalDebt/totalEquityQuarterly"))
        ic = safe_float(metrics.get("netInterestCoverageTTM"))
        it = safe_float(metrics.get("inventoryTurnoverTTM"))

        pts = 0
        max_pts = 0

        # PM
        if "profit_margin" in th:
            hi, mid = th["profit_margin"]
            max_pts += 2
            if pm > hi:
                pts += 2
            elif pm > mid:
                pts += 1
        # ROE
        hi, mid = (20, 15)
        if "roe" in th:
            hi, mid = th["roe"]
        max_pts += 2
        if roe > hi:
            pts += 2
        elif roe > mid:
            pts += 1
        # Current ratio
        if "current_ratio" in th:
            hi, _ = th["current_ratio"]
            max_pts += 1
            if cur > hi:
                pts += 1
        # D/E
        if "debt_equity" in th:
            low, med = th["debt_equity"]
            max_pts += 2
            if de < low:
                pts += 2
            elif de < med:
                pts += 1
        # Interest coverage
        hi, mid = th.get("interest_coverage", (50, 20))
        max_pts += 2
        if ic > hi:
            pts += 2
        elif ic > mid:
            pts += 1
        # Retail extra
        if kind == "retail" and "inventory_turnover" in th:
            hi, _ = th["inventory_turnover"]
            max_pts += 1
            if it > hi:
                pts += 1

        score = (pts / max_pts) * 100 if max_pts else 0
        label = "strong" if score >= 70 else "moderate" if score >= 40 else "weak"
        return f"{label} {round(score)}% - {industry or ''}".strip()
    except Exception as e:
        logger.error(f"health error: {e}")
        return "moderate"


def compile_report_human(data: Dict[str, Any]) -> str:
    prof = data.get("basic_info", {}).get("profile", {})
    metrics = data.get("basic_info", {}).get("basic_financials", {}).get("metric", {})
    price = data.get("current_price", {})
    ticker = prof.get("ticker", "Unknown")
    name = prof.get("name", ticker)

    is_fund, ftype = classify_security(prof)
    market_cap = prof.get("marketCapitalization")
    mc_display = format_currency((market_cap or 0) * 1_000_000) if market_cap else "N/A"

    if is_fund:
        etf = data.get("etf_data", {})
        report = [
            f"Investment Analysis: {name} ({ticker}) - {ftype}",
            f"Fund Overview:",
            f"• Type: {ftype} | Category: {prof.get('finnhubIndustry','Investment')}",
            f"• Current Price: ${safe_num_str(price.get('current_price'))} ({safe_num_str(price.get('change'))}%)",
            f"• Daily Range: ${safe_num_str(price.get('low'))} - ${safe_num_str(price.get('high'))}",
            "",
            "Performance:",
            f"• 1-Day Change: {safe_num_str(price.get('change'))}%",
            f"• Previous Close: ${safe_num_str(price.get('previous_close'))}",
        ]
        if ftype == "ETF":
            report += [
                "",
                "ETF Details:",
                f"• Expense Ratio: {format_pct(etf.get('expense_ratio'))}",
                f"• Yield: {format_pct(etf.get('yield'))}",
                f"• YTD Return: {format_pct(etf.get('ytd_return'))}",
                f"• 3 Year Return: {format_pct(etf.get('three_year_return'))}",
                f"• 5 Year Return: {format_pct(etf.get('five_year_return'))}",
                f"• Assets Under Management: {format_currency(etf.get('aum'))}",
                f"• Net Asset Value: {format_currency(etf.get('nav'))}",
                f"• Inception Date: {format_unix_date(etf.get('inception_date'))}",
                "",
                "Note: ETF metrics differ from single-stock fundamentals.",
            ]
        elif ftype == "Money Market Fund":
            report += [
                "",
                "Money Market Fund Details:",
                f"• Yield: {format_pct(etf.get('yield'))}",
                f"• 7-Day Yield: {format_pct(etf.get('seven_day_yield'))}",
                f"• YTD Return: {format_pct(etf.get('ytd_return'))}",
                f"• Weighted Avg Maturity: {etf.get('weighted_avg_maturity','N/A')}",
                f"• Total Net Assets: {format_currency(etf.get('aum'))}",
                "",
                "Note: MMFs focus on stability and liquidity, not typical stock ratios.",
            ]
    else:
        try:
            health = assess_financial_health(metrics, prof.get("finnhubIndustry", ""))
        except Exception:
            health = "moderate"
        report = [
            f"Stock Analysis: {name} ({ticker})",
            "",
            "Company Overview:",
            f"• {prof.get('finnhubIndustry','N/A')} | Market Cap: {mc_display} | Country: {prof.get('country','N/A')}",
            f"• Current Price: ${safe_num_str(price.get('current_price'))} ({safe_num_str(price.get('change'))}%) | YTD: {safe_num_str(metrics.get('yearToDatePriceReturnDaily'))}%",
            f"• 52W Range: ${safe_num_str(metrics.get('52WeekLow'))} - ${safe_num_str(metrics.get('52WeekHigh'))} | Beta: {safe_num_str(metrics.get('beta'))}",
            "",
            "Key Performance Indicators:",
            f"• Growth (5Y): Revenue {safe_num_str(metrics.get('revenueGrowth5Y'))}% | EPS {safe_num_str(metrics.get('epsGrowth5Y'))}%",
            f"• Margins: Gross {safe_num_str(metrics.get('grossMarginTTM'))}% | Operating {safe_num_str(metrics.get('operatingMarginTTM'))}% | Net {safe_num_str(metrics.get('netProfitMarginTTM'))}%",
            f"• Returns: ROE {safe_num_str(metrics.get('roeTTM'))}% | ROA {safe_num_str(metrics.get('roaTTM'))}%",
            "",
            "Financial Health:",
            f"• Liquidity: Current {safe_num_str(metrics.get('currentRatioQuarterly'))} | Quick {safe_num_str(metrics.get('quickRatioQuarterly'))}",
            f"• Leverage: D/E {safe_num_str(metrics.get('totalDebt/totalEquityQuarterly'))} | Interest Coverage {safe_num_str(metrics.get('netInterestCoverageTTM'))}x",
            f"• Per Share: EPS ${safe_num_str(metrics.get('epsTTM'))} | Book ${safe_num_str(metrics.get('bookValuePerShareQuarterly'))}",
            "",
            "Valuation:",
            f"• Multiples: P/E {safe_num_str(metrics.get('peTTM'))} | P/B {safe_num_str(metrics.get('pbQuarterly'))} | P/S {safe_num_str(metrics.get('psTTM'))}",
            f"• Dividend Yield: {safe_num_str(metrics.get('dividendYieldIndicatedAnnual'))}% | Payout Ratio: {safe_num_str(metrics.get('payoutRatioTTM'))}%",
            "",
            f"Summary Analysis: • Health Score: {health}",
        ]
        # strengths/risks
        strengths, risks = [], []
        try:
            if safe_float(metrics.get("revenueGrowth5Y")) > 10:
                strengths.append("High Growth")
            if safe_float(metrics.get("netProfitMarginTTM")) > 15:
                strengths.append("Strong Margins")
            if safe_float(metrics.get("roeTTM")) > 15:
                strengths.append("Solid ROE")
            if safe_float(metrics.get("totalDebt/totalEquityQuarterly")) < 0.5:
                strengths.append("Low Leverage")
            if safe_float(metrics.get("revenueGrowth5Y")) < 0:
                risks.append("Negative Growth")
            if safe_float(metrics.get("netProfitMarginTTM")) < 5:
                risks.append("Low Margins")
            if safe_float(metrics.get("totalDebt/totalEquityQuarterly")) > 2:
                risks.append("High Leverage")
            if safe_float(metrics.get("beta")) > 2:
                risks.append("High Beta")
        except Exception:
            pass
        report.append(
            f"• Key Strengths: {', '.join(strengths) if strengths else 'None identified'}"
        )
        report.append(
            f"• Key Risks: {', '.join(risks) if risks else 'None identified'}"
        )

    # sentiment summary (both types)
    sentiments = data.get("sentiments", [])
    if sentiments:
        pos = sum(
            1
            for s in sentiments
            if isinstance(s, dict) and s.get("sentiment") == "positive"
        )
        try:
            w_pos = sum(
                float(s.get("confidence", 0))
                for s in sentiments
                if s.get("sentiment") == "positive"
            )
            w_all = sum(float(s.get("confidence", 0)) for s in sentiments)
            ratio = (w_pos / w_all) if w_all else 0.5
            overall = (
                "Bullish" if ratio > 0.6 else "Bearish" if ratio < 0.4 else "Neutral"
            )
        except Exception:
            overall = "Neutral"
        report += [
            "",
            "Recent News Sentiment:",
            f"• Overall: {overall} ({pos}/{len(sentiments)} positive)",
        ]
        for s in sentiments:
            if not isinstance(s, dict):
                continue
            t = s.get("title", "No title")
            summ = s.get("summary", "")
            report.append(
                f"• {s.get('sentiment','neutral')} - {t}{(' - '+summ) if summ else ''}"
            )

    return "\n".join(report)


# token-optimized human output
def compile_report_optimized(data: Dict[str, Any]) -> str:
    prof = data.get("basic_info", {}).get("profile", {})
    metrics = data.get("basic_info", {}).get("basic_financials", {}).get("metric", {})
    price = data.get("current_price", {})
    ticker = prof.get("ticker", "UNK")
    name = prof.get("name", ticker)
    is_fund, ftype = classify_security(prof)

    def compact_cap(x):
        try:
            v = float(x)
            if v >= 1e12:
                return f"{v/1e12:.1f}T"
            if v >= 1e9:
                return f"{v/1e9:.1f}B"
            if v >= 1e6:
                return f"{v/1e6:.1f}M"
            if v >= 1e3:
                return f"{v/1e3:.1f}K"
            return f"{v:.2f}"
        except Exception:
            return "N/A"

    mc = prof.get("marketCapitalization")
    mc_display = compact_cap(mc * 1_000_000) if mc else "N/A"

    if is_fund:
        etf = data.get("etf_data", {})
        out = [
            f"{name} ({ticker}) - {ftype}",
            f"Price ${safe_num_str(price.get('current_price'))} ({safe_num_str(price.get('change'))}%) | Range ${safe_num_str(price.get('low'))}-${safe_num_str(price.get('high'))}",
            f"Industry {prof.get('finnhubIndustry','Investment')}",
        ]
        if etf:
            if ftype == "ETF":
                out.append(
                    f"Expense {format_pct(etf.get('expense_ratio'))} | Yield {format_pct(etf.get('yield'))} | AUM {format_currency(etf.get('aum'))} | YTD {format_pct(etf.get('ytd_return'))}"
                )
            else:
                out.append(
                    f"Yield {format_pct(etf.get('yield'))} | 7D {format_pct(etf.get('seven_day_yield'))} | Assets {format_currency(etf.get('aum'))}"
                )
    else:
        try:
            health = assess_financial_health(metrics, prof.get("finnhubIndustry", ""))
        except Exception:
            health = "moderate"
        out = [
            f"{name} ({ticker})",
            f"${safe_num_str(price.get('current_price'))} ({safe_num_str(price.get('change'))}%) | Cap ${mc_display} | Beta {safe_num_str(metrics.get('beta'))}",
            f"Growth Rev {safe_num_str(metrics.get('revenueGrowth5Y'))}% EPS {safe_num_str(metrics.get('epsGrowth5Y'))}% | YTD {safe_num_str(metrics.get('yearToDatePriceReturnDaily'))}%",
            f"Margins {safe_num_str(metrics.get('grossMarginTTM'))}%/{safe_num_str(metrics.get('operatingMarginTTM'))}%/{safe_num_str(metrics.get('netProfitMarginTTM'))}% (G/O/N)",
            f"Ratios P/E {safe_num_str(metrics.get('peTTM'))} P/B {safe_num_str(metrics.get('pbQuarterly'))} Current {safe_num_str(metrics.get('currentRatioQuarterly'))} | D/E {safe_num_str(metrics.get('totalDebt/totalEquityQuarterly'))} | Health {health}",
        ]
        strengths, risks = [], []
        try:
            if safe_float(metrics.get("revenueGrowth5Y")) > 10:
                strengths.append("Growth")
            if safe_float(metrics.get("netProfitMarginTTM")) > 15:
                strengths.append("Margins")
            if safe_float(metrics.get("roeTTM")) > 15:
                strengths.append("ROE")
            if safe_float(metrics.get("totalDebt/totalEquityQuarterly")) < 0.5:
                strengths.append("Low Debt")
            if safe_float(metrics.get("revenueGrowth5Y")) < 0:
                risks.append("Rev Decline")
            if safe_float(metrics.get("netProfitMarginTTM")) < 5:
                risks.append("Low Margins")
            if safe_float(metrics.get("totalDebt/totalEquityQuarterly")) > 2:
                risks.append("High Debt")
            if safe_float(metrics.get("beta")) > 2:
                risks.append("High Beta")
        except Exception:
            pass
        if strengths or risks:
            out.append(
                f"+: {', '.join(strengths) if strengths else 'None'} | -: {', '.join(risks) if risks else 'None'}"
            )

    sentiments = data.get("sentiments", [])
    if sentiments:
        pos = sum(
            1
            for s in sentiments
            if isinstance(s, dict) and s.get("sentiment") == "positive"
        )
        label = (
            "Bullish"
            if pos > len(sentiments) / 2
            else "Bearish" if pos < len(sentiments) / 3 else "Neutral"
        )
        out.append(f"News {label} ({pos}/{len(sentiments)} pos)")
    return "\n".join(out)


# ultra-compact, LLM-friendly line
def compile_report_compact(data: Dict[str, Any]) -> str:
    def fmt_num(v: Any, dec=1) -> str:
        try:
            s = f"{float(v):.{dec}f}"
            return s.rstrip("0").rstrip(".")
        except Exception:
            return ""

    def s_metric(m: Dict[str, Any], k: str) -> Optional[float]:
        try:
            return float(m.get(k))
        except Exception:
            return None

    def news_score(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "0.5"
        try:
            pos = sum(
                1
                for x in items
                if isinstance(x, dict) and x.get("sentiment") == "positive"
            )
            return f"{pos/len(items):.2f}".rstrip("0").rstrip(".")
        except Exception:
            return "0.5"

    prof = data.get("basic_info", {}).get("profile", {}) or {}
    metrics = (
        data.get("basic_info", {}).get("basic_financials", {}).get("metric", {}) or {}
    )
    price = data.get("current_price", {}) or {}
    sentiments = data.get("sentiments", []) or []

    is_fund, _ = classify_security(prof)
    if is_fund:
        etf = data.get("etf_data", {}) or {}
        er = (
            fmt_num((etf.get("expense_ratio") or 0) * 100)
            if etf.get("expense_ratio")
            else ""
        )
        yld = fmt_num((etf.get("yield") or 0) * 100) if etf.get("yield") else ""
        ytd = (
            fmt_num((etf.get("ytd_return") or 0) * 100) if etf.get("ytd_return") else ""
        )
        nav = fmt_num(etf.get("nav"))
        aum_m = ""
        if etf.get("aum") not in (None, "N/A"):
            try:
                aum_m = fmt_num(float(etf["aum"]) / 1_000_000.0)
            except Exception:
                aum_m = ""
        cat = (prof.get("finnhubIndustry") or "")[:6].upper()
        fields = [
            "F",
            prof.get("ticker", "UNK"),
            fmt_num(price.get("current_price")),
            fmt_num(price.get("change")),
            er,
            yld,
            ytd,
            cat,
            nav,
            aum_m,
            news_score(sentiments),
            "0.5",
        ]
        return "|".join(fields)
    else:
        pe = fmt_num(s_metric(metrics, "peTTM"))
        roe = fmt_num(s_metric(metrics, "roeTTM"))
        npm = fmt_num(s_metric(metrics, "netProfitMarginTTM"))
        rev5 = fmt_num(s_metric(metrics, "revenueGrowth5Y"))
        de = fmt_num(s_metric(metrics, "totalDebt/totalEquityQuarterly"))
        beta = fmt_num(s_metric(metrics, "beta"))
        # derive health -> 0..1 from label
        try:
            health_lbl = assess_financial_health(
                metrics, prof.get("finnhubIndustry", "")
            ).lower()
            h = (
                "0.8"
                if "strong" in health_lbl
                else "0.2" if "weak" in health_lbl else "0.5"
            )
        except Exception:
            h = "0.5"
        fields = [
            "S",
            prof.get("ticker", "UNK"),
            fmt_num(price.get("current_price")),
            fmt_num(price.get("change")),
            pe,
            roe,
            npm,
            rev5,
            de,
            beta,
            news_score(sentiments),
            h,
        ]
        return "|".join(fields)


# ---------------------------
# Tool interface
# ---------------------------

from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        FINNHUB_API_KEY: str = Field(default="", description="Required Finnhub API key")

    def __init__(self):
        self.citation = True
        self.valves = self.Valves()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "compile_stock_report",
                    "description": "BATCH-ONLY. Comprehensive stock analysis. Pass ALL symbols via `tickers`.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tickers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of tickers, e.g. ['AAPL','MSFT']",
                            },
                            "ticker": {
                                "type": "string",
                                "description": "(Legacy) CSV like 'AAPL,MSFT'. Prefer `tickers`.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_summary",
                    "description": "BATCH-ONLY. Token-optimized, human-readable summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tickers": {"type": "array", "items": {"type": "string"}},
                            "ticker": {
                                "type": "string",
                                "description": "(Legacy) CSV; prefer `tickers`.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_data_compact",
                    "description": "BATCH-ONLY. Ultra-compact structured line(s) for LLMs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tickers": {"type": "array", "items": {"type": "string"}},
                            "ticker": {
                                "type": "string",
                                "description": "(Legacy) CSV; prefer `tickers`.",
                            },
                        },
                    },
                },
            },
        ]

    # ---- public fns ----

    async def compile_stock_report(
        self,
        tickers: Optional[List[str]] = None,
        ticker: str = "",
        __user__={},
        __event_emitter__=None,
    ) -> str:
        syms = self._normalize_symbols(tickers, ticker)
        return await self._execute(
            syms, mode="full", __user__=__user__, __event_emitter__=__event_emitter__
        )

    async def get_stock_summary(
        self,
        tickers: Optional[List[str]] = None,
        ticker: str = "",
        __user__={},
        __event_emitter__=None,
    ) -> str:
        syms = self._normalize_symbols(tickers, ticker)
        return await self._execute(
            syms,
            mode="optimized",
            __user__=__user__,
            __event_emitter__=__event_emitter__,
        )

    async def get_stock_data_compact(
        self,
        tickers: Optional[List[str]] = None,
        ticker: str = "",
        __user__={},
        __event_emitter__=None,
    ) -> str:
        syms = self._normalize_symbols(tickers, ticker)
        return await self._execute(
            syms, mode="compact", __user__=__user__, __event_emitter__=__event_emitter__
        )

    # ---- internals ----

    def _normalize_symbols(
        self, tickers: Optional[List[str]], ticker: str
    ) -> List[str]:
        if tickers and isinstance(tickers, list):
            out = [str(t).strip().upper() for t in tickers if str(t).strip()]
            if out:
                return out
        if ticker:
            return [t.strip().upper() for t in str(ticker).split(",") if t.strip()]
        raise ValueError("No tickers provided")

    async def _execute(
        self,
        symbols: List[str],
        mode: str,
        __user__: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]],
    ) -> str:
        try:
            logger.info(f"Start {mode} analysis: {symbols}")
            if not self.valves.FINNHUB_API_KEY:
                raise RuntimeError("FINNHUB_API_KEY not provided in valves")

            client = finnhub.Client(api_key=self.valves.FINNHUB_API_KEY)
            cache_path = os.path.join(os.path.dirname(__file__), "stock_cache_shelve")

            with shelve.open(cache_path, writeback=True) as cache:
                parts: List[str] = []
                total = len(symbols)

                for i, sym in enumerate(symbols, 1):
                    if __event_emitter__ and (i == total or i % 4 == 1):
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Retrieving data for {sym} ({i}/{total})",
                                    "done": False,
                                },
                            }
                        )
                    data = await gather_stock_data(client, sym, cache)

                    if mode == "full":
                        rep = compile_report_human(data)
                    elif mode == "optimized":
                        rep = compile_report_optimized(data)
                    elif mode == "compact":
                        rep = compile_report_compact(data)
                    else:
                        rep = compile_report_human(data)

                    if parts and mode != "compact":
                        parts.append("\n=\n")
                    parts.append(rep)

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Finished {mode} analysis",
                                "done": True,
                            },
                        }
                    )

            logger.info(f"Done {mode} analysis")
            if mode == "compact":
                schema = (
                    "SCHEMA: S = Stock: ticker|price_usd|chg_day_pct|pe_ratio|roe_pct|npm_pct|rev5y_pct|de_ratio|beta|news_score_0to1|health_score_0to1\n"
                    "F = Fund/ETF: ticker|price_usd|chg_day_pct|exp_ratio_pct|yield_pct|ytd_pct|category|nav_usd|aum_millions|news_score_0to1|health_score_0to1\n"
                    "* '_0to1' fields are normalized scores: 0.0=low, 0.5=neutral, 1.0=high.\n"
                )
                return f"{schema}\n" + "\n".join(parts)
            return f"Analysis for {symbols}:\n\n" + "".join(parts)

        except Exception as e:
            logger.error(f"execution error: {e}", exc_info=True)
            raise RuntimeError(f"Error in {mode} analysis for {symbols}: {e}") from e
