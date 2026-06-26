"""
src/sentiment.py — News sentiment: fetch + score (FinBERT + VADER).

Self-contained side-experiment module (does NOT touch the locked modules). Two
stages:

  fetch  — stream the public FNSPID news corpus from Hugging Face and filter it
           down to just our 8 tickers' headlines (the full file is 5.7 GB; we
           keep only matched rows, a few MB). One-time ~12-minute scan.
  score  — (added next) score each headline with FinBERT and VADER, aggregate to
           a daily mean sentiment per ticker.

Run the fetch stage:
    python -m src.sentiment fetch
"""

from __future__ import annotations

# torch before pandas on this machine (WinError 1114); src.data imports pandas,
# so torch must be imported first.
import torch

import csv
import io
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data import TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_NEWS_PATH = PROJECT_ROOT / "data" / "raw" / "news_fnspid_raw.csv"
SENTIMENT_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment.csv"

# Cap headlines scored per (ticker, day) so FinBERT CPU runtime stays bounded; a
# daily mean over up to this many headlines is plenty for a daily signal.
MAX_HEADLINES_PER_DAY = 12
FINBERT_MODEL = "ProsusAI/finbert"

# Live news (NewsAPI). Finance-focused query per ticker so we get market news,
# not unrelated articles that merely mention the company name.
NEWSAPI_URL = "https://newsapi.org/v2/everything"
TICKER_QUERIES = {
    "AAPL": "Apple stock", "META": "Meta Platforms stock", "MSFT": "Microsoft stock",
    "TSLA": "Tesla stock", "AMZN": "Amazon stock", "XOM": "Exxon Mobil stock",
    "JNJ": "Johnson & Johnson stock", "JPM": "JPMorgan stock",
}
LIVE_SENTIMENT_PATH = PROJECT_ROOT / "outputs" / "live_sentiment.json"

# Public FNSPID news file — the big one covers 2009..2023 and is sorted by
# Stock_symbol. Columns: Unnamed:0, Date, Article_title, Stock_symbol, Url, ...
FNSPID_URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
              "Stock_news/nasdaq_exteral_data.csv")

# Each real row starts with: newline, a numeric index (a float like "0.0"),
# comma, ISO date. A reliable row-boundary marker even though later text columns
# contain embedded newlines.
ROW_RE = re.compile(rb"\n\d+(?:\.\d+)?,(\d{4}-\d{2}-\d{2})")

# Some tickers traded under a different symbol historically (Meta was "FB" until
# mid-2022). We pull both source symbols and map them to our canonical ticker.
SOURCE_SYMBOLS = {t: [t] for t in TICKERS}
SOURCE_SYMBOLS["META"] = ["META", "FB"]

csv.field_size_limit(10_000_000)


def _http_range(url: str, start: int, end: int, retries: int = 8) -> bytes:
    """GET bytes [start, end] with retry + exponential backoff.

    Backoff caps at 30s and spans ~90s total, so a brief DNS/network outage
    (e.g. getaddrinfo failures) doesn't abort a long fetch.
    """
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(30, 2 ** attempt))


def _total_size(url: str) -> int:
    """Total file size via a 1-byte Range GET (HF's HEAD omits Content-Length)."""
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
    r.raise_for_status()
    cr = r.headers.get("Content-Range", "")
    if "/" in cr:
        return int(cr.rsplit("/", 1)[-1])
    return int(r.headers.get("Content-Length", 0))


def _symbol_at(url: str, offset: int, total: int, probe: int = 262_144):
    """Return (row_start_byte, Stock_symbol) for the first full row at/after offset."""
    end = min(offset + probe, total - 1)
    chunk = _http_range(url, offset, end)
    m = ROW_RE.search(chunk)
    if not m:
        return None
    seg = chunk[m.start() + 1: m.start() + 1 + 65_536].decode("utf-8", "replace")
    try:
        row = next(csv.reader(io.StringIO(seg)))
    except Exception:
        return None
    if len(row) < 4:
        return None
    return offset + m.start() + 1, row[3].strip()


def _lower_bound(url: str, target: str, total: int) -> int:
    """Binary-search the byte offset where Stock_symbol >= target (file is sorted)."""
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi) // 2
        res = _symbol_at(url, mid, total)
        if res is None:
            hi = mid
            continue
        _, sym = res
        if sym < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _collect_symbol(url: str, symbol: str, canonical: str, total: int,
                    step: int = 4_000_000) -> list[tuple]:
    """Scan forward from the symbol's block, collecting (date, headline, ticker)."""
    pos = max(0, _lower_bound(url, symbol, total) - 1_000_000)  # back up to be safe
    rows: list[tuple] = []
    buf = b""
    started = False
    while pos < total:
        chunk = _http_range(url, pos, min(pos + step, total - 1))
        if not chunk:
            break
        buf += chunk
        pos += len(chunk)
        starts = [m.start() for m in ROW_RE.finditer(buf)]
        for i in range(len(starts) - 1):
            seg = buf[starts[i] + 1: starts[i + 1]].decode("utf-8", "replace")
            try:
                row = next(csv.reader(io.StringIO(seg)))
            except Exception:
                continue
            if len(row) < 4:
                continue
            sym = row[3].strip()
            if sym == symbol:
                rows.append((row[1][:10], row[2], canonical))
                started = True
            elif started and sym > symbol:
                return rows  # walked past this symbol's block
        buf = buf[starts[-1]:] if starts else buf[-1_000_000:]
    return rows


def fetch_fnspid_news(tickers=TICKERS, url: str = FNSPID_URL,
                      out_path: Path = RAW_NEWS_PATH) -> Path:
    """Fetch headlines for `tickers` from the big FNSPID file via byte-range
    binary search (downloads only each ticker's block, not the whole 23 GB).

    Resumable: each ticker is appended to out_path as soon as it's collected,
    and tickers already present are skipped on a re-run — so a mid-run network
    failure doesn't lose completed work.
    """
    total = _total_size(url)
    print(f"FNSPID file size: {total/1e9:.1f} GB", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            done = set(pd.read_csv(out_path)["ticker"].unique())
            print(f"resuming — already collected: {sorted(done)}", flush=True)
        except Exception:
            done = set()
    write_header = not (out_path.exists() and out_path.stat().st_size > 0)

    for ticker in tickers:
        if ticker in done:
            print(f"  {ticker:5s}: already collected, skipping", flush=True)
            continue
        rows: list[tuple] = []
        for symbol in SOURCE_SYMBOLS[ticker]:
            got = _collect_symbol(url, symbol, ticker, total)
            print(f"  {ticker:5s} (symbol {symbol:4s}): {len(got)} headlines", flush=True)
            rows.extend(got)
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Date", "headline", "ticker"])
                write_header = False
            writer.writerows(rows)

    print(f"DONE -> {out_path}", flush=True)
    return out_path


# --------------------------------------------------------------------------- #
# Scoring: FinBERT + VADER -> daily mean sentiment per ticker
# --------------------------------------------------------------------------- #

def load_raw_news(path: Path = RAW_NEWS_PATH) -> pd.DataFrame:
    """Load the filtered headlines, clean, and cap to MAX_HEADLINES_PER_DAY."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "headline"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 0]
    df["day"] = df["Date"].dt.normalize()
    # Sub-sample within each ticker-day to bound FinBERT cost.
    df = (df.groupby(["ticker", "day"], group_keys=False)
            .apply(lambda g: g.sample(min(len(g), MAX_HEADLINES_PER_DAY), random_state=0)))
    return df.reset_index(drop=True)


def vader_scores(texts) -> np.ndarray:
    """VADER compound sentiment in [-1, 1] per headline (lexicon baseline)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    return np.array([analyzer.polarity_scores(t)["compound"] for t in texts],
                    dtype=np.float32)


def load_finbert():
    """Load FinBERT tokenizer + model once (cache this at the call site)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).eval()
    return tok, model


def finbert_score_texts(texts, tok, model, batch_size: int = 32) -> np.ndarray:
    """FinBERT signed sentiment = P(positive) - P(negative), given a loaded model."""
    texts = list(texts)
    if not texts:
        return np.array([], dtype=np.float32)
    id2 = {i: l.lower() for i, l in model.config.id2label.items()}
    pos_i = next(i for i, l in id2.items() if l == "positive")
    neg_i = next(i for i, l in id2.items() if l == "negative")
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=1)
            out.append((probs[:, pos_i] - probs[:, neg_i]).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def finbert_scores(texts, batch_size: int = 32) -> np.ndarray:
    """Convenience: load FinBERT and score `texts` (loads the model each call)."""
    tok, model = load_finbert()
    return finbert_score_texts(texts, tok, model, batch_size)


# --------------------------------------------------------------------------- #
# Live news via NewsAPI + FinBERT-vs-VADER comparison
# --------------------------------------------------------------------------- #

def load_newsapi_key() -> str | None:
    """Read NEWSAPI_KEY from the environment or the local .env file."""
    key = os.environ.get("NEWSAPI_KEY")
    if key:
        return key
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("NEWSAPI_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def fetch_newsapi_headlines(ticker: str, page_size: int = 50, key: str | None = None) -> list[dict]:
    """Recent (de-duplicated) headlines for a ticker via NewsAPI /everything."""
    key = key or load_newsapi_key()
    if not key:
        raise RuntimeError("NEWSAPI_KEY not set (.env or environment).")
    params = {"q": TICKER_QUERIES.get(ticker, ticker), "language": "en",
              "sortBy": "publishedAt", "pageSize": page_size, "apiKey": key}
    r = requests.get(NEWSAPI_URL, params=params, timeout=30)
    r.raise_for_status()
    seen, out = set(), []
    for a in r.json().get("articles", []):
        title = (a.get("title") or "").strip()
        if title and title != "[Removed]" and title.lower() not in seen:
            seen.add(title.lower())
            out.append({"title": title, "publishedAt": a.get("publishedAt"),
                        "source": (a.get("source") or {}).get("name")})
    return out


def live_sentiment(ticker: str, key: str | None = None, finbert=None) -> dict:
    """Fetch recent headlines and score them with both FinBERT and VADER."""
    items = fetch_newsapi_headlines(ticker, key=key)
    titles = [x["title"] for x in items]
    if not titles:
        return {"ticker": ticker, "n": 0, "finbert_mean": 0.0, "vader_mean": 0.0,
                "agreement": 0.0, "headlines": []}
    tok, model = finbert or load_finbert()
    fin = finbert_score_texts(titles, tok, model)
    vad = vader_scores(titles)
    for x, f, v in zip(items, fin, vad):
        x["finbert"], x["vader"] = float(f), float(v)
    return {
        "ticker": ticker, "n": len(titles),
        "finbert_mean": float(fin.mean()), "vader_mean": float(vad.mean()),
        "agreement": float(np.mean(np.sign(fin) == np.sign(vad))),
        "headlines": items,
    }


def live_compare_all(tickers=TICKERS) -> dict:
    """Run live FinBERT-vs-VADER for every ticker; save JSON + a comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = load_newsapi_key()
    finbert = load_finbert()
    per_ticker, all_fin, all_vad = [], [], []
    for t in tickers:
        s = live_sentiment(t, key=key, finbert=finbert)
        per_ticker.append({k: v for k, v in s.items() if k != "headlines"} | {"headlines": s["headlines"][:10]})
        all_fin += [h["finbert"] for h in s["headlines"]]
        all_vad += [h["vader"] for h in s["headlines"]]
        print(f"  {t:5s} n={s['n']:3d}  FinBERT={s['finbert_mean']:+.3f}  "
              f"VADER={s['vader_mean']:+.3f}  agree={s['agreement']:.2f}", flush=True)

    all_fin, all_vad = np.array(all_fin), np.array(all_vad)
    corr = float(np.corrcoef(all_fin, all_vad)[0, 1]) if len(all_fin) > 1 else 0.0
    agree = float(np.mean(np.sign(all_fin) == np.sign(all_vad))) if len(all_fin) else 0.0

    # Plot: per-headline scatter + per-ticker aggregate bars.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(all_vad, all_fin, s=12, alpha=0.4, color="steelblue")
    ax1.axhline(0, color="grey", lw=0.6); ax1.axvline(0, color="grey", lw=0.6)
    ax1.set_xlabel("VADER score"); ax1.set_ylabel("FinBERT score")
    ax1.set_title(f"Per-headline agreement (r={corr:.2f}, sign-agree={agree:.0%})")
    names = [p["ticker"] for p in per_ticker]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, [p["finbert_mean"] for p in per_ticker], 0.4, label="FinBERT", color="darkorange")
    ax2.bar(x + 0.2, [p["vader_mean"] for p in per_ticker], 0.4, label="VADER", color="seagreen")
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_ylabel("Mean sentiment"); ax2.set_title("Aggregate sentiment by ticker")
    ax2.legend()
    fig.suptitle("Live news sentiment: FinBERT vs VADER (recent NewsAPI headlines)")
    fig.tight_layout()
    (PROJECT_ROOT / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(PROJECT_ROOT / "outputs" / "plots" / "finbert_vs_vader.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    result = {"correlation": corr, "sign_agreement": agree,
              "n_headlines": int(len(all_fin)), "per_ticker": per_ticker}
    LIVE_SENTIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIVE_SENTIMENT_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(result, f, indent=2)
    print(f"\ncorrelation r={corr:.3f}  sign-agreement={agree:.0%}  over {len(all_fin)} headlines")
    print(f"saved -> {LIVE_SENTIMENT_PATH} + plots/finbert_vs_vader.png")
    return result


def build_daily_sentiment(use_finbert: bool = True) -> Path:
    """Score headlines with VADER (always) and FinBERT (best-effort), then
    aggregate to daily means per ticker.

    VADER needs no download, so it always succeeds. FinBERT is attempted only if
    use_finbert; any failure (e.g. model download issues) falls back to VADER-only
    so this stage is guaranteed to produce a usable sentiment file.
    """
    df = load_raw_news()
    print(f"scoring {len(df)} headlines (<= {MAX_HEADLINES_PER_DAY}/ticker-day)...", flush=True)
    df["vader"] = vader_scores(df["headline"].tolist())

    agg = {"vader_score": ("vader", "mean"), "n_headlines": ("headline", "size")}
    if use_finbert:
        try:
            df["finbert"] = finbert_scores(df["headline"].tolist())
            agg["finbert_score"] = ("finbert", "mean")
        except Exception as exc:
            print(f"  FinBERT unavailable ({exc}); writing VADER-only.", flush=True)

    daily = (df.groupby(["ticker", "day"]).agg(**agg)
               .reset_index().rename(columns={"day": "Date"}))
    SENTIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(SENTIMENT_PATH, index=False)
    cols = [c for c in ("finbert_score", "vader_score") if c in daily.columns]
    print(f"saved {len(daily)} ticker-days ({', '.join(cols)}) -> {SENTIMENT_PATH}")
    return SENTIMENT_PATH


def main() -> None:
    args = sys.argv[1:]
    stage = args[0] if args else "fetch"
    if stage == "fetch":
        tickers = args[1:] if len(args) > 1 else TICKERS  # optional explicit ticker list
        fetch_fnspid_news(tickers=tickers)
    elif stage == "score":
        build_daily_sentiment(use_finbert=True)
    elif stage == "score-vader":
        build_daily_sentiment(use_finbert=False)
    elif stage == "live-compare":
        live_compare_all()
    else:
        raise SystemExit(f"unknown stage {stage!r} "
                         "(use 'fetch', 'score', 'score-vader', or 'live-compare')")


if __name__ == "__main__":
    main()
