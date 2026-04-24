#!/usr/bin/env python3
"""Scraper for https://aimcqs.com that builds a clean multi-choice-question dataset.

Changes in this revision
------------------------
* **Fixed UnicodeEncodeError on Windows** – the previous `USER_AGENT` string accidentally
  contained a non-ASCII *non-breaking hyphen* (U+2011).  On Windows+Python 3.10,
  `http.client` tries to encode headers as ASCII and crashes.  The user-agent is now
  a short, *pure-ASCII* string so the request goes through.
* No other logic has changed.

Run the script directly:  `python scraping_ML_questions.py`
It still saves to **aimcqs_mcqa.json**.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://aimcqs.com"
OUTPUT_FILE = Path("aimcqs_mcqa.json")

# ---------------------------------------------------------------------------
# HTTP settings
# ---------------------------------------------------------------------------
# IMPORTANT: keep this string *pure ASCII* to avoid UnicodeEncodeError on Windows.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_category_urls() -> List[str]:
    """Return all valid category landing-page URLs on aimcqs.com."""
    res = requests.get(BASE_URL, headers=HEADERS, timeout=20)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    candidates: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].rstrip("/")
        if href.startswith("/") and href.count("/") == 1 and not href.startswith("/page"):
            candidates.add(urljoin(BASE_URL, href))

    excluded = {
        "about",
        "contact",
        "privacy-policy",
        "terms-and-conditions",
        "ai-insights",
        "",
    }
    return [u for u in sorted(candidates) if u.rsplit("/", 1)[-1] not in excluded]


# ---------------------------------------------------------------------------
# Core extraction logic – line-based
# ---------------------------------------------------------------------------

def extract_questions_from_container(container: "bs4.element.Tag", category_name: str) -> List[Dict]:
    """Pull clean Q/A dictionaries from one <div class="text-box"> block."""
    raw = container.get_text(separator="\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    questions: List[Dict] = []
    i = 0
    while i < len(lines):
        # 1. Question line e.g. "1. What is ...?"
        if not re.match(r"^\d+\.\s+", lines[i]):
            i += 1
            continue
        q_text = lines[i]

        # 2. Four option lines (handles `a.` and `a)`)
        opts: List[str] = []
        for j in range(1, 5):
            if i + j >= len(lines):
                break
            m = re.match(r"^([a-d])[.)]\s+(.+)", lines[i + j], flags=re.I)
            if not m:
                break
            letter, txt = m.group(1).lower(), m.group(2).strip()
            opts.append(f"{letter}. {txt}")
        if len(opts) != 4:
            i += 1
            continue

        # 3. Answer line within the next few lines
        answer: str | None = None
        for k in range(i + 5, min(i + 12, len(lines))):
            m = re.match(r"^(?:Answer|Ans)\s*:\s*([A-D])", lines[k], flags=re.I)
            if m:
                answer = m.group(1).upper()
                break
        if not answer:
            i += 1
            continue

        # 4. Uniqueness check and append
        if len(set(o.split(". ", 1)[1] for o in opts)) == 4:
            questions.append(
                {
                    "question": q_text,
                    "choices": opts,
                    "answer": answer,
                    "category": category_name,
                }
            )
        i = k + 1  # jump beyond answer line

    return questions


# ---------------------------------------------------------------------------
# Scraping loop for one category
# ---------------------------------------------------------------------------

def scrape_category(category_url: str, category_name: str, max_pages: int = 20) -> List[Dict]:
    all_qs: List[Dict] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{category_url}?page={page}" if page > 1 else category_url
        res = requests.get(url, headers=HEADERS, timeout=20)
        if res.status_code != 200:
            break

        soup = BeautifulSoup(res.text, "html.parser")
        containers = soup.find_all("div", class_="text-box")
        if not containers:
            break

        new_found = 0
        for box in containers:
            qs = extract_questions_from_container(box, category_name)
            for q in qs:
                qid = q["question"]
                if qid not in seen:
                    seen.add(qid)
                    all_qs.append(q)
                    new_found += 1
        if new_found == 0:
            break
        time.sleep(1.5)
    return all_qs


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    print("📥 Gathering category URLs …")
    categories = get_category_urls()
    print(f"→ {len(categories)} categories detected\n")

    aggregated: List[Dict] = []
    for cat_url in tqdm(categories, desc="Scraping", unit="category"):
        cat_name = cat_url.rstrip('/').split('/')[-1]
        try:
            qs = scrape_category(cat_url, cat_name)
            print(f"  {cat_name:<30} {len(qs):>4} questions")
            aggregated.extend(qs)
        except Exception as exc:
            print(f"  !! {cat_name} – error: {exc}")
        time.sleep(2)

    if aggregated:
        OUTPUT_FILE.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 Saved {len(aggregated)} questions → {OUTPUT_FILE}")
    else:
        print("No data extracted.")


if __name__ == "__main__":
    main()
