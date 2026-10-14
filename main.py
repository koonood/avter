from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import FileResponse, StreamingResponse
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Tuple, Optional
import json
import re
import zipfile
import io
import hashlib
import os
import threading
import uuid

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

app = FastAPI()

# In-process analysis jobs. This keeps the UI responsive while AI calls run.
ANALYSIS_JOBS = {}
ANALYSIS_JOBS_LOCK = threading.Lock()

def _new_analysis_job():
    job_id = uuid.uuid4().hex
    job = {"status": "queued", "progress": 0, "stage": "queued", "result": None, "error": None}
    with ANALYSIS_JOBS_LOCK:
        ANALYSIS_JOBS[job_id] = job
    return job_id

def _update_analysis_job(job_id, **updates):
    with ANALYSIS_JOBS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        if job:
            job.update(updates)

def _get_analysis_job(job_id):
    with ANALYSIS_JOBS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        return dict(job) if job else None


# =========================================================
# NORMALIZED MESSAGE MODEL
# =========================================================


def normalize_message(
    dt,
    speaker,
    text,
    source,
    source_message_id=None,
    conversation_id=None,
):
    """Convert every platform into the same internal message shape."""
    if not isinstance(text, str):
        text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    speaker = str(speaker or "Unknown").strip() or "Unknown"
    if not text or dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return {
        "datetime": dt.isoformat(),
        "speaker": speaker,
        "text": text,
        "source": source,
        "source_message_id": str(source_message_id) if source_message_id is not None else None,
        "conversation_id": conversation_id,
    }


def parse_datetime(value, unit=None):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if unit == "ms" or v > 10_000_000_000:
            v /= 1000
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc).replace(tzinfo=None)
        except (ValueError, OSError, OverflowError):
            return None
    s = str(value).strip()
    if not s:
        return None
    # Telegram can use date_unixtime as a string.
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return parse_datetime(float(s))
    for candidate in (s, s.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate).replace(tzinfo=None)
        except ValueError:
            pass
    # WhatsApp iPhone exports may contain narrow no-break spaces and AM/PM.
    s = re.sub(r"[\u00a0\u202f\u2007]", " ", s)
    s = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", s)
    for fmt in (
        "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S",
        "%d/%m/%y %H:%M", "%d/%m/%y %H:%M:%S",
        "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M", "%m/%d/%y %H:%M:%S",
        "%d/%m/%Y %I:%M %p", "%d/%m/%Y %I:%M:%S %p",
        "%d/%m/%y %I:%M %p", "%d/%m/%y %I:%M:%S %p",
        "%m/%d/%Y %I:%M %p", "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%y %I:%M %p", "%m/%d/%y %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def clean_json_text(value):
    """Telegram sometimes stores text as a list of strings and entity objects."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value or "")

# =========================================================
# WHATSAPP
# =========================================================

WHATSAPP_PATTERNS = [
    # Android/older export: [31/12/2025, 23:59] Name: message
    re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?)\]\s([^:]+):\s?(.*)$"),
    # WhatsApp iPhone: [11/2/25, 2:20:53 PM] Name: message
    re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM))\]\s([^:]+):\s?(.*)$", re.I),
    re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?)\s[-–]\s([^:]+):\s?(.*)$"),
    re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM))\s[-–]\s([^:]+):\s?(.*)$", re.I),
]


def parse_whatsapp(text: str):
    messages = []
    current = None
    # WhatsApp adds invisible direction marks before some message lines and
    # narrow no-break spaces before AM/PM in iPhone exports.
    text = text.replace("\ufeff", "")
    for line in text.splitlines():
        line = re.sub(r"^[\u200e\u200f\u202a-\u202e\u2066-\u2069]+", "", line)
        line = re.sub(r"[\u00a0\u202f\u2007]", " ", line)
        match = next((m for p in WHATSAPP_PATTERNS if (m := p.match(line))), None)
        if match:
            if current:
                messages.append(current)
            date_str, time_str, speaker, body = match.groups()
            dt = parse_datetime(f"{date_str} {time_str}")
            current = normalize_message(dt, speaker, body, "whatsapp")
        elif current and line.strip():
            current["text"] += " " + line.strip()
    if current:
        messages.append(current)
    return [m for m in messages if m]

# =========================================================
# GENERIC JSON / META EXPORT HELPERS
# =========================================================


def walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_dicts(value)


def looks_like_message(d):
    keys = set(d.keys())
    has_text = bool(keys & {"text", "content", "body", "message"})
    has_sender = bool(keys & {"sender_name", "sender", "from", "author", "user_name", "username", "name"})
    has_time = bool(keys & {"timestamp_ms", "timestamp", "date", "date_unixtime", "created_at", "time"})
    return has_text and has_sender and has_time


def parse_generic_json(data, source):
    messages = []
    for d in walk_dicts(data):
        if not looks_like_message(d):
            continue
        sender = d.get("sender_name") or d.get("sender") or d.get("from") or d.get("user_name") or d.get("username")
        if isinstance(sender, dict):
            sender = sender.get("name") or sender.get("username") or sender.get("id")
        if isinstance(sender, list):
            sender = sender[0] if sender else "Unknown"
        author = d.get("author")
        if isinstance(author, dict):
            sender = author.get("name") or author.get("username") or sender
        body = d.get("content")
        if body is None:
            body = d.get("text")
        if body is None:
            body = d.get("body")
        if body is None and isinstance(d.get("message"), (str, list, dict)):
            body = d.get("message")
        body = clean_json_text(body)
        dt = parse_datetime(d.get("timestamp_ms"), "ms")
        if dt is None:
            dt = parse_datetime(d.get("date_unixtime"))
        if dt is None:
            dt = parse_datetime(d.get("timestamp"))
        if dt is None:
            dt = parse_datetime(d.get("date"))
        if dt is None:
            dt = parse_datetime(d.get("created_at"))
        if dt is None:
            dt = parse_datetime(d.get("time"))
        msg = normalize_message(dt, sender, body, source, d.get("id") or d.get("message_id"))
        if msg:
            messages.append(msg)
    return messages

# =========================================================
# TELEGRAM JSON / HTML
# =========================================================


def parse_telegram_json(data):
    messages = []
    roots = []
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        roots = data["messages"]
    else:
        roots = [d for d in walk_dicts(data) if d.get("type") == "message"]
    for d in roots:
        if d.get("type") == "service":
            continue
        sender = d.get("from") or d.get("from_name") or d.get("actor")
        text = clean_json_text(d.get("text"))
        dt = parse_datetime(d.get("date")) or parse_datetime(d.get("date_unixtime"))
        msg = normalize_message(dt, sender, text, "telegram", d.get("id"), d.get("reply_to_message_id"))
        if msg:
            messages.append(msg)
    return messages or parse_generic_json(data, "telegram")


def parse_telegram_html(text):
    # Telegram HTML exports have message blocks with .from_name, .date and .text.
    class TGParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_message = False
            self.current = {}
            self.field = None
            self.parts = []
            self.messages = []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            cls = attrs.get("class", "")
            if "message" in cls and "default" in cls:
                self.in_message = True
                self.current = {}
            if self.in_message and "from_name" in cls:
                self.field = "speaker"
                self.parts = []
            elif self.in_message and "text" in cls:
                self.field = "text"
                self.parts = []
            elif self.in_message and "date" in cls:
                self.field = "date"
                self.parts = []
        def handle_data(self, data):
            if self.in_message and self.field:
                self.parts.append(data)
        def handle_endtag(self, tag):
            if not self.in_message:
                return
            value = "".join(self.parts).strip()
            if value and self.field:
                self.current[self.field] = value
            self.field = None
            self.parts = []
            if tag == "div" and self.current.get("speaker") and self.current.get("text"):
                dt = parse_datetime(self.current.get("date"))
                msg = normalize_message(dt, self.current.get("speaker"), self.current.get("text"), "telegram")
                if msg:
                    self.messages.append(msg)
                self.current = {}
                self.in_message = False
    parser = TGParser()
    parser.feed(text)
    return parser.messages

# =========================================================
# INSTAGRAM / MESSENGER / DISCORD / VK
# =========================================================


def parse_meta_json(data, source):
    # Meta exports use sender_name + timestamp_ms + content.
    messages = parse_generic_json(data, source)
    # Keep only messages that really have content and timestamp.
    return messages


def parse_discord_json(data):
    messages = []
    roots = data.get("messages", []) if isinstance(data, dict) else []
    if not isinstance(roots, list):
        roots = []
    for d in roots:
        author = d.get("author") or {}
        speaker = author.get("username") or author.get("global_name") or author.get("name") or d.get("author")
        msg = normalize_message(parse_datetime(d.get("timestamp")), speaker, d.get("content"), "discord", d.get("id"))
        if msg:
            messages.append(msg)
    return messages or parse_generic_json(data, "discord")


def parse_vk_json(data):
    messages = []
    # VK exports commonly use from_id + date + text inside an items/response array.
    for d in walk_dicts(data):
        if "text" not in d and "message" not in d:
            continue
        sender = d.get("from_id") or d.get("from") or d.get("sender_id") or d.get("sender") or d.get("author")
        body = d.get("text") if d.get("text") is not None else d.get("message")
        dt = parse_datetime(d.get("date")) or parse_datetime(d.get("timestamp")) or parse_datetime(d.get("timestamp_ms"), "ms")
        msg = normalize_message(dt, sender, clean_json_text(body), "vk", d.get("id"))
        if msg:
            messages.append(msg)
    return messages or parse_generic_json(data, "vk")

# =========================================================
# SOURCE DETECTION
# =========================================================


def detect_source(filename: str, raw: bytes, text: str = "") -> Tuple[str, float]:
    name = (filename or "").lower()
    sample = (text or "")[:200_000].lower()
    if "_chat.txt" in name or "whatsapp" in name or re.search(r"(?:^|\n)[\u200e\u200f\u202a-\u202e\u2066-\u2069]*\[\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}", sample):
        return "whatsapp", 0.98
    if "telegram" in name or (('"messages"' in sample or "\"messages\"" in sample) and '"type"' in sample and ('"from"' in sample or '"from_name"' in sample or '"date_unixtime"' in sample)):
        return "telegram", 0.96
    if "instagram" in name or "instagram" in sample or "ig_" in name:
        return "instagram", 0.93
    if "messenger" in name or "facebook" in name or '"sender_name"' in sample:
        return "messenger", 0.90
    if "discord" in name or '"global_name"' in sample or '"channel_id"' in sample:
        return "discord", 0.93
    if "vk" in name or "vkontakte" in name or "vk.com" in sample or '"from_id"' in sample or '"sender_id"' in sample:
        return "vk", 0.94
    if name.endswith(".html") and "telegram" in sample:
        return "telegram", 0.85
    if name.endswith(".html"):
        return "html_generic", 0.45
    return "unknown", 0.15

# =========================================================
# FILE PARSING + ZIP RECURSION
# =========================================================


def parse_payload(filename: str, raw: bytes):
    detected_files = []
    all_messages = []

    if filename.lower().endswith(".zip") or raw[:4] == b"PK\x03\x04":
        outer_source, outer_confidence = detect_source(filename, raw, "")
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for info in z.infolist():
                    if info.is_dir() or info.file_size > 50_000_000:
                        continue
                    inner_name = info.filename
                    inner_raw = z.read(info)
                    msgs, source, confidence = parse_single_payload(inner_name, inner_raw)
                    # Preserve the platform identified from the outer export name.
                    # This matters for Meta archives where inner JSON paths can look identical.
                    if outer_source in {"instagram", "messenger", "telegram", "whatsapp", "vk", "discord"} and source in {"unknown", "messenger", "html_generic"}:
                        source = outer_source
                        confidence = max(confidence, outer_confidence)
                        for msg in msgs:
                            msg["source"] = source
                    if msgs:
                        detected_files.append({"filename": inner_name, "source": source, "confidence": confidence, "messages": len(msgs)})
                        all_messages.extend(msgs)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {filename}")
    else:
        msgs, source, confidence = parse_single_payload(filename, raw)
        detected_files.append({"filename": filename, "source": source, "confidence": confidence, "messages": len(msgs)})
        all_messages.extend(msgs)

    return all_messages, detected_files


def parse_single_payload(filename: str, raw: bytes):
    text = raw.decode("utf-8-sig", errors="ignore")
    source, confidence = detect_source(filename, raw, text)
    suffix = Path(filename).suffix.lower()

    if source == "whatsapp":
        return parse_whatsapp(text), source, confidence

    if source == "telegram":
        if suffix == ".json" or text.lstrip().startswith(("{", "[")):
            try:
                data = json.loads(text)
                return parse_telegram_json(data), source, confidence
            except json.JSONDecodeError:
                pass
        return parse_telegram_html(text), source, confidence

    if suffix == ".json" or text.lstrip().startswith(("{", "[")):
        try:
            data = json.loads(text)
            if source == "instagram":
                return parse_meta_json(data, "instagram"), source, confidence
            if source == "messenger":
                return parse_meta_json(data, "messenger"), source, confidence
            if source == "discord":
                return parse_discord_json(data), source, confidence
            if source == "vk":
                return parse_vk_json(data), source, confidence
            return parse_generic_json(data, source), source, confidence
        except json.JSONDecodeError:
            pass

    if suffix in {".html", ".htm"}:
        if source == "telegram":
            return parse_telegram_html(text), source, confidence
        return [], source, confidence

    # Last chance for plain text exports.
    return parse_whatsapp(text), source, confidence

# =========================================================
# MERGE / DEDUPLICATION
# =========================================================


def merge_messages(message_sets):
    seen = set()
    merged = []
    duplicates = 0
    for messages in message_sets:
        for m in messages:
            key = (
                m.get("datetime"),
                re.sub(r"\s+", " ", m.get("speaker", "")).strip().casefold(),
                re.sub(r"\s+", " ", m.get("text", "")).strip(),
            )
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            m["id"] = hashlib.sha1(("|".join(map(str, key))).encode("utf-8")).hexdigest()[:16]
            merged.append(m)
    merged.sort(key=lambda x: x["datetime"])
    return merged, duplicates

# =========================================================
# BASIC ANALYSIS
# =========================================================


def calculate_statistics(messages):
    if not messages:
        return {"total_messages": 0, "participants": [], "start_date": None, "end_date": None, "messages_by_participant": {}, "messages_by_month": {}, "messages_by_day": {}}
    dates = [datetime.fromisoformat(m["datetime"]) for m in messages]
    participants = sorted(set(m["speaker"] for m in messages))
    by_participant = Counter(m["speaker"] for m in messages)
    by_month = Counter()
    by_day = Counter()
    by_week = Counter()
    by_year = Counter()
    # Keep a complete, reproducible activity breakdown. The UI can filter each
    # granularity by year without relying on AI-generated timeline periods.
    for m in messages:
        dt = datetime.fromisoformat(m["datetime"])
        by_month[dt.strftime("%Y-%m")] += 1
        by_day[dt.strftime("%Y-%m-%d")] += 1
        iso_year, iso_week, _ = dt.isocalendar()
        by_week[f"{iso_year}-W{iso_week:02d}"] += 1
        by_year[str(dt.year)] += 1
    return {
        "total_messages": len(messages),
        "participants": participants,
        "start_date": min(dates).date().isoformat(),
        "end_date": max(dates).date().isoformat(),
        "messages_by_participant": dict(by_participant),
        "messages_by_month": dict(sorted(by_month.items())),
        "messages_by_day": dict(sorted(by_day.items())),
        "messages_by_week": dict(sorted(by_week.items())),
        "messages_by_year": dict(sorted(by_year.items())),
    }


def calculate_response_times(messages):
    response_times = defaultdict(list)
    for prev, cur in zip(messages, messages[1:]):
        if prev["speaker"] == cur["speaker"]:
            continue
        diff = (datetime.fromisoformat(cur["datetime"]) - datetime.fromisoformat(prev["datetime"])).total_seconds() / 60
        if 0 <= diff <= 24 * 60:
            response_times[cur["speaker"]].append(diff)
    by_participant = {s: round(sum(v) / len(v), 1) for s, v in response_times.items() if v}
    all_values = [x for v in response_times.values() for x in v]
    return {"overall_average_minutes": round(sum(all_values) / len(all_values), 1) if all_values else None, "by_participant": by_participant}


def calculate_message_lengths(messages):
    lengths = defaultdict(list)
    for m in messages:
        lengths[m["speaker"]].append(len(m["text"]))
    return {s: {"average_characters": round(sum(v) / len(v), 1), "median_characters": sorted(v)[len(v) // 2], "longest_message": max(v)} for s, v in lengths.items() if v}


def calculate_initiation(messages):
    initiations = Counter()
    last_date = None
    for m in messages:
        d = datetime.fromisoformat(m["datetime"]).date()
        if d != last_date:
            initiations[m["speaker"]] += 1
            last_date = d
    total = sum(initiations.values())
    return {s: {"count": initiations.get(s, 0), "share_percent": round(initiations.get(s, 0) / total * 100, 1) if total else 0} for s in set(m["speaker"] for m in messages)}


def calculate_gaps(messages):
    gaps = []
    for prev, cur in zip(messages, messages[1:]):
        hours = (datetime.fromisoformat(cur["datetime"]) - datetime.fromisoformat(prev["datetime"])).total_seconds() / 3600
        if hours > 0:
            gaps.append({"hours": hours, "from": prev["speaker"], "to": cur["speaker"], "datetime": cur["datetime"]})
    largest = sorted(gaps, key=lambda x: x["hours"], reverse=True)[:10]
    for g in largest:
        g["hours"] = round(g["hours"], 1)
    return {"average_gap_hours": round(sum(g["hours"] for g in gaps) / len(gaps), 1) if gaps else 0, "largest_gaps": largest}


def calculate_activity_periods(messages):
    monthly = defaultdict(lambda: {"messages": 0, "participants": set(), "sources": set()})
    for m in messages:
        key = datetime.fromisoformat(m["datetime"]).strftime("%Y-%m")
        monthly[key]["messages"] += 1
        monthly[key]["participants"].add(m["speaker"])
        monthly[key]["sources"].add(m.get("source", "unknown"))
    return [{"month": month, "messages": d["messages"], "participants": sorted(d["participants"]), "sources": sorted(d["sources"])} for month, d in sorted(monthly.items())]


def get_longest_messages(messages, limit=10):
    return [{"id": m.get("id"), "datetime": m["datetime"], "speaker": m["speaker"], "text": m["text"][:1000], "characters": len(m["text"]), "source": m.get("source")} for m in sorted(messages, key=lambda m: len(m["text"]), reverse=True)[:limit]]



# Observable language markers. These are deliberately broad heuristics used only
# for descriptive communication indices; they are never treated as measurements
# of love, intent, or private emotion.
REPAIR_MARKERS = re.compile(r"\b(sorry|apolog|forgive|let's talk|lets talk|can we talk|i understand|understand you|make it right|work this out|прощ|извини|прости|давай поговор|поговорим|понимаю тебя|исправ|помир|извиня|scusa|mi dispiace|perdon|parliamo|capisco|arregl|perd[oó]n|hablemos|lo siento|entschuld|verzeih|reden|ich verstehe)\b", re.I)
CONFLICT_MARKERS = re.compile(r"\b(fight|argue|argument|angry|mad|hate|break up|breakup|leave me|don't want|dont want|зл|руг|ссор|конфликт|расстан|ненавиж|уйди|не хочу|scusa|litig|arrab|odio|lasciami|rompere|pelea|enfad|odio|déjame|trennung|streit|wüt|hass|geh weg)\b", re.I)
FUTURE_MARKERS = re.compile(r"\b(tomorrow|next week|next month|next year|summer|holiday|vacation|trip|travel|visit|move in|marry|marriage|wedding|children|kids|future|plan|plans|план|завтра|на следующ|лето|отпуск|поезд|путеше|приезжай|переед|женит|свадьб|дет|будущ|вместе|завтра|зимой|весной|scorsa|domani|prossim|estate|vacanza|viagg|spos|matrimonio|figli|futuro|plan|mañana|próxim|verano|vacacion|viaj|casar|boda|hijos|futuro|morgen|nächste|sommer|urlaub|reise|heirat|hochzeit|kinder|zukunft)\b", re.I)
AFFECTION_MARKERS = re.compile(r"\b(love|loved|loving|miss you|miss u|kiss|kisses|hug|hugs|darling|baby|babe|sweetheart|beautiful|handsome|❤️|♥|люблю|любов|скучаю|целую|поцелу|обнима|солныш|милый|милая|дорогой|дорогая|любим|ti amo|amore|mi manchi|bacio|baci|abbraccio|tesoro|bella|bello|te quiero|te amo|amor|te extraño|beso|besos|abrazo|cariño|ich liebe|liebe dich|vermiss|kuss|küsse|umarm|schatz|liebling)\b", re.I)

# Repeated relationship rituals are tracked separately from emotional markers.
# They are observable routines (not proof of love/intent): greetings, good-night,
# check-ins, recurring pet names, and recurring daily-contact phrases.
RITUAL_PATTERNS = {
    "good_night": re.compile(r"\b(good night|goodnight|gn|sweet dreams|sleep well|спокойной ночи|сладких снов|доброй ночи|доброй ночи тебе|buona notte|dormi bene|buenas noches|gute nacht)\b", re.I),
    "good_morning": re.compile(r"\b(good morning|morning babe|morning baby|gm|доброе утро|с добрым утром|доброе утро тебе|buongiorno|buenos días|guten morgen)\b", re.I),
    "check_in": re.compile(r"\b(how are you|how r u|how's your day|how was your day|you home|are you home|did you eat|have you eaten|как ты|как дела|как день|ты дома|ты уже дома|поел|поела|ты ел|ты ела|как прошел день|как прошёл день|come stai|hai mangiato|come è andata)\b", re.I),
    "affectionate_address": re.compile(r"\b(babe|baby|darling|sweetheart|honey|love|любимый|любимая|милый|милая|дорогой|дорогая|солнышко|котик|зайка|amore|tesoro|cariño|schatz|liebling)\b", re.I),
}


def calculate_relationship_rituals(messages):
    """Measure recurring contact rituals across the full archive.

    These are behavioral counts, not measures of love or commitment. The goal is
    to detect routines and changes in routines without asking the model to infer
    them from a tiny evidence sample.
    """
    if not messages:
        return {"rituals": [], "monthly": []}
    monthly = defaultdict(lambda: {k: 0 for k in RITUAL_PATTERNS})
    speakers = defaultdict(lambda: Counter())
    days = defaultdict(lambda: {k: set() for k in RITUAL_PATTERNS})
    for m in messages:
        dt = datetime.fromisoformat(m["datetime"])
        month = dt.strftime("%Y-%m")
        day = dt.strftime("%Y-%m-%d")
        text = m["text"]
        for key, rx in RITUAL_PATTERNS.items():
            if rx.search(text):
                monthly[month][key] += 1
                days[month][key].add(day)
                speakers[(month, key)][m["speaker"]] += 1
    rows=[]
    for month in sorted(monthly):
        row={"month":month}
        for key in RITUAL_PATTERNS:
            row[key]=monthly[month][key]
            row[key+"_days"]=len(days[month][key])
        rows.append(row)
    rituals=[]
    labels={"good_night":"Good-night ritual","good_morning":"Good-morning ritual","check_in":"Daily check-ins","affectionate_address":"Recurring affectionate address"}
    for key,label in labels.items():
        total=sum(r[key] for r in rows)
        active=[r for r in rows if r[key]>0]
        if not active: continue
        rituals.append({
            "type":key,
            "label":label,
            "total_occurrences":total,
            "active_months":len(active),
            "first_month":active[0]["month"],
            "last_month":active[-1]["month"],
            "monthly":[{"month":r["month"],"count":r[key],"days":r[key+"_days"]} for r in rows],
            "participants":sorted({sp for (mo,k),counts in speakers.items() if k==key for sp in counts}),
        })
    return {"rituals":rituals,"monthly":rows}


def _month_key(dt):
    return dt.strftime("%Y-%m")

def _safe_index(numerator, denominator):
    return round(max(0.0, min(100.0, numerator / denominator * 100.0)), 1) if denominator else None

def calculate_monthly_relationship_metrics(messages):
    """Compute reproducible, observable monthly communication indices.

    These indices are deliberately behavioral proxies. They are not sentiment
    scores and must not be described as percentages of love or commitment.
    """
    if not messages:
        return []
    by_month = defaultdict(list)
    for m in messages:
        by_month[_month_key(datetime.fromisoformat(m["datetime"]))].append(m)
    months = []
    for month, ms in sorted(by_month.items()):
        speakers = sorted({m["speaker"] for m in ms})
        counts = Counter(m["speaker"] for m in ms)
        total = len(ms)
        if len(speakers) >= 2:
            vals = sorted(counts.values(), reverse=True)
            reciprocity = 100.0 - abs(vals[0] - vals[1]) / max(1, vals[0] + vals[1]) * 100.0
        else:
            reciprocity = 0.0

        conflict_times = []
        repair_hits = 0
        future_hits = 0
        affection_hits = 0
        for idx, m in enumerate(ms):
            text = m["text"]
            has_conflict = bool(CONFLICT_MARKERS.search(text))
            has_repair = bool(REPAIR_MARKERS.search(text))
            if has_conflict:
                conflict_times.append(idx)
            if has_repair:
                repair_hits += 1
            if FUTURE_MARKERS.search(text):
                future_hits += 1
            if AFFECTION_MARKERS.search(text):
                affection_hits += 1
        # A repair index is based on repair language, with a small bonus when a
        # repair marker follows a conflict marker within the same 48-hour window.
        repair_pairs = 0
        for i, m in enumerate(ms):
            if not REPAIR_MARKERS.search(m["text"]):
                continue
            cur_dt = datetime.fromisoformat(m["datetime"])
            for prev in reversed(ms[:i]):
                prev_dt = datetime.fromisoformat(prev["datetime"])
                delta_h = (cur_dt - prev_dt).total_seconds() / 3600
                if delta_h > 48:
                    break
                if CONFLICT_MARKERS.search(prev["text"]):
                    repair_pairs += 1
                    break
        repair_index = _safe_index(repair_hits + repair_pairs, max(1, total) * 0.08)
        future_index = _safe_index(future_hits, max(1, total) * 0.10)
        affection_index = _safe_index(affection_hits, max(1, total) * 0.12)
        signal_parts = [x for x in [reciprocity, repair_index, future_index, affection_index] if x is not None]
        signal = round(sum(signal_parts) / len(signal_parts), 1) if signal_parts else 0.0
        months.append({
            "month": month,
            "messages": total,
            "participants": speakers,
            "reciprocity": round(reciprocity, 1),
            "repair": round(repair_index or 0, 1),
            "future_plans": round(future_index or 0, 1),
            "affection": round(affection_index or 0, 1),
            "signal": signal,
        })
    return months

def _parse_report_date(value):
    if not value:
        return None
    s = str(value).strip()
    for candidate in (s, s[:7] if len(s) >= 7 else s):
        try:
            if re.fullmatch(r"\d{4}-\d{2}", candidate):
                return datetime.fromisoformat(candidate + "-01")
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    m = re.search(r"(\d{4})[-/](\d{1,2})", s)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)
    return None

def _aggregate_metric_rows(rows):
    if not rows:
        return None
    keys = ["signal", "reciprocity", "repair", "future_plans", "affection"]
    return {k: round(sum(r[k] for r in rows) / len(rows), 1) for k in keys}

def build_observable_before_after(monthly_metrics, shift_date):
    if not monthly_metrics:
        return []
    dates = [datetime.fromisoformat(r["month"] + "-01") for r in monthly_metrics]
    if shift_date is None:
        midpoint = len(monthly_metrics) // 2
        before = monthly_metrics[:max(1, midpoint)]
        after = monthly_metrics[max(1, midpoint):]
    else:
        before = [r for r in monthly_metrics if datetime.fromisoformat(r["month"] + "-01") < shift_date]
        after = [r for r in monthly_metrics if datetime.fromisoformat(r["month"] + "-01") >= shift_date]
        if not before or not after:
            midpoint = len(monthly_metrics) // 2
            before = monthly_metrics[:max(1, midpoint)]
            after = monthly_metrics[max(1, midpoint):]
    b = _aggregate_metric_rows(before)
    a = _aggregate_metric_rows(after)
    if not b or not a:
        return []
    return [
        {"metric": "Reciprocity", "before": f"{b['reciprocity']}/100", "after": f"{a['reciprocity']}/100", "before_value": b["reciprocity"], "after_value": a["reciprocity"]},
        {"metric": "Conflict repair language", "before": f"{b['repair']}/100", "after": f"{a['repair']}/100", "before_value": b["repair"], "after_value": a["repair"]},
        {"metric": "Future-oriented language", "before": f"{b['future_plans']}/100", "after": f"{a['future_plans']}/100", "before_value": b["future_plans"], "after_value": a["future_plans"]},
        {"metric": "Affectionate language markers", "before": f"{b['affection']}/100", "after": f"{a['affection']}/100", "before_value": b["affection"], "after_value": a["affection"]},
    ]


def build_free_report(messages, analysis):
    """Build the deterministic report payload from the full archive. No AI is used."""
    if not messages:
        return {
            "participants": [], "years": [], "months": [], "days": [], "weeks": [],
            "active_days": 0, "duration_days": 0, "longest_response_gap_hours": 0,
            "avg_messages_per_active_day": 0, "message_length": [],
            "signals": {"conflict_markers": 0, "repair_markers": 0, "future_markers": 0, "affection_markers": 0},
            "rituals": [], "derived_insights": {}, "activity_rhythm": {},
        }

    stats = (analysis or {}).get("statistics", {}) or {}
    by_participant = stats.get("messages_by_participant", {}) or {}
    names = sorted(by_participant, key=lambda n: (-int(by_participant[n]), n))
    total = len(messages)
    dates = [datetime.fromisoformat(x["datetime"]) for x in messages]
    start, end = min(dates), max(dates)

    def grouped_rows(key_fn):
        rows = defaultdict(lambda: {"messages": Counter(), "words": Counter()})
        for msg in messages:
            dt = datetime.fromisoformat(msg["datetime"])
            key = key_fn(dt)
            speaker = msg["speaker"]
            rows[key]["messages"][speaker] += 1
            rows[key]["words"][speaker] += len(msg.get("text", "").split())
        out=[]
        for key,data in sorted(rows.items()):
            counts=data["messages"]
            row={"period": key, "total": sum(counts.values()), "total_words": sum(data["words"].values())}
            if key_fn.__name__ == "year_key": row["year"] = key
            elif key_fn.__name__ == "week_key": row["week"] = key
            elif key_fn.__name__ == "day_key": row["date"] = key
            else: row["month"] = key
            row.update(dict(counts))
            for speaker,word_count in data["words"].items():
                row[f"{speaker}__words"] = word_count
            out.append(row)
        return out

    def year_key(dt): return dt.strftime("%Y")
    def month_key(dt): return dt.strftime("%Y-%m")
    def day_key(dt): return dt.strftime("%Y-%m-%d")
    def week_key(dt): return (dt - __import__('datetime').timedelta(days=dt.weekday())).date().isoformat()

    years = grouped_rows(year_key)
    months = grouped_rows(month_key)
    days = grouped_rows(day_key)
    weeks = grouped_rows(week_key)

    msg_lengths=[]
    for name in names:
        rows=[m for m in messages if m["speaker"]==name]
        words=sum(len(m.get("text","").split()) for m in rows)
        msg_lengths.append({"name":name,"avg_words":round(words/max(1,len(rows)),1),"messages":len(rows)})

    # Observable markers over the full archive.
    marker_counts={"conflict_markers":0,"repair_markers":0,"future_markers":0,"affection_markers":0}
    for msg in messages:
        text=msg.get("text","")
        marker_counts["conflict_markers"] += bool(CONFLICT_MARKERS.search(text))
        marker_counts["repair_markers"] += bool(REPAIR_MARKERS.search(text))
        marker_counts["future_markers"] += bool(FUTURE_MARKERS.search(text))
        marker_counts["affection_markers"] += bool(AFFECTION_MARKERS.search(text))

    # Initiation = first message on a calendar day, matching the existing deterministic metric.
    initiation=analysis.get("initiation", {}) or {}
    response_by=(analysis.get("response_times", {}) or {}).get("by_participant", {}) or {}
    participants=[]
    for name in names[:2]:
        count=int(by_participant.get(name,0))
        participants.append({
            "name":name,
            "messages":count,
            "share":round(count/max(1,total)*100,1),
            "words":sum(len(x.get("text","").split()) for x in messages if x["speaker"]==name),
            "initiation_share":float((initiation.get(name) or {}).get("share_percent",0)),
            "avg_response_hours":round(float(response_by.get(name,0))/60,1),
        })
    while len(participants)<2:
        participants.append({"name":f"Participant {len(participants)+1}","messages":0,"share":0,"words":0,"initiation_share":0,"avg_response_hours":0})

    gaps=analysis.get("gaps", {}) or {}
    largest=(gaps.get("largest_gaps") or [])
    longest=max([float(x.get("hours",0)) for x in largest] or [0])
    active_days=len(days)

    # Peaks and quietest active day.
    busiest_day=max(days,key=lambda x:int(x.get("total",0))) if days else None
    quietest=min(days,key=lambda x:int(x.get("total",0))) if days else None
    busiest_week=max(weeks,key=lambda x:int(x.get("total",0))) if weeks else None
    day_keys=[x["date"] for x in days]
    longest_streak=0; current=0; prev=None
    for key in day_keys:
        d=datetime.fromisoformat(key).date()
        if prev is not None and (d-prev).days==1: current+=1
        else: current=1
        longest_streak=max(longest_streak,current); prev=d

    hours=Counter(datetime.fromisoformat(x["datetime"]).hour for x in messages)
    weekdays=Counter(datetime.fromisoformat(x["datetime"]).strftime("%A") for x in messages)
    peak_hour=hours.most_common(1)[0][0] if hours else None
    peak_weekday=weekdays.most_common(1)[0][0] if weekdays else None

    # Monthly response-time and message-length series for visual exploration.
    # Keep this deterministic and robust: timestamps are normalized to naive UTC
    # by normalize_message, but older/custom parsers can still occasionally
    # produce mixed timezone values.
    monthly_response=defaultdict(lambda: {name: [] for name in names[:2]})
    for prev, cur in zip(messages, messages[1:]):
        if prev["speaker"] == cur["speaker"]:
            continue
        try:
            pdt=datetime.fromisoformat(prev["datetime"].replace("Z", "+00:00"))
            cdt=datetime.fromisoformat(cur["datetime"].replace("Z", "+00:00"))
            if pdt.tzinfo: pdt=pdt.astimezone(timezone.utc).replace(tzinfo=None)
            if cdt.tzinfo: cdt=cdt.astimezone(timezone.utc).replace(tzinfo=None)
            diff_hours=(cdt-pdt).total_seconds()/3600
            if 0 <= diff_hours <= 24 and cur["speaker"] in names[:2]:
                monthly_response[cdt.strftime("%Y-%m")][cur["speaker"]].append(diff_hours)
        except Exception:
            continue
    response_by_month=[]
    response_by_week_groups=defaultdict(lambda: {name: [] for name in names[:2]})
    response_by_day_groups=defaultdict(lambda: {name: [] for name in names[:2]})
    for prev, cur in zip(messages, messages[1:]):
        if prev["speaker"] == cur["speaker"]:
            continue
        try:
            pdt=datetime.fromisoformat(prev["datetime"].replace("Z", "+00:00"))
            cdt=datetime.fromisoformat(cur["datetime"].replace("Z", "+00:00"))
            if pdt.tzinfo: pdt=pdt.astimezone(timezone.utc).replace(tzinfo=None)
            if cdt.tzinfo: cdt=cdt.astimezone(timezone.utc).replace(tzinfo=None)
            diff_hours=(cdt-pdt).total_seconds()/3600
            if 0 <= diff_hours <= 24 and cur["speaker"] in names[:2]:
                day=cdt.strftime("%Y-%m-%d")
                monday=(cdt.date()-timedelta(days=cdt.weekday())).isoformat()
                response_by_day_groups[day][cur["speaker"]].append(diff_hours)
                response_by_week_groups[monday][cur["speaker"]].append(diff_hours)
        except Exception:
            continue
    for month in sorted(monthly_response):
        row={"month":month}
        for name in names[:2]:
            vals=monthly_response[month][name]
            row[name]=round(sum(vals)/len(vals),1) if vals else 0
        response_by_month.append(row)
    response_by_week=[]
    for week in sorted(response_by_week_groups):
        row={"week":week}
        for name in names[:2]:
            vals=response_by_week_groups[week][name]
            row[name]=round(sum(vals)/len(vals),1) if vals else 0
        response_by_week.append(row)
    response_by_day=[]
    for day in sorted(response_by_day_groups):
        row={"date":day}
        for name in names[:2]:
            vals=response_by_day_groups[day][name]
            row[name]=round(sum(vals)/len(vals),1) if vals else 0
        response_by_day.append(row)

    length_by_month=defaultdict(lambda: {name: [0,0] for name in names[:2]})
    for msg in messages:
        month=msg["datetime"][:7]
        speaker=msg["speaker"]
        if speaker in names[:2]:
            length_by_month[month][speaker][0] += len(msg.get("text", "").split())
            length_by_month[month][speaker][1] += 1
    message_length_by_month=[]
    for month in sorted(length_by_month):
        row={"month":month}
        for name in names[:2]:
            words,count=length_by_month[month][name]
            row[name]=round(words/count,1) if count else 0
        message_length_by_month.append(row)

    length_by_week=defaultdict(lambda: {name: [0,0] for name in names[:2]})
    length_by_day=defaultdict(lambda: {name: [0,0] for name in names[:2]})
    for msg in messages:
        try:
            dt=datetime.fromisoformat(msg["datetime"].replace("Z", "+00:00"))
            if dt.tzinfo: dt=dt.astimezone(timezone.utc).replace(tzinfo=None)
            speaker=msg["speaker"]
            if speaker not in names[:2]: continue
            monday=(dt.date()-timedelta(days=dt.weekday())).isoformat()
            day=dt.strftime("%Y-%m-%d")
            words=len(msg.get("text", "").split())
            for bucket in (length_by_week[monday], length_by_day[day]):
                bucket[speaker][0]+=words
                bucket[speaker][1]+=1
        except Exception:
            continue
    message_length_by_week=[]
    for week in sorted(length_by_week):
        row={"week":week}
        for name in names[:2]:
            words,count=length_by_week[week][name]
            row[name]=round(words/count,1) if count else 0
        message_length_by_week.append(row)
    message_length_by_day=[]
    for day in sorted(length_by_day):
        row={"date":day}
        for name in names[:2]:
            words,count=length_by_day[day][name]
            row[name]=round(words/count,1) if count else 0
        message_length_by_day.append(row)

    heat=defaultdict(int)
    for msg in messages:
        dt=datetime.fromisoformat(msg["datetime"])
        heat[(dt.weekday(),dt.hour)] += 1
    activity_heatmap=[{"weekday":d,"hour":h,"messages":heat.get((d,h),0)} for d in range(7) for h in range(24)]

    # Deeper deterministic relationship-structure metrics. These describe
    # communication mechanics only; they are deliberately not emotional scores.
    def pct(values, q):
        vals=sorted(float(x) for x in values)
        if not vals: return 0
        pos=(len(vals)-1)*q
        lo=int(pos); hi=min(lo+1,len(vals)-1); frac=pos-lo
        return vals[lo]*(1-frac)+vals[hi]*frac

    monthly_balance=[]
    monthly_turntaking=[]
    monthly_silence=[]
    monthly_night=[]
    month_groups=defaultdict(list)
    for msg in messages: month_groups[msg["datetime"][:7]].append(msg)
    for month in sorted(month_groups):
        ms=month_groups[month]
        counts=Counter(m["speaker"] for m in ms)
        a=int(counts.get(names[0],0)) if names else 0
        b=int(counts.get(names[1],0)) if len(names)>1 else 0
        denom=max(1,a+b)
        balance=round(100-abs(a-b)/denom*100,1)
        monthly_balance.append({"month":month,"balance":balance,"participant_a_share":round(a/denom*100,1),"participant_b_share":round(b/denom*100,1)})
        switches=sum(1 for x,y in zip(ms,ms[1:]) if x["speaker"]!=y["speaker"])
        switch_rate=round(switches/max(1,len(ms)-1)*100,1)
        runs=[]; run=1
        for x,y in zip(ms,ms[1:]):
            if x["speaker"]==y["speaker"]: run+=1
            else: runs.append(run); run=1
        if ms: runs.append(run)
        monthly_turntaking.append({"month":month,"balance":balance,"switch_rate":switch_rate,"avg_run":round(sum(runs)/len(runs),2) if runs else 0,"max_run":max(runs) if runs else 0})
        gaps=[]
        for x,y in zip(ms,ms[1:]):
            try:
                xdt=datetime.fromisoformat(x["datetime"]); ydt=datetime.fromisoformat(y["datetime"])
                h=(ydt-xdt).total_seconds()/3600
                if h>0: gaps.append(h)
            except Exception: pass
        monthly_silence.append({"month":month,"median_gap_hours":round(pct(gaps,.5),1) if gaps else 0,"p90_gap_hours":round(pct(gaps,.9),1) if gaps else 0,"longest_gap_hours":round(max(gaps),1) if gaps else 0})
        late=sum(1 for m in ms if datetime.fromisoformat(m["datetime"]).hour>=23 or datetime.fromisoformat(m["datetime"]).hour<6)
        monthly_night.append({"month":month,"late_night_share":round(late/max(1,len(ms))*100,1)})

    # Global conversation mechanics.
    runs_by=Counter(); all_runs=[]; run=1
    for x,y in zip(messages,messages[1:]):
        if x["speaker"]==y["speaker"]: run+=1
        else:
            all_runs.append((x["speaker"],run)); run=1
    if messages: all_runs.append((messages[-1]["speaker"],run))
    for sp,n in all_runs: runs_by[sp]+=n
    transition_count=max(0,len(messages)-1)
    switch_count=sum(1 for x,y in zip(messages,messages[1:]) if x["speaker"]!=y["speaker"])
    global_gaps=[]
    for x,y in zip(messages,messages[1:]):
        try:
            h=(datetime.fromisoformat(y["datetime"])-datetime.fromisoformat(x["datetime"])).total_seconds()/3600
            if h>0: global_gaps.append(h)
        except Exception: pass
    deep={
        "monthly_balance":monthly_balance,
        "monthly_turntaking":monthly_turntaking,
        "monthly_silence":monthly_silence,
        "monthly_night_activity":monthly_night,
        "global":{
            "speaker_switch_rate":round(switch_count/max(1,transition_count)*100,1),
            "average_run_length":round(sum(r for _,r in all_runs)/max(1,len(all_runs)),2),
            "max_run_length":max([r for _,r in all_runs] or [0]),
            "median_gap_hours":round(pct(global_gaps,.5),1) if global_gaps else 0,
            "p90_gap_hours":round(pct(global_gaps,.9),1) if global_gaps else 0,
            "late_night_share":round(sum(x["late_night_share"]*sum(1 for m in month_groups[x["month"]]) for x in monthly_night)/max(1,total),1) if monthly_night else 0,
            "active_day_rate":round(active_days/max(1,(end.date()-start.date()).days+1)*100,1),
        }
    }

    rituals=(analysis.get("relationship_rituals", {}) or {}).get("rituals", [])
    return {
        "participants":participants,
        "years":years,
        "months":months,
        "days":days,
        "weeks":weeks,
        "active_days":active_days,
        "duration_days":(end.date()-start.date()).days+1,
        "longest_response_gap_hours":round(longest,1),
        "avg_messages_per_active_day":round(total/max(1,active_days),1),
        "message_length":msg_lengths,
        "signals":marker_counts,
        "rituals":rituals,
        "derived_insights":{
            "busiest_day":busiest_day,
            "busiest_week":busiest_week,
            "quietest_active_day":quietest,
            "longest_active_streak_days":longest_streak,
        },
        "activity_rhythm":{"peak_hour":peak_hour,"peak_weekday":peak_weekday},
        "response_by_month":response_by_month,
        "response_by_week":response_by_week,
        "response_by_day":response_by_day,
        "message_length_by_month":message_length_by_month,
        "message_length_by_week":message_length_by_week,
        "message_length_by_day":message_length_by_day,
        "activity_heatmap":activity_heatmap,
        "deep_deterministic":deep,
    }

def analyze_conversation(messages):
    return {
        "statistics": calculate_statistics(messages),
        "response_times": calculate_response_times(messages),
        "message_lengths": calculate_message_lengths(messages),
        "initiation": calculate_initiation(messages),
        "gaps": calculate_gaps(messages),
        "activity": calculate_activity_periods(messages),
        "longest_messages": get_longest_messages(messages),
        "monthly_relationship_metrics": calculate_monthly_relationship_metrics(messages),
        "relationship_rituals": calculate_relationship_rituals(messages),
    }


# =========================================================
# AI RELATIONSHIP ANALYSIS
# =========================================================

AI_MODEL = os.getenv("AFTER_AI_MODEL", "gpt-5.6-terra")
AI_CHUNK_MODEL = os.getenv("AFTER_AI_CHUNK_MODEL", "gpt-5.6-luna")
AI_MAX_CHARS_PER_CHUNK = int(os.getenv("AFTER_AI_CHUNK_CHARS", "60000"))

AI_CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "period": {"type": "string"},
        "summary": {"type": "string"},
        "observations": {"type": "array", "items": {"type": "string"}},
        "turning_points": {"type": "array", "items": {"type": "string"}},
        "communication_patterns": {"type": "array", "items": {"type": "string"}},
        "conflicts_and_loops": {"type": "array", "items": {"type": "string"}},
        "repair_attempts": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 500}},
        "affection_and_closeness": {"type": "array", "items": {"type": "string"}},
        "future_language": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "date": {"type": "string"},
                    "speaker": {"type": "string"},
                    "quote": {"type": "string"},
                    "why_it_matters": {"type": "string"}
                },
                "required": ["message_id", "date", "speaker", "quote", "why_it_matters"],
                "additionalProperties": False
            }
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["period", "summary", "observations", "turning_points", "communication_patterns", "conflicts_and_loops", "repair_attempts", "affection_and_closeness", "future_language", "evidence", "uncertainties"],
    "additionalProperties": False
}

AI_FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "overall_story": {"type": "string"},
        "structural_shift": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "description": {"type": "string"},
                "confidence": {"type": "string"}
            },
            "required": ["date", "description", "confidence"],
            "additionalProperties": False
        },
        "relationship_phases": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "period": {"type": "string"},
                    "summary": {"type": "string", "maxLength": 700},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "period", "summary", "evidence_ids"],
                "additionalProperties": False
            }
        },
        "timeline": {
            "type": "array",
            "maxItems": 14,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "period": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "summary": {"type": "string", "maxLength": 500}
                },
                "required": ["label", "period", "start_date", "end_date", "summary"],
                "additionalProperties": False
            }
        },
        "turning_points": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "title": {"type": "string"},
                    "what_changed": {"type": "string", "maxLength": 500},
                    "why_it_matters": {"type": "string", "maxLength": 500},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["date", "title", "what_changed", "why_it_matters", "evidence_ids"],
                "additionalProperties": False
            }
        },
        "communication_dynamics": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 500}},
        "recurring_loops": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 500}},
        "repair_attempts": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 500}},
        "affection_patterns": {"type": "array", "maxItems": 5, "items": {"type": "string", "maxLength": 500}},
        "future_plans": {"type": "array", "maxItems": 5, "items": {"type": "string", "maxLength": 500}},
        "ritual_insights": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "ritual": {"type": "string"},
                    "observation": {"type": "string"},
                    "change": {"type": "string"},
                    "interpretation": {"type": "string"}
                },
                "required": ["ritual", "observation", "change", "interpretation"],
                "additionalProperties": False
            }
        },
        "executive_insights": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "finding": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "confidence": {"type": "string"},
                    "category": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["headline", "finding", "why_it_matters", "confidence", "category", "evidence_ids"],
                "additionalProperties": False
            }
        },
        "relationship_diagnosis": {
            "type": "object",
            "properties": {
                "bottom_line": {"type": "string"},
                "breakup_mechanism": {"type": "string"},
                "what_changed_first": {"type": "string"},
                "primary_drivers": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 500}},
                "contribution_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "participant": {"type": "string"},
                            "observed_contribution": {"type": "string"},
                            "impact_on_relationship": {"type": "string"},
                            "confidence": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["participant", "observed_contribution", "impact_on_relationship", "confidence", "evidence_ids"],
                        "additionalProperties": False
                    }
                },
                "missed_signals": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 500}},
                "intervention_summary": {"type": "string"},
                "not_supported": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 400}}
            },
            "required": ["bottom_line", "breakup_mechanism", "what_changed_first", "primary_drivers", "contribution_analysis", "missed_signals", "intervention_summary", "not_supported"],
            "additionalProperties": False
        },
        "before_after": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                    "before_value": {"type": "number"},
                    "after_value": {"type": "number"}
                },
                "required": ["metric", "before", "after", "before_value", "after_value"],
                "additionalProperties": False
            }
        },
        "intervention_windows": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "period": {"type": "string"},
                    "type": {"type": "string"},
                    "assessment": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["period", "type", "assessment", "reason"],
                "additionalProperties": False
            }
        },
        "breakup_answers": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "confidence": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["question", "answer", "confidence", "evidence_ids"],
                "additionalProperties": False
            }
        },
        "evidence": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "date": {"type": "string"},
                    "speaker": {"type": "string"},
                    "source": {"type": "string"},
                    "quote": {"type": "string"},
                    "context": {"type": "string"},
                    "interpretation": {"type": "string"}
                },
                "required": ["message_id", "date", "speaker", "source", "quote", "context", "interpretation"],
                "additionalProperties": False
            }
        },
        "limitations": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 400}}
    },
    "required": ["title", "overall_story", "structural_shift", "relationship_phases", "timeline", "turning_points", "communication_dynamics", "recurring_loops", "repair_attempts", "affection_patterns", "future_plans", "ritual_insights", "executive_insights", "relationship_diagnosis", "before_after", "intervention_windows", "breakup_answers", "evidence", "limitations"],
    "additionalProperties": False
}


def _require_ai():
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install -U openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI()


def _ai_json(client, model, system_prompt, user_prompt, schema_name, schema, max_output_tokens=4000, reasoning_effort=None):
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens,
        **({"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}),
        text={"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
    )
    if getattr(response, "status", None) == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) if details is not None else None
        # A structured JSON response can occasionally run out of output budget,
        # especially on the final synthesis. Retry once with a larger budget and
        # low reasoning. This is only used on failure, so normal analyses remain cheap.
        if reason in {None, "max_output_tokens"}:
            retry_tokens = min(max_output_tokens * 2, 14000)
            retry = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=retry_tokens,
                reasoning={"effort": "low"},
                text={"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
            )
            if getattr(retry, "status", None) != "incomplete":
                return json.loads(retry.output_text)
            details = getattr(retry, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details is not None else reason
        raise RuntimeError(f"AI response was incomplete{': ' + str(reason) if reason else ''}")
    text = response.output_text
    return json.loads(text)


def _format_messages_for_ai(messages):
    lines = []
    for m in messages:
        lines.append(
            f"[{m['id']}] {m['datetime']} | {m['speaker']} | {m.get('source','unknown')} | {m['text']}"
        )
    return "\n".join(lines)



def select_evidence_messages(messages, per_month=40):
    """Compress a large archive into a balanced evidence set for AI synthesis.

    Deterministic metrics still use every parsed message. This selector only
    controls what the language model reads, keeping large exports practical.
    Priority goes to conflict/repair/future/affection markers, long messages,
    and evenly spaced messages from each month, while preserving both speakers.
    """
    if not messages:
        return []
    grouped = defaultdict(list)
    for m in messages:
        grouped[_month_key(datetime.fromisoformat(m["datetime"]))].append(m)
    selected=[]
    for month, ms in sorted(grouped.items()):
        scored=[]
        for idx,m in enumerate(ms):
            text=m["text"]
            score=0
            if CONFLICT_MARKERS.search(text): score += 5
            if REPAIR_MARKERS.search(text): score += 5
            if FUTURE_MARKERS.search(text): score += 4
            if AFFECTION_MARKERS.search(text): score += 3
            if any(rx.search(text) for rx in RITUAL_PATTERNS.values()): score += 4
            if len(text) >= 220: score += 2
            if idx in {0,len(ms)-1}: score += 2
            scored.append((score,idx,m))
        chosen=[]
        # Preserve both participants whenever possible.
        for speaker in sorted({m["speaker"] for m in ms}):
            speaker_rows=[x for x in scored if x[2]["speaker"]==speaker]
            if speaker_rows:
                chosen.append(max(speaker_rows,key=lambda x:(x[0],len(x[2]["text"]))))
        for item in sorted(scored,key=lambda x:(x[0],len(x[2]["text"])),reverse=True):
            if item not in chosen:
                chosen.append(item)
            if len(chosen)>=per_month:
                break
        # Add a few evenly spaced ordinary messages so the sample is not only
        # emotionally loaded content.
        chosen_idx={x[1] for x in chosen}
        stride=max(1,len(ms)//max(1,per_month))
        for idx in range(0,len(ms),stride):
            if len(chosen)>=per_month: break
            if idx not in chosen_idx:
                chosen.append((0,idx,ms[idx])); chosen_idx.add(idx)
        selected.extend(m for _,_,m in sorted(chosen,key=lambda x:x[1])[:per_month])
    return sorted(selected,key=lambda m:m["datetime"])

def _chunk_messages(messages, max_chars=AI_MAX_CHARS_PER_CHUNK):
    chunks = []
    current = []
    chars = 0
    for m in messages:
        line_len = len(m.get("text", "")) + 90
        if current and chars + line_len > max_chars:
            chunks.append(current)
            current = []
            chars = 0
        current.append(m)
        chars += line_len
    if current:
        chunks.append(current)
    return chunks


def _ai_chunk_prompt(chunk, language, chunk_number, total_chunks):
    return f"""Analyze chunk {chunk_number} of {total_chunks} of a relationship message archive.
Output in {language}. This is evidence extraction, not creative writing.

Rules:
- Use only the supplied messages.
- Do not invent feelings, motives, diagnoses, infidelity, or events.
- Separate observable communication changes from interpretations.
- Look for repeated patterns, not isolated dramatic messages.
- Preserve exact message IDs for evidence.
- Quotes must be short excerpts from the supplied messages, maximum about 180 characters.
- If evidence is insufficient, say so in uncertainties.
- Consider both participants and changes over time.
- Pay special attention to recurring routines (good-night/good-morning, check-ins, affectionate address) and whether they repeat or change.

Messages:
{_format_messages_for_ai(chunk)}"""


def analyze_with_ai(messages, deterministic_analysis, language="English", progress_callback=None):
    client = _require_ai()
    if not messages:
        return None

    if progress_callback:
        progress_callback(8, "selecting_evidence")
    evidence_messages = select_evidence_messages(messages, per_month=int(os.getenv("AFTER_AI_EVIDENCE_PER_MONTH", "20")))
    chunks = _chunk_messages(evidence_messages, max_chars=min(AI_MAX_CHARS_PER_CHUNK, 60000))
    chunk_results = []
    if progress_callback:
        progress_callback(12, "preparing_ai")
    chunk_system = "You are After's evidence analyst. Reconstruct relationship dynamics from message archives with strict epistemic discipline."
    for i, chunk in enumerate(chunks, 1):
        if progress_callback:
            progress_callback(12 + int(48 * (i - 1) / max(1, len(chunks))), f"analyzing_chunk_{i}_of_{len(chunks)}")
        chunk_results.append(_ai_json(
            client,
            AI_CHUNK_MODEL,
            chunk_system,
            _ai_chunk_prompt(chunk, language, i, len(chunks)),
            "after_chunk_analysis",
            AI_CHUNK_SCHEMA,
            2200,
            reasoning_effort="low",
        ))
        if progress_callback:
            progress_callback(12 + int(48 * i / max(1, len(chunks))), f"chunk_complete_{i}_of_{len(chunks)}")

    synthesis = {
        "deterministic": deterministic_analysis,
        "ai_evidence_message_count": len(evidence_messages),
        "full_archive_message_count": len(messages),
        "chunk_analyses": chunk_results,
    }
    # The synthesis prompt receives summaries, not the full archive again.
    synthesis_text = json.dumps(synthesis, ensure_ascii=False)
    final_system = """You are After, an evidence-first relationship analysis engine.
Your job is to reconstruct the story of a relationship from a complete archive.
Do not produce generic therapy advice or a generic chat summary.
Answer the questions a person asks after a painful breakup: what happened, when it changed, whether a partner was pulling away, whether there is evidence of another person, whether blame is supported, whether there was an intervention window, why the breakup may have happened when it did, and what should have been noticed earlier.

Hard rules:
1. Never claim access to private feelings or intentions.
2. Never invent an affair or another person. If the archive lacks direct evidence, explicitly say so.
3. Distinguish observation, interpretation, and uncertainty.
4. Major conclusions should be supported by multiple dated observations when possible.
5. Evidence IDs must refer to IDs that actually appeared in the chunk analyses.
6. Do not fabricate quotes.
7. Use calibrated language such as 'the archive shows', 'consistent with', 'suggests', or 'insufficient evidence'.
8. Do not turn communication metrics into fake 'love percentages'.
9. Preserve chronology.
10. Write the final result in the requested language.
11. For timeline, create 6-14 chronological periods spanning the available archive. Do NOT invent numeric scores; the backend will attach reproducible observable metrics.
12. Identify the strongest structural shift as a date/month supported by converging changes in communication.
13. For before_after, describe what changed, but do not invent numeric values; the backend will compute them from observable message features.
14. Treat recurring relationship rituals as first-class evidence: repeated good-night/good-morning messages, daily check-ins, recurring forms of address, and similar routines. Describe when a ritual is established, stable, intensifies, weakens, or disappears only when the archive supports that pattern. A ritual's presence is observable; its emotional meaning is an interpretation and must be labeled as such.
15. Pay attention to transitions, not just dramatic messages: changes in response rhythm, routine contact, initiation, gaps, repair after conflict, future-oriented language, and affection markers.
16. If two signals move together repeatedly, describe the convergence. Do not claim causality unless the archive directly supports it.
17. If the archive is sparse, use fewer timeline periods and lower confidence rather than inventing detail.
18. The deterministic statistics cover the full archive; the message excerpts are a balanced evidence sample. Never imply that an exact quote or example represents every message.
19. The first part of the report must be insight-dense. Produce 8-12 executive_insights. Each insight must answer a concrete question such as what changed, what started the deterioration, what maintained it, what ended it, whether distance was reciprocal, whether one participant contributed more to a specific pattern, or what the archive does NOT support. Avoid generic statements like “communication changed” unless you immediately specify what changed, when, and why it matters.
20. Produce relationship_diagnosis as the core causal reconstruction. Explain the breakup mechanism as a chain of observable changes (for example: conflict → failed repair → reduced routine contact → lower future orientation), but only use links that the archive supports. Distinguish trigger, maintaining factors, and late-stage consequence.
21. For contribution_analysis, assess each participant separately and concretely. Do not assign moral blame. If the evidence shows asymmetric contribution to a communication pattern, say so directly and explain the evidence. If it is genuinely reciprocal or unclear, say that instead. “Both were at fault” is not an acceptable answer without specific observed behaviors.
22. Explicitly search for the earliest detectable deterioration, the first missed repair opportunity, the strongest structural break, and the point after which intervention became materially harder. These may be different dates.
23. Explicitly compare routine contact before and after deterioration: good-night/good-morning rituals, check-ins, affectionate address, initiation, response gaps, conflict repair, and future plans. A repeated ritual that disappears or changes is a potentially important signal, but do not equate disappearance with loss of love.
24. Prioritize dated, falsifiable statements over psychological generalities. Whenever possible, connect a conclusion to 2+ dated evidence items or a deterministic full-archive pattern.
"""
    if progress_callback:
        progress_callback(64, "building_report")
    final_prompt = f"""Synthesize the supplied deterministic statistics and chunk-level evidence into one coherent After report.

IMPORTANT OUTPUT PRIORITY: this is a forensic relationship reconstruction, not a generic summary. Put the most useful conclusions first. The user should learn in the first screen what most likely changed, what drove the deterioration, what each participant concretely contributed, the strongest turning points, the intervention window, and what is not supported. Prefer 8-12 short, high-information executive insights over long descriptive prose.

Requested language: {language}

DATA:
{synthesis_text}
"""
    report = _ai_json(client, AI_MODEL, final_system, final_prompt, "after_relationship_report", AI_FINAL_SCHEMA, 9000, reasoning_effort="low")
    if progress_callback:
        progress_callback(88, "finalizing_report")

    monthly_metrics = deterministic_analysis.get("monthly_relationship_metrics", [])
    shift_date = _parse_report_date(report.get("structural_shift", {}).get("date"))
    # The numeric timeline is always backend-derived. AI supplies labels/summaries.
    if monthly_metrics:
        n = min(14, max(6, len(monthly_metrics))) if len(monthly_metrics) >= 6 else len(monthly_metrics)
        step = max(1, len(monthly_metrics) // n)
        groups = [monthly_metrics[i:i + step] for i in range(0, len(monthly_metrics), step)]
        groups = groups[:14]
        ai_periods = report.get("timeline", [])
        timeline = []
        for idx, group in enumerate(groups):
            ai_p = ai_periods[idx] if idx < len(ai_periods) else {}
            agg = _aggregate_metric_rows(group)
            timeline.append({
                "label": ai_p.get("label") or group[0]["month"],
                "period": f"{group[0]['month']} → {group[-1]['month']}",
                "start_date": group[0]["month"] + "-01",
                "end_date": group[-1]["month"] + "-01",
                "summary": ai_p.get("summary", ""),
                "signal": agg["signal"],
                "reciprocity": agg["reciprocity"],
                "repair": agg["repair"],
                "future_plans": agg["future_plans"],
                "affection": agg["affection"],
            })
        report["timeline"] = timeline
        report["before_after"] = build_observable_before_after(monthly_metrics, shift_date)
    report.setdefault("ritual_insights", [])
    report.setdefault("executive_insights", [])
    report.setdefault("relationship_diagnosis", {"bottom_line":"","breakup_mechanism":"","what_changed_first":"","primary_drivers":[],"contribution_analysis":[],"missed_signals":[],"intervention_summary":"","not_supported":[]})
    if progress_callback:
        progress_callback(96, "checking_report")
    report["data_basis"] = {
        "timeline_metrics": "backend_observable_indices",
        "ai_interpretation": "model_synthesis",
        "warning": "Indices describe communication markers, not love, intent, or private feelings."
    }
    if progress_callback:
        progress_callback(100, "complete")
    return report

# =========================================================
# ROUTES
# =========================================================


@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/report")
def relationship_report():
    return FileResponse("report.html")


@app.get("/report.html")
def relationship_report_html():
    return FileResponse("report.html")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Detect the source without loading huge exports completely into RAM."""
    filename = file.filename or "upload"

    # We only need a small sample for platform detection. The old version
    # read the entire export first, which made large Telegram/VK result.json
    # files (>100 MB) fail with 413 and left the UI stuck on "detecting".
    sample_size = 2_000_000
    raw = await file.read(sample_size)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # A generic ZIP may have no useful platform name in its filename. Read only
    # the ZIP directory/metadata from the upload stream when possible; do not
    # load every archive member just for the detection preview.
    is_zip = filename.lower().endswith(".zip") or raw[:4] == b"PK\x03\x04"
    text = raw.decode("utf-8-sig", errors="ignore")
    source, confidence = detect_source(filename, raw, text)

    inner = []
    if is_zip:
        try:
            await file.seek(0)
            with zipfile.ZipFile(file.file) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    # Inspect filenames first. Only read a tiny sample from an
                    # inner file, rather than extracting a potentially huge file.
                    inner_name = info.filename
                    name_source, name_confidence = detect_source(inner_name, b"", "")
                    if name_source != "unknown":
                        inner.append({"filename": inner_name, "source": name_source, "confidence": name_confidence})
                        continue
                    if info.file_size > 2_000_000:
                        continue
                    with z.open(info) as member:
                        inner_raw = member.read(2_000_000)
                    inner_text = inner_raw.decode("utf-8-sig", errors="ignore")
                    inner_source, inner_confidence = detect_source(inner_name, inner_raw, inner_text)
                    if inner_source != "unknown":
                        inner.append({"filename": inner_name, "source": inner_source, "confidence": inner_confidence})
        except (zipfile.BadZipFile, OSError):
            pass

    return {
        "filename": filename,
        "source": source,
        "confidence": confidence,
        "inner": inner,
    }



@app.post("/report/pdf")
async def report_pdf(payload: dict):
    """Generate the compact, chart-led After report PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
        from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle, PolyLine
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF support is not installed. Run: pip install reportlab")

    r = payload.get("ai_report") or {}
    analysis = payload.get("analysis", {}) or {}
    stats = analysis.get("statistics", {}) or {}
    if not r:
        raise HTTPException(status_code=400, detail="No AI report available")

    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/DejaVu Sans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_path = next((x for x in font_candidates if os.path.exists(x)), None)
    if font_path:
        pdfmetrics.registerFont(TTFont("AfterSans", font_path))
        base_font = "AfterSans"
    else:
        base_font = "Helvetica"

    PURPLE = colors.HexColor("#6B55F5")
    PURPLE_LIGHT = colors.HexColor("#EEEAFE")
    BG = colors.HexColor("#F5F4F1")
    TEXT = colors.HexColor("#171716")
    MUTED = colors.HexColor("#74716B")
    LINE = colors.HexColor("#DDD9D1")
    DARK = colors.HexColor("#1A1918")
    WHITE = colors.white
    GREEN = colors.HexColor("#2F8B68")
    RED = colors.HexColor("#B65760")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
        topMargin=15*mm, bottomMargin=16*mm,
        title=str(r.get("title") or "After Relationship Report"), author="After",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ATitle", parent=styles["Title"], fontName=base_font, fontSize=27, leading=29, textColor=TEXT, spaceAfter=8))
    styles.add(ParagraphStyle(name="AH1", parent=styles["Heading1"], fontName=base_font, fontSize=18, leading=21, textColor=TEXT, spaceBefore=0, spaceAfter=6))
    styles.add(ParagraphStyle(name="AH2", parent=styles["Heading2"], fontName=base_font, fontSize=11, leading=14, textColor=TEXT, spaceBefore=0, spaceAfter=4))
    styles.add(ParagraphStyle(name="ABody", parent=styles["BodyText"], fontName=base_font, fontSize=8.5, leading=12, textColor=TEXT, spaceAfter=4))
    styles.add(ParagraphStyle(name="ASmall", parent=styles["BodyText"], fontName=base_font, fontSize=7, leading=9, textColor=MUTED, spaceAfter=3))
    styles.add(ParagraphStyle(name="AWhite", parent=styles["BodyText"], fontName=base_font, fontSize=8.5, leading=12, textColor=WHITE, spaceAfter=4))
    styles.add(ParagraphStyle(name="AWhiteTitle", parent=styles["Title"], fontName=base_font, fontSize=25, leading=27, textColor=WHITE, spaceAfter=6))
    styles.add(ParagraphStyle(name="AKicker", parent=styles["BodyText"], fontName=base_font, fontSize=6.5, leading=8, textColor=PURPLE, spaceAfter=4))

    from xml.sax.saxutils import escape
    def P(v, style="ABody"):
        return Paragraph(escape(str(v if v is not None else "")).replace("\n", "<br/>"), styles[style])
    def safe(v): return str(v if v is not None else "")
    def compact(n):
        try: n=float(n)
        except: return "0"
        if n>=1e6: return f"{n/1e6:.1f}m"
        if n>=1e3: return f"{n/1e3:.1f}k"
        return str(int(n)) if n.is_integer() else f"{n:.1f}"
    def header(kicker,title,sub=None):
        x=[P(kicker,"AKicker"),Paragraph(escape(title),styles["AH1"])]
        if sub: x.append(P(sub,"ASmall"))
        return x
    def box(flowables,bg=WHITE,pad=10,border=LINE):
        t=Table([[flowables]],colWidths=[178*mm])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.5,border),("LEFTPADDING",(0,0),(-1,-1),pad),("RIGHTPADDING",(0,0),(-1,-1),pad),("TOPPADDING",(0,0),(-1,-1),pad),("BOTTOMPADDING",(0,0),(-1,-1),pad)]))
        return t
    def two_col(left,right,widths=(88*mm,88*mm)):
        t=Table([[left,right]],colWidths=list(widths))
        t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0)]))
        return t

    monthly = analysis.get("monthly_relationship_metrics") or []
    if not monthly:
        monthly = [{"month":k,"messages":v} for k,v in (stats.get("messages_by_month") or {}).items()]
    timeline = r.get("timeline") or []
    before_after = r.get("before_after") or []
    phases = r.get("relationship_phases") or []
    turns = r.get("turning_points") or []
    evidence = r.get("evidence") or []
    windows = r.get("intervention_windows") or []
    answers = r.get("breakup_answers") or []
    shift = r.get("structural_shift") or {}
    participants = stats.get("messages_by_participant") or {}

    def volume_chart(rows):
        rows=rows[-18:]
        W,H=500,190; d=Drawing(W,H); maxv=max([float(x.get("messages",0)) for x in rows] or [1]);
        chart_h=135; base=25; gap=W/max(1,len(rows)); bar=max(8,gap*0.55)
        for i,x in enumerate(rows):
            val=float(x.get("messages",0)); h=max(2,chart_h*val/maxv); xx=i*gap+(gap-bar)/2
            d.add(Rect(xx,base,bar,h,fillColor=PURPLE,strokeColor=None,rx=3,ry=3))
            lab=safe(x.get("month",""))[-5:]
            d.add(String(xx+bar/2,10,lab,fontName=base_font,fontSize=6,fillColor=MUTED,textAnchor="middle"))
            if len(rows)<=12 or i%2==0:
                d.add(String(xx+bar/2,base+h+4,compact(val),fontName=base_font,fontSize=6,fillColor=MUTED,textAnchor="middle"))
        d.add(Line(0,base,W,base,strokeColor=LINE,strokeWidth=.7)); return d

    def signal_chart(rows):
        rows=rows[-14:]
        W,H=500,205; d=Drawing(W,H); L,R,T,B=28,10,18,34; iw=W-L-R; ih=H-T-B
        for v in (0,25,50,75,100):
            y=B+ih*v/100; d.add(Line(L,y,W-R,y,strokeColor=LINE,strokeWidth=.5)); d.add(String(L-5,y-2,str(v),fontName=base_font,fontSize=6,fillColor=MUTED,textAnchor="end"))
        pts=[]
        for i,x in enumerate(rows):
            xx=L+(iw*i/max(1,len(rows)-1)); yy=B+ih*float(x.get("signal",0))/100; pts.append((xx,yy))
        if len(pts)>1: d.add(PolyLine(pts,strokeColor=PURPLE,strokeWidth=3,strokeLineJoin=1,strokeLineCap=1))
        for i,(xx,yy) in enumerate(pts):
            d.add(Circle(xx,yy,4,fillColor=WHITE,strokeColor=PURPLE,strokeWidth=2))
            d.add(String(xx,yy+9,str(round(float(rows[i].get("signal",0))),),fontName=base_font,fontSize=6,fillColor=TEXT,textAnchor="middle"))
            d.add(String(xx,H-7,safe(rows[i].get("label") or rows[i].get("period") or "")[:12],fontName=base_font,fontSize=6,fillColor=MUTED,textAnchor="middle"))
        return d

    def before_after_chart(rows):
        W,H=500,max(130,45*len(rows)+25); d=Drawing(W,H); y=H-24
        for m in rows:
            name=safe(m.get("metric","")); name={"Affectionate language markers":"Affection markers","Conflict repair language":"Conflict repair","Future-oriented language":"Future plans"}.get(name,name)[:27]; b=float(m.get("before_value",0) or 0); a=float(m.get("after_value",0) or 0)
            d.add(String(0,y+8,name,fontName=base_font,fontSize=7,fillColor=TEXT))
            d.add(Rect(145,y,250,7,fillColor=colors.HexColor("#ECE9E3"),strokeColor=None,rx=3,ry=3)); d.add(Rect(145,y,250*b/100,7,fillColor=colors.HexColor("#BDB9B0"),strokeColor=None,rx=3,ry=3))
            d.add(Rect(145,y-10,250,7,fillColor=colors.HexColor("#F1EEFF"),strokeColor=None,rx=3,ry=3)); d.add(Rect(145,y-10,250*a/100,7,fillColor=PURPLE,strokeColor=None,rx=3,ry=3))
            d.add(String(405,y-1,f"{b:g} → {a:g}",fontName=base_font,fontSize=7,fillColor=TEXT))
            y-=45
        return d

    story=[]
    title=safe(r.get("title") or "After Relationship Report")
    date_range=f"{safe(stats.get('start_date'))} → {safe(stats.get('end_date'))}"
    total=stats.get("total_messages",payload.get("total_messages",0))

    # PAGE 1 — cover/dashboard
    story += header("AFTER · RELATIONSHIP REPORT",title,date_range+f" · {compact(total)} unique messages")
    story.append(Spacer(1,5))
    story.append(box([
        P("THE STORY", "AKicker"),
        P(r.get("overall_story") or "The archive does not support a concise overall story.","ABody")
    ], bg=PURPLE_LIGHT, border=PURPLE))
    story.append(Spacer(1,10))
    kpis=[
        [P("MESSAGES","ASmall"),P("PHASES","ASmall"),P("TURNING POINTS","ASmall"),P("SHIFT","ASmall")],
        [P(compact(total),"AH2"),P(str(len(phases)),"AH2"),P(str(len(turns)),"AH2"),P(safe(shift.get("date") or "—"),"AH2")]
    ]
    kt=Table(kpis,colWidths=[44*mm]*4,rowHeights=[9*mm,14*mm])
    kt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),WHITE),("BOX",(0,0),(-1,-1),.5,LINE),("INNERGRID",(0,0),(-1,-1),.5,LINE),("LEFTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),5),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(kt); story.append(Spacer(1,10))
    story.append(box([
        P("STRONGEST STRUCTURAL SHIFT","AKicker"),
        Paragraph(escape(safe(shift.get("date") or "Not established")),styles["AWhiteTitle"]),
        P(shift.get("description") or "No single structural shift was established with enough confidence.","AWhite"),
        P("Confidence · "+safe(shift.get("confidence") or "not stated"),"AWhite")
    ], bg=DARK, border=DARK))
    story.append(PageBreak())

    # PAGE 2 — charts
    story += header("01 · COMMUNICATION", "See the relationship move.", "Charts are derived from observable communication measures, not private feelings.")
    story.append(box([P("RELATIONSHIP SIGNAL","AKicker"),P("How the communication system changed","AH2"),signal_chart(timeline)],bg=WHITE))
    story.append(Spacer(1,10))
    story.append(box([P("MESSAGE VOLUME","AKicker"),P("Communication intensity by month","AH2"),volume_chart(monthly)],bg=WHITE))
    story.append(Spacer(1,10))
    part_rows=[[P("PARTICIPANT","ASmall"),P("MESSAGES","ASmall"),P("SHARE","ASmall")]]
    ptotal=sum(float(v) for v in participants.values()) or 1
    for name,count in sorted(participants.items(),key=lambda x:x[1],reverse=True)[:6]: part_rows.append([P(name,"ABody"),P(compact(count),"ABody"),P(f"{float(count)/ptotal*100:.1f}%","ABody")])
    story.append(box([P("PARTICIPATION","AKicker"),P("Who carried the volume?","AH2"),Table(part_rows,colWidths=[100*mm,35*mm,35*mm],style=TableStyle([("GRID",(0,0),(-1,-1),.4,LINE),("BACKGROUND",(0,0),(-1,0),BG),("FONTNAME",(0,0),(-1,-1),base_font),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))],bg=WHITE))
    story.append(PageBreak())

    # PAGE 3 — before/after + phases
    story += header("02 · CHANGE", "Before vs. after.", "A compact comparison around the structural shift.")
    story.append(box([P("OBSERVABLE COMMUNICATION MEASURES","AKicker"),before_after_chart(before_after),P("Grey = before · purple = after · 0–100 index","ASmall")],bg=WHITE))
    story.append(Spacer(1,10))
    phase_cells=[]
    for i,pv in enumerate(phases):
        phase_cells.append([P(f"PHASE {i+1:02d}","AKicker"),P(pv.get("name") or "Phase","AH2"),P(pv.get("period") or "","ASmall"),P(pv.get("summary") or "","ASmall")])
    if phase_cells:
        cols=4; rows=[]
        for i in range(0,len(phase_cells),cols):
            row=phase_cells[i:i+cols]; row += [[]]*(cols-len(row)); rows.append(row)
        # flatten each phase as a nested table
        nested=[]
        for i in range(0,len(phase_cells),cols):
            rr=[]
            for c in phase_cells[i:i+cols]: rr.append(Table([[x] for x in c],colWidths=[42*mm],style=TableStyle([("BOX",(0,0),(-1,-1),.5,LINE),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),4),("VALIGN",(0,0),(-1,-1),"TOP")])))
            rr += ["—"]*(cols-len(rr)); nested.append(rr)
        pt=Table(nested,colWidths=[44*mm]*cols,hAlign="LEFT");pt.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
        story += [P("RELATIONSHIP PHASES","AKicker"),P("How the relationship evolved","AH2"),pt]
    story.append(PageBreak())

    # PAGE 4 — turning points + patterns
    story += header("03 · STORY ARC", "Where the system actually changed.")
    turn_rows=[]
    for i,tv in enumerate(turns):
        turn_rows.append([P(str(i+1),"AH2"),P(f"{safe(tv.get('date'))} · {safe(tv.get('title'))}","AH2"),P(tv.get("what_changed") or "","ASmall"),P(tv.get("why_it_matters") or "","ASmall")])
    if turn_rows:
        tt=Table(turn_rows,colWidths=[9*mm,53*mm,57*mm,57*mm]);tt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.4,LINE),("BACKGROUND",(0,0),(0,-1),PURPLE_LIGHT),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]));story.append(box([P("TURNING POINTS","AKicker"),tt]))
    story.append(Spacer(1,10))
    loops=(r.get("recurring_loops") or [])[:6]; repairs=(r.get("repair_attempts") or [])[:6]
    left=[P("RECURRING PATTERNS","AKicker"),P("The loops underneath the story.","AH2")]+[P(f"• {x}","ABody") for x in loops]
    right=[P("REPAIR ATTEMPTS","AKicker"),P("What happened after conflict?","AH2")]+[P(f"• {x}","ABody") for x in repairs]
    story.append(two_col(box(left),box(right)))
    story.append(Spacer(1,10))
    aff=(r.get("affection_patterns") or [])[:4]; fut=(r.get("future_plans") or [])[:4]
    story.append(two_col(box([P("CLOSENESS","AKicker")]+[P(f"• {x}") for x in aff]),box([P("FUTURE","AKicker")]+[P(f"• {x}") for x in fut])))
    story.append(PageBreak())

    # PAGE 5 — evidence + intervention
    story += header("04 · EVIDENCE", "Show me what supports the conclusion.", "Selected source messages with the interpretation kept separate.")
    ev_rows=[]
    for e in evidence[:8]:
        ev_rows.append([P(f"{safe(e.get('date'))} · {safe(e.get('speaker'))}","ASmall"),P("“"+safe(e.get("quote") or "")+"”","ABody"),P(e.get("interpretation") or e.get("context") or "","ASmall")])
    if ev_rows:
        et=Table(ev_rows,colWidths=[38*mm,68*mm,70*mm]);et.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.4,LINE),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]));story.append(box([et]))
    story.append(Spacer(1,10))
    if windows:
        wrows=[[P("PERIOD","ASmall"),P("TYPE","ASmall"),P("ASSESSMENT","ASmall"),P("WHY","ASmall")]]
        for w in windows: wrows.append([P(w.get("period") or "","ASmall"),P(w.get("type") or "","ASmall"),P(w.get("assessment") or "","ASmall"),P(w.get("reason") or "","ASmall")])
        wt=Table(wrows,colWidths=[43*mm,37*mm,43*mm,53*mm],repeatRows=1);wt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.4,LINE),("BACKGROUND",(0,0),(-1,0),PURPLE_LIGHT),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]));story.append(box([P("INTERVENTION WINDOWS","AKicker"),P("When could something realistically have changed?","AH2"),wt]))
    story.append(PageBreak())

    # PAGE 6 — questions + takeaway
    story += header("05 · THE QUESTIONS THAT MATTER", "So… why did it end?", "Direct answers, with confidence and uncertainty kept visible.")
    for a in answers[:7]:
        story.append(box([P(a.get("question") or "","AH2"),P(a.get("answer") or "","ABody"),P("Confidence · "+safe(a.get("confidence") or "not stated"),"ASmall")],bg=WHITE));story.append(Spacer(1,5))
    story.append(Spacer(1,6))
    story.append(box([P("FINAL TAKEAWAY","AKicker"),Paragraph("The short version.",styles["AH1"]),P(r.get("overall_story") or "","AWhite")],bg=DARK,border=DARK))
    story.append(Spacer(1,7))
    story.append(P("After separates measured communication behavior from interpretation. It cannot know another person's private feelings from text alone, and it does not treat the absence of evidence as proof of an affair or motive.","ASmall"))

    def footer(canvas, doc):
        canvas.saveState();canvas.setStrokeColor(LINE);canvas.line(16*mm,11*mm,194*mm,11*mm);canvas.setFont(base_font,6.5);canvas.setFillColor(MUTED);canvas.drawString(16*mm,6.5*mm,"after. · Relationship analysis");canvas.drawRightString(194*mm,6.5*mm,f"Page {doc.page}");canvas.restoreState()

    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    buf.seek(0)
    return StreamingResponse(buf,media_type="application/pdf",headers={"Content-Disposition":'attachment; filename="after_relationship_report.pdf"'})

@app.post("/preview")
async def preview_upload(file: UploadFile = File(...)):
    """Parse one selected file and return exact counts before full analysis."""
    raw = await file.read()
    if not raw:
        return {"filename": file.filename or "upload", "total_messages": 0, "source_counts": {}, "detected_files": [], "date_min": None, "date_max": None, "warning": "The file is empty."}
    if len(raw) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large: {file.filename}. Maximum is 500 MB.")
    messages, detected_files = parse_payload(file.filename or "upload", raw)
    counts = Counter(m.get("source", "unknown") for m in messages)
    dated = sorted(m.get("datetime") for m in messages if m.get("datetime"))
    speaker_counts = Counter(m.get("speaker", "") for m in messages if m.get("speaker"))
    return {
        "filename": file.filename or "upload",
        "total_messages": len(messages),
        "source_counts": dict(counts),
        "detected_files": detected_files,
        "participants": [{"name": name, "messages": count} for name, count in speaker_counts.most_common()],
        "date_min": dated[0] if dated else None,
        "date_max": dated[-1] if dated else None,
        "warning": "No messages were parsed. Check the export format and date format." if not messages else None,
    }


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), language: str = Form("English"), speaker_mapping: str = Form("{}")):
    """Parse the archive and build the free report only. NEVER starts AI here."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    message_sets = []
    detected = []
    filenames = []
    try:
        parsed_speaker_mapping = json.loads(speaker_mapping or "{}")
        if not isinstance(parsed_speaker_mapping, dict):
            parsed_speaker_mapping = {}
    except (json.JSONDecodeError, TypeError):
        parsed_speaker_mapping = {}

    for file_index, file in enumerate(files):
        raw = await file.read()
        if not raw:
            continue
        if len(raw) > 500 * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large: {file.filename}. Maximum is 500 MB.")
        messages, file_detected = parse_payload(file.filename or "upload", raw)
        # Apply the user-confirmed identity mapping before deduplication and statistics.
        for message in messages:
            original_speaker = str(message.get("speaker", ""))
            mapped_speaker = parsed_speaker_mapping.get(f"{file_index}::{original_speaker}")
            if isinstance(mapped_speaker, str) and mapped_speaker.strip():
                message["speaker"] = mapped_speaker.strip()
        filenames.append(file.filename)
        detected.extend(file_detected)
        message_sets.append(messages)

    messages, duplicates_removed = merge_messages(message_sets)
    analysis = analyze_conversation(messages)
    free_report = build_free_report(messages, analysis)
    source_counts = Counter(m.get("source", "unknown") for m in messages)

    job_id = _new_analysis_job()
    base_payload = {
        "analysis_id": job_id,
        "language": language,
        "filenames": filenames,
        "detected_sources": detected,
        "source_summary": dict(source_counts),
        "duplicates_removed": duplicates_removed,
        "total_messages": len(messages),
        "analysis": analysis,
        "free_report": free_report,
        "ai_price_eur": 2.99,
        "messages": messages[:100],
        "ai_report": None,
        "ai_error": None,
    }

    # Keep the full parsed archive server-side for the explicit AI step only.
    # It is never sent to the model during /upload.
    with ANALYSIS_JOBS_LOCK:
        ANALYSIS_JOBS[job_id]["messages_for_ai"] = messages
        ANALYSIS_JOBS[job_id]["language"] = language

    # IMPORTANT: upload is COMPLETE at this point. No OpenAI call is made.
    # The report page opens with the free/local analysis and only the explicit
    # AI button can call /ai-analyze/{job_id}.
    _update_analysis_job(
        job_id,
        progress=100,
        stage="free_report_ready",
        status="complete",
        result=base_payload,
    )
    return {
        "job_id": job_id,
        "analysis_id": job_id,
        "status": "complete",
        "progress": 100,
        "stage": "free_report_ready",
        "total_messages": len(messages),
    }

@app.post("/ai-analyze/{job_id}")
def start_ai_analysis(job_id: str, request: dict = Body(default={})):
    """Start the paid/explicit AI layer in the language currently selected by the user.

    The frontend sends the active report language explicitly. This is intentionally
    separate from the upload language so a user can switch languages on the report
    before paying/starting the deep analysis.
    """
    job = _get_analysis_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    if job.get("status") == "ai_running":
        return {"status": "running", "job_id": job_id}
    if job.get("status") == "ai_complete":
        return {"status": "complete", "job_id": job_id}
    if not job.get("result"):
        raise HTTPException(status_code=409, detail="Free analysis is not ready")

    base_payload = job["result"]
    analysis = base_payload.get("analysis", {})
    messages = base_payload.get("_messages_for_ai")
    if messages is None:
        # Keep the full parsed archive server-side for the AI job without
        # putting it into the browser payload.
        messages = job.get("messages_for_ai", [])
    requested_language = (request or {}).get("language")
    allowed_languages = {"English", "Russian", "Spanish", "Italian", "German"}
    if requested_language in allowed_languages:
        language = requested_language
    else:
        language = job.get("language") or base_payload.get("language") or "English"

    # Persist the exact language used for this AI run so the result and subsequent
    # status responses remain tied to the user's selection.
    with ANALYSIS_JOBS_LOCK:
        if job_id in ANALYSIS_JOBS:
            ANALYSIS_JOBS[job_id]["language"] = language

    _update_analysis_job(job_id, status="ai_running", progress=5, stage="starting_ai")

    def run_ai():
        try:
            ai_report = analyze_with_ai(
                messages,
                analysis,
                language=language,
                progress_callback=lambda pct, stage: _update_analysis_job(
                    job_id, progress=pct, stage=stage, status="ai_running"
                ),
            )
            result = dict(base_payload)
            result["language"] = language
            result["ai_language"] = language
            result["ai_report"] = ai_report
            result["ai_error"] = None
            _update_analysis_job(job_id, progress=100, stage="complete", status="ai_complete", result=result)
        except Exception as exc:
            result = dict(base_payload)
            result["language"] = language
            result["ai_language"] = language
            result["ai_error"] = str(exc)
            _update_analysis_job(job_id, progress=100, stage="error", status="ai_error", result=result, error=str(exc))

    # The complete archive is attached only to the in-memory server job.
    # It is never sent to AI during /upload.
    with ANALYSIS_JOBS_LOCK:
        ANALYSIS_JOBS[job_id]["messages_for_ai"] = messages
    threading.Thread(target=run_ai, daemon=True).start()
    return {"status": "running", "job_id": job_id}

@app.get("/analysis/{job_id}")
def analysis_status(job_id: str):
    job = _get_analysis_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return {
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "stage": job.get("stage"),
        "error": job.get("error"),
        "result": job.get("result") if job.get("status") in {"complete", "error", "ai_complete", "ai_error"} else None,
    }

