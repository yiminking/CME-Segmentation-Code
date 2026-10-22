import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

MIN_QUALITY = 3
MAX_OFFSET_SECONDS = 3600
HEADER = "First C2 Appearance Date Time [UT]:"
QUALITY_PREFIX = re.compile(r"^#QUALITY[ _]INDEX\s*:", re.IGNORECASE)
QUALITY_VALUE = re.compile(r"^#QUALITY[ _]INDEX\s*:\s*([+-]?\d+)(?=\s|\(|$)", re.IGNORECASE)
TIMESTAMP = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?$")
C2_LABELS = {"C2", "2", "LASCO_C2", "LASCO-C2"}


def parse_utc(value):
    """Parse an ISO or YYYY/MM/DD timestamp; timezone-free input means UTC."""
    value = value.strip()
    if not TIMESTAMP.fullmatch(value):
        raise ValueError(f"Invalid timestamp: {value!r}")
    result = datetime.fromisoformat(value.replace("/", "-").replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def get_quality_index_value(value):
    """Return an integer quality index, or None for missing/invalid input."""
    match = QUALITY_VALUE.match(value.strip()) if value else None
    return int(match.group(1)) if match else None


def parse_source_file(file_path):
    """Read original, processed, or simplified event-block text files.

    Each event starts with HEADER and a timestamp, or a bare timestamp.
    Measurement columns are HEIGHT DATE TIME ANGLE TEL FC [COL ROW].
    Event titles, separator lines and other non-measurement text are ignored.
    """
    events = []
    current = None
    for line in Path(file_path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith(HEADER) or TIMESTAMP.fullmatch(line):
            if current is not None:
                events.append(current)
            event_time = line[len(HEADER):].strip() if line.startswith(HEADER) else line
            current = {"time": event_time, "quality_index": None, "trajectory_data": []}
            continue
        if current is None or not line:
            continue
        if QUALITY_PREFIX.match(line):
            if current["quality_index"] is not None and current["quality_index"] != line:
                current["conflicting_quality"] = True
            current["quality_index"] = line
            continue
        parts = line.split()
        if len(parts) >= 5 and not line.startswith("#"):
            try:
                float(parts[0])
            except ValueError:
                continue
            current["trajectory_data"].append(line)
    if current is not None:
        events.append(current)
    return events


def filter_events_by_quality(events, min_quality=MIN_QUALITY):
    """Retain only known integer quality indices >= 3 (or a stricter limit)."""
    if min_quality < MIN_QUALITY:
        raise ValueError("The minimum quality threshold cannot be below 3")
    kept = []
    for event in events:
        quality = get_quality_index_value(event.get("quality_index"))
        if quality is not None and quality >= min_quality and not event.get("conflicting_quality"):
            kept.append(event)
    return kept, len(events) - len(kept)


def first_c2_time(event):
    """Return the earliest listed C2 time, ignoring C3 measurements.

    Any invalid C2 timestamp rejects the event: silently skipping it could
    change which observation is treated as the first one.
    """
    times = []
    for line in event["trajectory_data"]:
        parts = line.split()
        if len(parts) >= 5 and parts[4].upper() in C2_LABELS:
            times.append(parse_utc(parts[1] + "T" + parts[2]))
    return min(times) if times else None


def select_events(events):
    """Apply quality first, then abs(first C2 - catalog start) <= 3600 s."""
    kept, audit = [], []
    for index, event in enumerate(events, 1):
        quality = get_quality_index_value(event.get("quality_index"))
        entry = {"event_number": index, "event_time": event["time"],
                 "quality_index": quality, "first_c2_time": None,
                 "signed_offset_seconds": None, "absolute_offset_seconds": None,
                 "kept": False}
        if not event.get("quality_index"):
            reason = "missing_quality"
        elif quality is None or event.get("conflicting_quality"):
            reason = "invalid_quality"
        elif quality < MIN_QUALITY:
            reason = "quality_below_3"
        else:
            try:
                start = parse_utc(event["time"])
            except ValueError:
                reason = "invalid_event_time"
            else:
                try:
                    first = first_c2_time(event)
                except ValueError:
                    reason = "invalid_c2_time"
                else:
                    if first is None:
                        reason = "no_c2_observations"
                    else:
                        offset = (first - start).total_seconds()
                        entry.update(first_c2_time=first.isoformat().replace("+00:00", "Z"),
                                     signed_offset_seconds=offset,
                                     absolute_offset_seconds=abs(offset))
                        reason = "kept" if abs(offset) <= MAX_OFFSET_SECONDS else "offset_exceeds_one_hour"
        entry["reason"] = reason
        entry["kept"] = reason == "kept"
        if entry["kept"]:
            kept.append(event)
        audit.append(entry)
    return kept, audit


def generate_processed_file(events, output_file):
    """Write selected event blocks, preserving all supplied measurement rows."""
    lines = ["CME events selected by quality and start-time offset", f"Event count: {len(events)}", ""]
    for event in events:
        lines.extend([HEADER + " " + event["time"], event["quality_index"],
                      "# HEIGHT DATE TIME ANGLE TEL FC COL ROW", *event["trajectory_data"], ""])
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", help="One or more event-block text files")
    parser.add_argument("--output", required=True, help="Selected event-block text file")
    parser.add_argument("--report", help="Audit JSON; defaults to OUTPUT.audit.json")
    args = parser.parse_args()
    inputs = [Path(value).resolve() for value in args.input]
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else output.with_suffix(".audit.json")
    if output in inputs or report_path in inputs or output == report_path:
        parser.error("Inputs, selected-event output and audit output must be distinct files")
    selected, audit, counts = [], [], Counter()
    for path in inputs:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
        events = parse_source_file(path)
        if not events:
            parser.error(f"No event blocks found: {path}")
        retained, decisions = select_events(events)
        selected.extend(retained)
        for entry in decisions:
            entry["source_file"] = path.name
        audit.extend(decisions)
        counts.update(entry["reason"] for entry in decisions)
    generate_processed_file(selected, output)
    report = {"criteria": {"minimum_quality_index": MIN_QUALITY,
                            "maximum_absolute_offset_seconds": MAX_OFFSET_SECONDS,
                            "one_hour_boundary": "inclusive",
                            "time_basis": "earliest C2 timestamp in the supplied event record",
                            "missing_or_invalid_required_values": "reject"},
              "summary": {"input_events": len(audit), "kept_events": len(selected),
                          "removed_events": len(audit) - len(selected), "reasons": dict(counts)},
              "events": audit}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Selected events: {output}\nAudit report: {report_path}")


if __name__ == "__main__":
    main()
