"""Offline tests for quality selection and the one-hour event criterion."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fix_processed_files as selection
import download


def event(quality="#QUALITY_INDEX: 3 (Typical)", start="2023-07-29T12:00:00", rows=None):
    return {"time": start, "quality_index": quality,
            "trajectory_data": rows if rows is not None else ["3.0 2023/07/29 12:12:00 90 C2 0 0 0"]}


class EventSelectionTests(unittest.TestCase):
    def decision(self, item):
        kept, audit = selection.select_events([item])
        self.assertEqual(bool(kept), audit[0]["kept"])
        return audit[0]

    def test_quality_values(self):
        for quality, expected in [(None, False), ("", False), ("#QUALITY_INDEX: bad", False),
                                  ("#QUALITY_INDEX: 3.5", False), ("#QUALITY_INDEX: 2", False),
                                  ("#QUALITY_INDEX: -3", False), ("#QUALITY_INDEX: 3", True),
                                  ("#QUALITY INDEX: 4 (Good)", True), ("#QUALITY_INDEX: 5", True)]:
            with self.subTest(quality=quality):
                kept, removed = selection.filter_events_by_quality([event(quality=quality)])
                self.assertEqual(bool(kept), expected)
                self.assertEqual(removed, int(not expected))
                self.assertEqual(self.decision(event(quality=quality))["kept"], expected)

    def test_cannot_lower_quality_threshold(self):
        with self.assertRaises(ValueError):
            selection.filter_events_by_quality([event()], 2)

    def test_positive_one_hour_boundary(self):
        for stamp, expected in [("12:59:59", True), ("13:00:00", True),
                                ("13:00:00.001", False), ("13:00:01", False)]:
            with self.subTest(stamp=stamp):
                item = event(rows=[f"3 2023/07/29 {stamp} 90 C2 0"])
                self.assertEqual(self.decision(item)["kept"], expected)

    def test_absolute_negative_offset_boundary(self):
        self.assertTrue(self.decision(event(rows=["3 2023/07/29 11:00:00 90 C2 0"]))["kept"])
        self.assertFalse(self.decision(event(rows=["3 2023/07/29 10:59:59 90 C2 0"]))["kept"])

    def test_midnight_and_year_boundary(self):
        item = event(start="2023-12-31T23:30:00", rows=["3 2024/01/01 00:30:00 90 C2 0"])
        decision = self.decision(item)
        self.assertTrue(decision["kept"])
        self.assertEqual(decision["absolute_offset_seconds"], 3600)

    def test_utc_offsets(self):
        item = event(start="2023-07-29T20:00:00+08:00")
        self.assertEqual(self.decision(item)["signed_offset_seconds"], 720)

    def test_first_means_earliest_c2_not_first_row_or_c3(self):
        item = event(rows=["4 2023/07/29 16:00:00 90 C3 0",
                           "4 2023/07/29 14:00:00 90 C2 0",
                           "3 2023/07/29 12:20:00 90 C2 0"])
        self.assertEqual(self.decision(item)["signed_offset_seconds"], 1200)

    def test_numeric_detector_label(self):
        self.assertTrue(self.decision(event(rows=["3 2023/07/29 12:20:00 90 2 0"]))["kept"])

    def test_later_observations_not_truncated_and_input_not_mutated(self):
        item = event(rows=["3 2023/07/29 12:12:00 90 C2 0", "5 2023/07/29 16:00:00 90 C2 0"])
        before = deepcopy(item)
        kept, _ = selection.select_events([item])
        self.assertEqual(kept[0], before)
        self.assertEqual(item, before)

    def test_missing_or_invalid_time_data(self):
        cases = [(event(start="not-a-time"), "invalid_event_time"),
                 (event(start="2023-02-30T12:00:00"), "invalid_event_time"),
                 (event(rows=[]), "no_c2_observations"),
                 (event(rows=["3 2023/07/29 12:12:00 90 C3 0"]), "no_c2_observations"),
                 (event(rows=["3 2023/02/30 12:12:00 90 C2 0"]), "invalid_c2_time"),
                 (event(rows=["3 invalid 12:12:00 90 C2 0", "4 2023/07/29 12:30:00 90 C2 0"]), "invalid_c2_time")]
        for item, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(self.decision(item)["reason"], reason)

    def test_parse_supported_formats_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.txt"
            path.write_text("Event 1:\nFirst C2 Appearance Date Time [UT]: 2023/07/29 12:00:00\n"
                            "#QUALITY_INDEX: 3 (Typical)\n# HEIGHT DATE TIME ANGLE TEL FC\n"
                            "3 2023/07/29 12:12:00 90 C2 0\n\n"
                            "2023-07-30T12:00:00\n#QUALITY INDEX: 4 (Good)\n"
                            "3 2023/07/30 12:12:00 90 C2 0\n", encoding="utf-8")
            events = selection.parse_source_file(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(len(selection.select_events(events)[0]), 2)
            output = Path(tmp) / "selected.txt"
            selection.generate_processed_file(events, output)
            self.assertEqual(selection.parse_source_file(output), events)

    def test_conflicting_quality_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conflict.txt"
            path.write_text("First C2 Appearance Date Time [UT]: 2023/07/29 12:00:00\n"
                            "#QUALITY_INDEX: 2\n#QUALITY_INDEX: 4\n"
                            "3 2023/07/29 12:12:00 90 C2 0\n", encoding="utf-8")
            self.assertEqual(self.decision(selection.parse_source_file(path)[0])["reason"], "invalid_quality")

    def test_cli_example_and_rejection_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "selected.txt"
            result = subprocess.run([sys.executable, str(ROOT / "fix_processed_files.py"),
                                     "--input", str(ROOT / "examples/catalog_example.txt"),
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["summary"]["input_events"], 5)
            self.assertEqual(audit["summary"]["kept_events"], 2)
            self.assertEqual(audit["summary"]["reasons"], {"kept": 2, "quality_below_3": 1,
                                                            "offset_exceeds_one_hour": 1, "missing_quality": 1})
            self.assertEqual(len(selection.parse_source_file(output)), 2)

    def test_cli_does_not_overwrite_input(self):
        source = ROOT / "examples/catalog_example.txt"
        original = source.read_bytes()
        result = subprocess.run([sys.executable, str(ROOT / "fix_processed_files.py"),
                                 "--input", str(source), "--output", str(source)], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(source.read_bytes(), original)

    def test_download_validation_without_network(self):
        with self.assertRaises(ValueError):
            download.get_lasco_fits("2023-07-29T13:00:00", "2023-07-29T12:00:00", "unused")
        with self.assertRaises(ValueError):
            download.get_lasco_fits("2023-07-29T12:00:00", "2023-07-29T13:00:00", "unused", "invalid")


if __name__ == "__main__":
    unittest.main()
