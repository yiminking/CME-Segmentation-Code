#Request LASCO FITS files with a Level(1) archive filter for an interval.
import argparse
from pathlib import Path
from fix_processed_files import parse_utc


def get_lasco_fits(time_start, time_end, workdir, detector="c2"):
    """Search LASCO Level 1 with SunPy Fido and download matching files."""
    start, end = parse_utc(time_start), parse_utc(time_end)
    if end < start:
        raise ValueError("End time must not precede start time")
    detector = detector.lower()
    if detector not in {"c2", "c3"}:
        raise ValueError("Detector must be c2 or c3")

    from sunpy.net import Fido, attrs as a
    detector_attr = a.Detector.c2 if detector == "c2" else a.Detector.c3
    query = Fido.search(a.Time(start, end), a.Instrument.lasco, detector_attr, a.Level(1))
    output = Path(workdir)
    output.mkdir(parents=True, exist_ok=True)
    files = Fido.fetch(query, path=str(output / "{file}"))
    if getattr(files, "errors", None):
        raise RuntimeError(f"Some downloads failed: {files.errors}")
    if not len(files):
        raise RuntimeError("No files were downloaded for this interval")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC start time, e.g. 2023-07-29T12:00:00")
    parser.add_argument("--end", required=True, help="UTC end time")
    parser.add_argument("--output", required=True, help="Download directory")
    parser.add_argument("--detector", choices=("c2", "c3"), default="c2")
    args = parser.parse_args()
    print(get_lasco_fits(args.start, args.end, args.output, args.detector))


if __name__ == "__main__":
    main()
