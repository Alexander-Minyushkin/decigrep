"""Command-line interface for DeciGrep."""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Iterator, List, Optional, Sequence, Tuple

from . import __version__
from .client import SystemOneClient
from .matcher import (
    DEFAULT_CRITERIA,
    parse_criteria,
    scan_lines,
)

#: How often to refresh the stderr progress line.
_PROGRESS_EVERY = 250
#: How many per-line error details to print before summarizing.
_MAX_ERROR_DETAILS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decigrep",
        description=(
            "Search a file for lines that match a pattern, deciding each match "
            "with an Ollama decision model (System One). The pattern is "
            "interpreted semantically, not as a regular expression."
        ),
        epilog=(
            "Exit status: 0 if at least one line was printed, 1 if no line was "
            "printed, 2 on error. Requires Ollama v0.35+ with a decision model "
            "such as nimble (`ollama pull nimble`)."
        ),
    )
    parser.add_argument("pattern", help="pattern to match lines against")
    parser.add_argument(
        "file",
        nargs="?",
        default="-",
        help="input file to scan, or '-' for standard input (default: -)",
    )
    parser.add_argument(
        "-m", "--model",
        default="nimble",
        help="Ollama decision model to use (default: nimble)",
    )
    parser.add_argument(
        "-u", "--url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        metavar="P",
        help="print a line when the probability of the first (positive) "
             "criterion is greater than P (default: 0.5)",
    )
    parser.add_argument(
        "-c", "--criteria",
        default="yes,no",
        metavar="SPEC",
        help="comma-separated key[:description] criteria; the first key is the "
             "positive one (default: yes,no)",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        metavar="TEXT",
        help="custom question instructions for the model; '{pattern}' is "
             "replaced with the search pattern (a sensible default is used "
             "when omitted)",
    )
    parser.add_argument(
        "-n", "--line-number",
        action="store_true",
        help="prefix each printed line with its 1-based line number",
    )
    parser.add_argument(
        "-v", "--invert-match",
        action="store_true",
        help="print lines that do NOT match",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        metavar="N",
        help="number of concurrent Ollama requests (default: 1)",
    )
    parser.add_argument(
        "-r", "--retries",
        type=int,
        default=2,
        metavar="N",
        help="retries per line after a failed request (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="S",
        help="per-request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--keep-alive",
        default="-1",
        metavar="VALUE",
        help="Ollama keep_alive value; -1 keeps the model loaded between "
             "requests (default: -1)",
    )
    parser.add_argument(
        "-V", "--verbose",
        action="store_true",
        help="print per-line probabilities and skip reasons to stderr",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="suppress progress and warnings on stderr",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def iter_input_lines(file_path: str) -> Iterator[Tuple[int, str]]:
    """Yield ``(1-based line number, line)`` pairs.

    Original line endings are preserved (``newline=""``); for standard input
    the platform's normal text-mode translation applies.
    """
    if file_path == "-":
        yield from enumerate(sys.stdin, start=1)
        return
    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from enumerate(handle, start=1)


def _print_line(number: int, line: str, show_numbers: bool) -> None:
    text = line if line.endswith("\n") else line + "\n"
    if show_numbers:
        sys.stdout.write(f"{number}:{text}")
    else:
        sys.stdout.write(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be greater than 0 and at most 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.retries < 0:
        parser.error("--retries must not be negative")
    try:
        criteria = parse_criteria(args.criteria)
    except ValueError as exc:
        parser.error(str(exc))
    positive_key = criteria[0].key

    if args.file != "-" and not os.path.isfile(args.file):
        print(f"decigrep: {args.file}: no such file", file=sys.stderr)
        return 2

    # Write exact bytes to stdout (no newline translation) so lines keep their
    # original \r\n or \n endings regardless of the platform.
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding=getattr(sys.stdout, "encoding", None) or "utf-8",
            newline="",
        )

    client = SystemOneClient(base_url=args.url, timeout=args.timeout)
    isatty = getattr(sys.stderr, "isatty", lambda: False)
    show_progress = (not args.quiet) and bool(isatty())

    if args.verbose and not args.quiet:
        print(
            f"decigrep: contacting Ollama at {args.url} with model "
            f"{args.model} (the first request can take a few seconds "
            "while Ollama starts and loads the model)",
            file=sys.stderr,
        )

    printed = 0
    total = 0
    errors = 0
    error_details_shown = 0

    try:
        for total, decision in enumerate(
            scan_lines(
                client,
                iter_input_lines(args.file),
                model=args.model,
                pattern=args.pattern,
                criteria=criteria,
                positive_key=positive_key,
                threshold=args.threshold,
                instructions=args.instructions,
                keep_alive=args.keep_alive,
                retries=args.retries,
                workers=args.workers,
            ),
            start=1,
        ):
            if decision.skipped:
                if args.verbose and not args.quiet:
                    print(
                        f"decigrep: line {decision.line_number}: {decision.error}",
                        file=sys.stderr,
                    )
                continue
            if decision.error:
                errors += 1
                if not args.quiet and error_details_shown < _MAX_ERROR_DETAILS:
                    error_details_shown += 1
                    print(
                        f"decigrep: line {decision.line_number}: {decision.error}",
                        file=sys.stderr,
                    )
                continue
            if args.verbose and not args.quiet:
                probabilities = ", ".join(
                    f"{key}={value:.4f}" for key, value in decision.probabilities.items()
                )
                verdict = "MATCH" if decision.match else "no match"
                print(
                    f"decigrep: line {decision.line_number}: [{verdict}] {probabilities}",
                    file=sys.stderr,
                )
            if decision.match != args.invert_match:
                printed += 1
                _print_line(decision.line_number, decision.line, args.line_number)
            if show_progress and total % _PROGRESS_EVERY == 0:
                sys.stderr.write(f"\rdecigrep: processed {total} lines, {printed} printed")
                sys.stderr.flush()
    except OSError as exc:
        print(f"decigrep: {args.file}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    finally:
        if show_progress:
            sys.stderr.write(f"\rdecigrep: processed {total} lines, {printed} printed\n")
            sys.stderr.flush()

    if errors:
        remaining = errors - error_details_shown
        if not args.quiet and remaining > 0:
            print(f"decigrep: {remaining} more line(s) failed", file=sys.stderr)
        return 2
    return 0 if printed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())