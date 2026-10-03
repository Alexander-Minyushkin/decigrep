"""Semantic line matching against Ollama decision models (System One).

The core idea: for every input line the model receives a ``choice`` question
``"Does the line match the pattern ...?"`` with the configured criteria
(``yes`` / ``no`` by default). A line is considered a match when the
probability of the first (positive) criterion exceeds the threshold.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .client import SystemOneClient, SystemOneError

#: Maximum size of ``state`` for image-less requests (docs/Ollama-SystemOne.md).
MAX_STATE_BYTES = 64 * 1024

#: Name of the single question inside every System One request.
QUESTION_NAME = "match"


@dataclass(frozen=True)
class Criterion:
    """One named option of a ``choice`` question."""

    key: str
    description: Optional[str]


def parse_criteria(spec: str) -> List[Criterion]:
    """Parse comma-separated ``key[:description]`` criteria.

    Examples::

        "yes,no"                        -> yes, no (descriptions fall back to keys)
        "yes:It matches,no:It does not" -> two criteria with descriptions

    Raises :class:`ValueError` for specs that violate the System One API
    limits (2..26 unique, non-blank keys).
    """
    criteria: List[Criterion] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, _, description = part.partition(":")
            key = key.strip()
            description = description.strip()
            if not key:
                raise ValueError(f"invalid criterion {part!r}: the key is empty")
        else:
            key, description = part, None
        criteria.append(Criterion(key, description))

    if len(criteria) < 2:
        raise ValueError("at least two criteria are required (e.g. --criteria yes,no)")
    if len(criteria) > 26:
        raise ValueError("the System One API allows at most 26 criteria")
    if len({c.key for c in criteria}) != len(criteria):
        raise ValueError("criterion keys must be unique")
    return criteria


DEFAULT_CRITERIA: Sequence[Criterion] = parse_criteria("yes,no")

#: Default question instructions; ``{pattern}`` is replaced with the pattern.
DEFAULT_INSTRUCTIONS = (
    'Does the line match the pattern "{pattern}"? '
    "Judge by meaning and topic, not just exact wording."
)


def build_question(
    pattern: str,
    criteria: Sequence[Criterion],
    instructions: Optional[str] = None,
) -> Mapping[str, object]:
    """Build the ``questions`` payload for a single-line decision request."""
    display_pattern = " ".join(pattern.splitlines())
    text = instructions or DEFAULT_INSTRUCTIONS
    text = text.replace("{pattern}", display_pattern)
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": text,
            "criteria": {c.key: c.description for c in criteria},
        }
    }


def decide_match(probabilities: Mapping[str, float], positive_key: str, threshold: float) -> bool:
    """Return True when ``P(positive_key) > threshold`` (strictly greater)."""
    return probabilities.get(positive_key, 0.0) > threshold


def skip_reason(line: str) -> Optional[str]:
    """Return a reason when *line* cannot be sent to the API, else ``None``.

    The API requires a non-empty ``state`` and rejects requests above 64 KiB.
    """
    state = line.rstrip("\n")
    if not state:
        return "blank line skipped"
    if len(state.encode("utf-8")) > MAX_STATE_BYTES:
        return "line too long (exceeds the 64 KiB request limit), skipped"
    return None


@dataclass(frozen=True)
class Decision:
    """Result of evaluating one input line."""

    line_number: int
    line: str
    match: bool
    probabilities: Mapping[str, float]
    error: Optional[str] = None
    skipped: bool = False


def evaluate_line(
    client: SystemOneClient,
    *,
    model: str,
    pattern: str,
    line: str,
    criteria: Sequence[Criterion],
    instructions: Optional[str],
    keep_alive: Optional[str],
    retries: int,
    retry_delay: float,
) -> Tuple[Mapping[str, float], Optional[str]]:
    """Ask Ollama whether *line* matches *pattern*.

    Returns ``(probabilities, error)``; exactly one of them is meaningful on
    success/failure respectively. Transient failures are retried with
    exponential backoff up to *retries* times.
    """
    state = line.rstrip("\n")
    question = build_question(pattern, criteria, instructions)
    for attempt in range(retries + 1):
        try:
            response = client.decide(
                model=model, state=state, questions=question, keep_alive=keep_alive
            )
            answer = response["answers"][QUESTION_NAME]
            return dict(answer["probabilities"]), None
        except (SystemOneError, KeyError, TypeError, ValueError) as exc:
            if attempt >= retries:
                if isinstance(exc, SystemOneError):
                    message = str(exc)
                else:
                    message = f"unexpected API response: {exc!r}"
                return {}, message
            time.sleep(retry_delay * (2 ** attempt))
    return {}, "unreachable"  # pragma: no cover


def scan_lines(
    client: SystemOneClient,
    lines: Iterable[Tuple[int, str]],
    *,
    model: str,
    pattern: str,
    criteria: Sequence[Criterion],
    positive_key: str,
    threshold: float,
    instructions: Optional[str] = None,
    keep_alive: Optional[str] = None,
    retries: int = 2,
    retry_delay: float = 1.0,
    workers: int = 1,
) -> Iterator[Decision]:
    """Evaluate every line and yield a :class:`Decision` in input order.

    With ``workers > 1``, lines are evaluated concurrently while results are
    still yielded in the original file order.
    """

    def work(item: Tuple[int, str]) -> Decision:
        number, line = item
        reason = skip_reason(line)
        if reason:
            return Decision(number, line, match=False, probabilities={}, error=reason, skipped=True)
        probabilities, error = evaluate_line(
            client,
            model=model,
            pattern=pattern,
            line=line,
            criteria=criteria,
            instructions=instructions,
            keep_alive=keep_alive,
            retries=retries,
            retry_delay=retry_delay,
        )
        match = error is None and decide_match(probabilities, positive_key, threshold)
        return Decision(number, line, match=match, probabilities=probabilities, error=error)

    if workers <= 1:
        for item in lines:
            yield work(item)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(work, lines)