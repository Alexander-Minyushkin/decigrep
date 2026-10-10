"""HTTP client for the Ollama System One (decision models) API.

The wire protocol is documented in ``docs/Ollama-SystemOne.md``. In short:

* ``POST <base_url>/v1/systemone`` with ``model``, ``state`` and
  ``questions`` returns one JSON response with per-question answers.
* No API key is required for local requests.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 60.0


class SystemOneError(RuntimeError):
    """Raised when the Ollama System One API cannot be used."""


class SystemOneClient:
    """Minimal thread-safe client for ``POST /v1/systemone``.

    A separate :class:`requests.Session` is kept per thread so the same
    client instance can safely be shared between worker threads.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = requests.Session()
        return session

    def decide(
        self,
        model: str,
        state: str,
        questions: Mapping[str, Any],
        keep_alive: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send one System One request and return the parsed JSON response."""
        payload: dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": dict(questions),
        }
        if keep_alive is not None:
            # The API accepts either a duration string ("5m") or a number of
            # seconds. A numeric string like "-1" (keep loaded) must be sent as
            # a JSON number, otherwise the server rejects it as a bad duration.
            try:
                payload["keep_alive"] = int(keep_alive)
            except (TypeError, ValueError):
                payload["keep_alive"] = keep_alive

        url = f"{self.base_url}/v1/systemone"
        try:
            response = self._session().post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise SystemOneError(
                f"could not reach Ollama at {self.base_url}: {exc}\n"
                "Is Ollama running? Start it with `ollama serve`."
            ) from exc

        if response.status_code != 200:
            raise SystemOneError(self._format_error(response))

        try:
            return response.json()
        except ValueError as exc:
            raise SystemOneError(
                f"Ollama returned an invalid JSON response: {response.text[:300]!r}"
            ) from exc

    @staticmethod
    def _format_error(response: requests.Response) -> str:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text

        hint = ""
        if response.status_code == 404:
            hint = " Download the model first, e.g. `ollama pull tev1:0.8b`."
        elif response.status_code == 400:
            hint = (
                " Check that the model supports System One scoring (e.g. "
                "`ollama pull tev1:0.8b`) and that request options such as "
                "--keep-alive are valid."
            )
        return (
            f"Ollama returned HTTP {response.status_code} for "
            f"{response.request.url}: {detail}{hint}"
        )