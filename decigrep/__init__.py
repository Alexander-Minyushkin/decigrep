"""DeciGrep: grep-like search powered by Ollama decision models.

DeciGrep scans a file line by line and asks an Ollama System One decision
model (e.g. ``nimble``) whether each line matches a given pattern. Lines
whose positive criterion probability exceeds the confidence threshold are
printed to standard output.
"""

__version__ = "0.1.0"