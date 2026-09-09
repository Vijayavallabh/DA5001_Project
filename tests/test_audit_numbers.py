"""The manuscript number audit must not manufacture or swallow numbers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.audit_numbers import literals  # noqa: E402


def write(tmp_path, text):
    p = tmp_path / "s.tex"; p.write_text(text); return str(p)


def test_thousands_separator_is_one_number_not_two(tmp_path):
    """LaTeX writes 32{,}011; scanning before stripping it reported the fragments 32 and 011."""
    got = [n for n, _ in literals(write(tmp_path, "vocabulary $32{,}011$ tokens\n"))]
    assert got == ["32011"]


def test_a_real_comma_still_separates(tmp_path):
    """[0,1] is two numbers. Stripping every comma would manufacture 01."""
    got = [n for n, _ in literals(write(tmp_path, r"$\theta \in [0,1]$" + "\n"))]
    assert got == ["0", "1"]


def test_commented_lines_are_skipped(tmp_path):
    assert literals(write(tmp_path, "% a note about $999$\n$7$\n")) == [("7", 2)]


def test_line_numbers_are_reported(tmp_path):
    assert literals(write(tmp_path, "text\n\nvalue $1.25$\n")) == [("1.25", 3)]


def test_decimals_and_negatives_survive(tmp_path):
    got = [n for n, _ in literals(write(tmp_path, "$-0.45$ and $2.35$\n"))]
    assert got == ["-0.45", "2.35"]
