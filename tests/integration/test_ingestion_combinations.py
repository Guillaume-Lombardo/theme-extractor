from __future__ import annotations

import json

from theme_extractor.cli import main

_EXPECTED_INGESTED_DOCS = 2


def test_ingest_combines_manual_and_auto_stopwords(tmp_path, capsys) -> None:
    doc_a = tmp_path / "a.txt"
    doc_b = tmp_path / "b.txt"

    doc_a.write_text("alpha beta beta gamma", encoding="utf-8")
    doc_b.write_text("alpha beta delta", encoding="utf-8")

    exit_code = main(
        [
            "ingest",
            "--input",
            str(tmp_path),
            "--recursive",
            "--manual-stopwords",
            "alpha",
            "--auto-stopwords",
            "--auto-stopwords-min-doc-ratio",
            "1.0",
            "--auto-stopwords-max-terms",
            "10",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["processed_documents"] == _EXPECTED_INGESTED_DOCS
    assert payload["manual_stopwords"] == ["alpha"]
    assert "beta" in payload["auto_stopwords"]


def test_ingest_cleaning_options_are_applied(tmp_path, capsys) -> None:
    doc = tmp_path / "index.html"
    doc.write_text("<html><body>Résumé https://example.com Contact: x@y.z</body></html>", encoding="utf-8")

    exit_code = main(
        [
            "ingest",
            "--input",
            str(doc),
            "--cleaning-options",
            "html_strip,boilerplate,accent_normalization,whitespace",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["processed_documents"] == 1
    preview = payload["documents"][0]["clean_text_preview"].lower()
    assert "resume" in preview
    assert "https" not in preview
    assert "x@y.z" not in preview
