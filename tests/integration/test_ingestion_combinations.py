from __future__ import annotations

import csv
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
            "--auto-stopwords-min-corpus-ratio",
            "0.1",
            "--auto-stopwords-max-terms",
            "10",
            "--no-default-stopwords",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["processed_documents"] == _EXPECTED_INGESTED_DOCS
    assert payload["manual_stopwords"] == ["alpha"]
    assert "beta" in payload["auto_stopwords"]


def test_ingest_manual_stopwords_from_csv_file(tmp_path, capsys) -> None:
    doc = tmp_path / "a.txt"
    csv_file = tmp_path / "stopwords.csv"
    doc.write_text("facture copropriete copropriete alpha", encoding="utf-8")
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stopword"])
        writer.writerow(["copropriete"])

    exit_code = main(
        [
            "ingest",
            "--input",
            str(doc),
            "--manual-stopwords-file",
            str(csv_file),
            "--no-default-stopwords",
        ],
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manual_stopwords"] == ["copropriete"]
    assert payload["manual_stopwords_files"] == [str(csv_file.resolve())]


def test_ingest_manual_stopwords_from_json_file(tmp_path, capsys) -> None:
    doc = tmp_path / "a.txt"
    json_file = tmp_path / "stopwords.json"
    doc.write_text("facture copropriete copropriete alpha", encoding="utf-8")
    json_file.write_text('{"stopwords": ["copropriété"]}', encoding="utf-8")

    exit_code = main(
        [
            "ingest",
            "--input",
            str(doc),
            "--manual-stopwords-file",
            str(json_file),
            "--no-default-stopwords",
        ],
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manual_stopwords"] == ["copropriete"]
    assert payload["manual_stopwords_files"] == [str(json_file.resolve())]


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


def test_ingest_streaming_and_non_streaming_modes_match_counts(tmp_path, capsys) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text("alpha alpha beta gamma", encoding="utf-8")

    streaming_exit = main(
        [
            "ingest",
            "--input",
            str(doc),
            "--manual-stopwords",
            "alpha",
            "--no-default-stopwords",
            "--streaming-mode",
        ],
    )
    assert streaming_exit == 0
    streaming_payload = json.loads(capsys.readouterr().out)

    classic_exit = main(
        [
            "ingest",
            "--input",
            str(doc),
            "--manual-stopwords",
            "alpha",
            "--no-default-stopwords",
            "--no-streaming-mode",
        ],
    )
    assert classic_exit == 0
    classic_payload = json.loads(capsys.readouterr().out)

    assert streaming_payload["streaming_mode"] is True
    assert classic_payload["streaming_mode"] is False
    assert streaming_payload["documents"][0]["token_count"] == classic_payload["documents"][0]["token_count"]
    assert (
        streaming_payload["documents"][0]["removed_stopword_count"]
        == classic_payload["documents"][0]["removed_stopword_count"]
    )
