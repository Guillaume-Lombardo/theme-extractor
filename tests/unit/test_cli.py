from __future__ import annotations

from theme_extractor.cli import main


def test_main_prints_expected_output(capsys) -> None:
    main()
    captured = capsys.readouterr().out.splitlines()

    assert captured == [
        "Welcome to the Theme Extractor CLI!",
        "This is a placeholder for the actual CLI implementation.",
        "Please refer to the documentation for usage instructions.",
    ]
