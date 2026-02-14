"""CLI entrypoint execution flow."""

from __future__ import annotations

from theme_extractor.cli.argument_parser import build_parser
from theme_extractor.cli.common_runtime import apply_proxy_environment, emit_payload


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        int: Process exit code.

    """
    parser = build_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    apply_proxy_environment(getattr(args, "proxy_url", None))
    payload = handler(args)
    emit_payload(payload=payload, output=args.output)
    return 0
