from .denoise import denoise_tensor

__all__ = ["denoise_tensor"]


def main() -> None:
    from .cli import main as cli_main

    cli_main()
