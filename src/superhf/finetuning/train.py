"""
SuperHF finetuning training script.

Only used for testing our linter, formatter, and type checker for now.
"""


def main() -> None:
    """Main function."""
    print("Hello world!")
    function_with_types(5)


def function_with_types(number: int) -> int:
    """Function with types."""
    return number + 5


if __name__ == "__main__":
    main()
