"""
A script to iteratively convert flepiMoP CLI man pages into markdown files.
"""

import subprocess
from pathlib import Path


def main():
    """
    Main execution function.
    """

    # Path to find man pages, path to output files
    script_dir = Path(__file__).resolve().parent
    folder_path = script_dir / "cli_man"
    output_dir = folder_path / "markdown-files"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in folder_path.rglob("*.1"):
        if file.is_file():
            # Create file name
            output_md = output_dir / (file.stem + ".md")

            # Construct pandoc command and run
            command = [
                "pandoc",
                str(file),
                "-f",
                "man",
                "-t",
                "markdown",
                "-o",
                str(output_md),
            ]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
