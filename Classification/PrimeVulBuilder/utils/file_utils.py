from pathlib import Path
from typing import Iterable


def ensure_directory(path: Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The same directory path.
    """

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def file_exists(path: Path) -> bool:
    """
    Return True if the path exists and is a regular file.
    """

    path = Path(path)

    return (
        path.exists()
        and path.is_file()
    )


def directory_exists(path: Path) -> bool:
    """
    Return True if the path exists and is a directory.
    """

    path = Path(path)

    return (
        path.exists()
        and path.is_dir()
    )


def read_text(
    path: Path,
    encoding: str = "utf-8",
) -> str:
    """
    Read a text file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If the path is not a file.

    OSError
        If the file cannot be read.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file, got: {path}"
        )

    try:
        return path.read_text(
            encoding=encoding
        )

    except OSError as exc:
        raise OSError(
            f"Could not read file: {path}"
        ) from exc


def write_text(
    path: Path,
    content: str,
    encoding: str = "utf-8",
) -> Path:
    """
    Write text to a file.

    Parent directories are created automatically.
    """

    path = Path(path)

    ensure_directory(
        path.parent
    )

    try:
        path.write_text(
            content,
            encoding=encoding,
        )

    except OSError as exc:
        raise OSError(
            f"Could not write file: {path}"
        ) from exc

    return path


def get_scenario_directories(
    dataset_root: Path,
) -> list[Path]:
    """
    Return immediate scenario directories below DATASET_ROOT.

    Expected structure:

        SFT_Dataset/
            scenario_001/
            scenario_002/
            scenario_003/

    Only directories directly inside dataset_root are returned.

    Results are sorted deterministically by directory name.
    """

    dataset_root = Path(
        dataset_root
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: "
            f"{dataset_root}"
        )

    if not dataset_root.is_dir():
        raise ValueError(
            f"Dataset root is not a directory: "
            f"{dataset_root}"
        )

    scenarios = [
        path
        for path in dataset_root.iterdir()
        if path.is_dir()
    ]

    return sorted(
        scenarios,
        key=lambda path: path.name.lower(),
    )


def get_files_recursive(
    directory: Path,
) -> list[Path]:
    """
    Return all files recursively below a directory.

    Results are sorted deterministically.
    """

    directory = Path(
        directory
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Expected a directory, got: "
            f"{directory}"
        )

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file()
    ]

    return sorted(
        files,
        key=lambda path: str(path).lower(),
    )


def get_relative_path(
    path: Path,
    root: Path,
) -> str:
    """
    Return a stable POSIX-style path relative to root.

    Example
    -------
    root:
        F:/Capstone/SFT/SFT_Dataset/scenario_001

    path:
        F:/Capstone/SFT/SFT_Dataset/scenario_001/files/app.py

    result:
        files/app.py
    """

    path = Path(path).resolve()
    root = Path(root).resolve()

    try:
        relative = path.relative_to(
            root
        )

    except ValueError as exc:
        raise ValueError(
            f"Path '{path}' is not inside "
            f"root '{root}'."
        ) from exc

    return relative.as_posix()


def normalize_path_string(
    path: str,
) -> str:
    """
    Normalize a path string for comparison.

    Converts Windows separators to POSIX separators.

    Examples
    --------
        .\\files\\backend\\app.py
        files\\backend\\app.py
        ./files/backend/app.py

    become:

        files/backend/app.py
    """

    if not isinstance(
        path,
        str,
    ):
        raise TypeError(
            "path must be a string."
        )

    normalized = path.strip()

    normalized = normalized.replace(
        "\\",
        "/",
    )

    while normalized.startswith(
        "./"
    ):
        normalized = normalized[2:]

    return normalized


def is_path_inside(
    path: Path,
    root: Path,
) -> bool:
    """
    Check whether path is located inside root.

    This is important when resolving file paths supplied by
    metadata.json.

    Returns False instead of raising if the path is outside root.
    """

    try:
        path = Path(
            path
        ).resolve()

        root = Path(
            root
        ).resolve()

        path.relative_to(
            root
        )

        return True

    except (
        ValueError,
        OSError,
    ):
        return False


def get_file_extension(
    path: Path,
) -> str:
    """
    Return the lowercase extension of a file.

    Example
    -------
        app.py -> .py
    """

    return Path(
        path
    ).suffix.lower()


def get_file_size(
    path: Path,
) -> int:
    """
    Return file size in bytes.
    """

    path = Path(
        path
    )

    if not path.is_file():
        raise ValueError(
            f"Expected a file: {path}"
        )

    return path.stat().st_size


def count_lines(
    content: str,
) -> int:
    """
    Count the number of lines in loaded text.
    """

    if not content:
        return 0

    return len(
        content.splitlines()
    )


def count_characters(
    content: str,
) -> int:
    """
    Count characters in loaded text.
    """

    if not content:
        return 0

    return len(
        content
    )


def safe_filename(
    filename: str,
) -> str:
    """
    Convert a filename into a filesystem-safe filename.

    Mainly intended for debug-output filenames.

    This function must NOT be used to modify actual source-code
    paths used by the dataset.
    """

    if not isinstance(
        filename,
        str,
    ):
        raise TypeError(
            "filename must be a string."
        )

    unsafe_characters = (
        '\\/:*?"<>|'
    )

    result = filename

    for character in unsafe_characters:
        result = result.replace(
            character,
            "_",
        )

    return result.strip()


def find_file(
    directory: Path,
    filename: str,
) -> Path | None:
    """
    Recursively search for a specific filename.

    Returns the first deterministic match.

    NOTE:
    The PrimeVul builder should normally use
    metadata.json -> relevant_files rather than relying on
    blind filename searching.
    """

    directory = Path(
        directory
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Expected a directory: "
            f"{directory}"
        )

    matches = sorted(
        directory.rglob(
            filename
        ),
        key=lambda path: str(
            path
        ).lower(),
    )

    if not matches:
        return None

    return matches[0]


def validate_required_files(
    directory: Path,
    filenames: Iterable[str],
) -> None:
    """
    Verify that required files exist inside directory.

    Raises FileNotFoundError if one or more files are missing.
    """

    directory = Path(
        directory
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Expected a directory: "
            f"{directory}"
        )

    missing = []

    for filename in filenames:

        path = (
            directory / filename
        )

        if not path.is_file():
            missing.append(
                filename
            )

    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join(
                f"  - {filename}"
                for filename in missing
            )
        )