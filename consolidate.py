#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


EXCLUDE_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".vscode",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "out",
    ".next",
    ".svelte-kit",
    ".astro",
    ".cache",
    ".turbo",
}

# Only include text/code files that are typically relevant in repos like this.
INCLUDE_EXTS = {
    ".py",
    ".js",
    ".cjs",
    ".mjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".svelte",
    ".astro",
    ".json",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".html",
    ".htm",
    ".mdx",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".txt",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".sql",
    ".graphql",
    ".gql",
    ".env",
}

EXCLUDE_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "consolidate.py",
}


def _language_for_fence(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".cjs": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".svelte": "svelte",
        ".astro": "astro",
        ".json": "json",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".html": "html",
        ".htm": "html",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".ps1": "powershell",
        ".sql": "sql",
        ".graphql": "graphql",
        ".gql": "graphql",
        ".env": "dotenv",
    }.get(ext, "")


def _should_include_file(path: Path, output_names: set[str]) -> bool:
    name = path.name
    if name in output_names:
        return False
    if name in EXCLUDE_FILE_NAMES:
        return False
    # skip common lockfiles even if renamed
    if name.endswith(".lock"):
        return False
    ext = path.suffix.lower()
    return ext in INCLUDE_EXTS


def _iter_files(root: Path, output_names: set[str]) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES and not d.startswith(".git")]
        for filename in filenames:
            p = Path(dirpath) / filename
            if _should_include_file(p, output_names):
                files.append(p)
    files.sort(key=lambda p: str(p).lower())
    return files


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _write_markdown(output_path: Path, files: list[Path], base_dir: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Consolidated Code: `{base_dir.name}`")
    lines.append("")
    for file_path in files:
        rel = file_path.relative_to(base_dir).as_posix()
        lines.append(f"## `{rel}`")
        lines.append("")
        lang = _language_for_fence(file_path)
        fence = f"```{lang}".rstrip()
        lines.append(fence)
        lines.append(_read_text(file_path).rstrip("\n"))
        lines.append("```")
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    base_dir = Path.cwd()

    out_all = base_dir / "consolidated_all.md"
    out_backend = base_dir / "consolidated_backend.md"
    out_frontend = base_dir / "consolidated_frontend.md"
    output_names = {out_all.name, out_backend.name, out_frontend.name}

    backend_dir = base_dir / "backend"
    frontend_dir = base_dir / "frontend"

    all_files = _iter_files(base_dir, output_names)
    backend_files = _iter_files(backend_dir, output_names) if backend_dir.is_dir() else []
    frontend_files = _iter_files(frontend_dir, output_names) if frontend_dir.is_dir() else []

    _write_markdown(out_all, all_files, base_dir)
    _write_markdown(out_backend, backend_files, backend_dir if backend_dir.is_dir() else base_dir)
    _write_markdown(out_frontend, frontend_files, frontend_dir if frontend_dir.is_dir() else base_dir)

    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_backend}")
    print(f"Wrote: {out_frontend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
