"""Utilities for locating and compiling PRMS/GSFLOW executables.

The binary naming conventions mirror those in test_data/generate/conftest.py.
Compilation logic mirrors autotest/ci_local.sh.

PRMS binaries are not kept in the repository; they are compiled from the
sources in ``prms_src/`` with gfortran (supplied by environment.yml) the
first time they are needed.  GSFLOW is the exception: its source is not
part of this repository, so its binaries are checked in.

Supported platforms are Windows, Linux, and Apple Silicon macOS.  Intel
macOS is no longer supported.
"""

import contextlib
import os
import pathlib as pl
import shutil
import subprocess
import sys
from typing import Optional

#: PRMS versions in ``prms_src/`` that can be compiled from source, mapped
#: to their source directory names.
COMPILABLE_SOURCES = {
    "5.2.1": "prms5.2.1",
    "5.2.1.1": "prms5.2.1.1",
}

#: The version behind the plain "prms" executable, i.e. control files whose
#: ``executable_desc`` is "PRMS 4" or absent.
DEFAULT_PRMS_SOURCE = "5.2.1"


def _get_repo_root() -> pl.Path:
    from pywatershed.utils.notebook_utils import get_repo_root

    return get_repo_root()


def _get_bin_dir() -> pl.Path:
    return _get_repo_root() / "bin"


def _get_src_dir() -> pl.Path:
    return _get_repo_root() / "prms_src"


def _get_platform_tag() -> str:
    """Return the platform tag used in binary file names.

    Returns
    -------
    str
        One of ``"win"``, ``"mac_arm"``, or ``"linux"``.

    Raises
    ------
    ValueError
        When the platform is not recognised.
    """
    platform = sys.platform.lower()
    if platform == "win32":
        return "win"
    elif platform == "darwin":
        return "mac_arm"
    elif platform == "linux":
        return "linux"
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def _get_source_for_exe_desc(exe_desc: str) -> Optional[str]:
    """Return the compilable source version for *exe_desc*, or None.

    Parameters
    ----------
    exe_desc : str
        See :func:`get_prms_exe_name`.

    Returns
    -------
    str or None
        A key of :data:`COMPILABLE_SOURCES`, or ``None`` when *exe_desc*
        names an executable this repository cannot build (GSFLOW).
    """
    exe_desc_lower = exe_desc.lower()
    if "gsflow" in exe_desc_lower:
        return None
    if "5.2.1.1" in exe_desc_lower:
        return "5.2.1.1"
    return DEFAULT_PRMS_SOURCE


def get_prms_exe_name(exe_desc: str = "prms") -> str:
    """Return the binary filename for the given exe_desc and current platform.

    Parameters
    ----------
    exe_desc : str
        Description string from the control file's ``executable_desc`` field,
        or a plain version/type string such as ``"5.2.1.1"``, ``"gsflow"``,
        or ``"prms"``.

    Returns
    -------
    str
        The binary filename (no directory component).

    Raises
    ------
    ValueError
        When the platform is not recognised.
    """
    exe_desc_lower = exe_desc.lower()
    tag = _get_platform_tag()
    suffix = ".exe" if tag == "win" else ""

    if "gsflow" in exe_desc_lower:
        # GSFLOW source is not in this repository, so these binaries are
        # checked in rather than compiled on demand.
        return {
            "win": "gsflow_2.4.0_gfortran_windows_dbl_prec.exe",
            "mac_arm": "gsflow_2.4.0_gfortran_mac_arm_dbl_prec",
            "linux": "gsflow_2.4.0_gfortran_linux_dbl_prec",
        }[tag]

    elif "5.2.1.1" in exe_desc_lower:
        return f"prms_5.2.1.1_gfort_{tag}_dbl_prec{suffix}"

    else:
        # Default PRMS binary, version 5.2.1 -- the repository's
        # prms_src/prms5.2.1 source, with the cascades and G0-precision
        # CBH output patches, compiled on demand.
        return f"prms_{tag}_gfort_dbl_prec{suffix}"


def get_prms_exe_path(exe_desc: str = "prms") -> pl.Path:
    """Return the full path to the PRMS/GSFLOW binary in the repo bin/ dir.

    Parameters
    ----------
    exe_desc : str
        See :func:`get_prms_exe_name`.

    Returns
    -------
    pathlib.Path
        Absolute path to the binary (may or may not exist yet).
    """
    return (_get_bin_dir() / get_prms_exe_name(exe_desc)).resolve()


@contextlib.contextmanager
def _compile_lock(source: str):
    """Serialize compilation of *source* across processes.

    ``generate_test_data.py`` runs pytest with ``-n=auto``, so several
    workers can ask for the same binary at once.  Without a lock they would
    run ``make clean`` in the same source tree while another was compiling.
    ``filelock`` is a test dependency, not a runtime one; when it is absent
    compilation proceeds unserialized, which is correct for a single process.
    """
    try:
        from filelock import FileLock
    except ImportError:
        yield
        return

    bin_dir = _get_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(bin_dir / f".compile_prms_{source}.lock"), timeout=1800):
        yield


def compile_prms(
    source: str = DEFAULT_PRMS_SOURCE,
    force: bool = False,
) -> pl.Path:
    """Compile PRMS from source and return the path to the resulting binary.

    Parameters
    ----------
    source : str
        Source version to compile; a key of :data:`COMPILABLE_SOURCES`.
    force : bool
        If ``True``, recompile even when the binary already exists.
        Default is ``False`` (skip if binary present).

    Returns
    -------
    pathlib.Path
        Absolute path to the compiled binary.

    Raises
    ------
    ValueError
        If *source* is not a supported value.
    FileNotFoundError
        If make, gcc, or gfortran is not found on the PATH.
    RuntimeError
        If compilation succeeds but the expected output binary is missing.
    subprocess.CalledProcessError
        If any subprocess step fails.
    """
    if source not in COMPILABLE_SOURCES:
        raise ValueError(
            f"Unsupported source '{source}'. Supported sources are: "
            f"{sorted(COMPILABLE_SOURCES)}."
        )

    binary_path = get_prms_exe_path(source)
    src_dir = _get_src_dir() / COMPILABLE_SOURCES[source]

    if binary_path.exists() and not force:
        print(f"PRMS {source} binary already exists: {binary_path}")
        return binary_path

    with _compile_lock(source):
        # Another process may have compiled it while this one waited.
        if binary_path.exists() and not force:
            print(f"PRMS {source} binary already exists: {binary_path}")
            return binary_path

        print(
            f"\n{'*' * 30}\n"
            f"Compiling PRMS {source}\n"
            f"{'*' * 30}\n"
            f"Source dir : {src_dir}\n"
            f"Binary dest: {binary_path}\n"
        )

        # Prefer the compilers of the active python environment (e.g.
        # conda-forge gcc/gfortran) over whatever is on the PATH: local
        # PATHs can put stale/mismatched toolchains first. The compiler
        # names must remain plain "gcc"/"gfortran" as the makelists key
        # their flags on these exact names.
        sub_env = os.environ.copy()
        env_bin = pl.Path(sys.prefix) / "bin"
        if (env_bin / "gfortran").exists():
            sub_env["PATH"] = str(env_bin) + os.pathsep + sub_env["PATH"]

        missing = [
            tool
            for tool in ("make", "gcc", "gfortran")
            if shutil.which(tool, path=sub_env["PATH"]) is None
        ]
        if missing:
            raise FileNotFoundError(
                f"Cannot compile PRMS {source}: {', '.join(missing)} not "
                "found on PATH. A gcc/gfortran/make toolchain is required "
                "(e.g. from conda-forge); see DEVELOPER.md."
            )

        orig_dir = pl.Path.cwd()
        try:
            os.chdir(src_dir)

            # Clean previous build
            subprocess.run(
                ["make", "clean", "MAKE=make"],
                check=True,
                env=sub_env,
            )

            # Build with double precision
            subprocess.run(
                [
                    "make",
                    "DBL_PREC=true",
                    "FC=gfortran",
                    "CC=gcc",
                    "MAKE=make",
                ],
                check=True,
                env=sub_env,
            )

            # The makefile links with `-o ../bin/prms`.  On Windows the
            # gfortran driver appends ".exe" when the -o name carries no
            # suffix, so the file to look for is bin/prms.exe there.
            suffix = ".exe" if _get_platform_tag() == "win" else ""
            compiled_bin = src_dir / "bin" / f"prms{suffix}"
            if not compiled_bin.exists():
                raise RuntimeError(
                    "Compilation appeared to succeed but "
                    f"bin/{compiled_bin.name} not found in {src_dir}"
                )

            _get_bin_dir().mkdir(parents=True, exist_ok=True)
            shutil.copy(compiled_bin, binary_path)
            print(f"Successfully compiled and installed to {binary_path}")

        finally:
            os.chdir(orig_dir)

    return binary_path


def get_or_compile_prms_exe(
    exe_desc: str = "prms",
    force: bool = False,
    compile_source: Optional[str] = None,
) -> pl.Path:
    """Return the PRMS/GSFLOW executable, compiling from source if needed.

    PRMS binaries are compiled automatically when they are not present (or
    when *force* is ``True``).  GSFLOW has no source in this repository, so
    its binary must already exist in ``bin/``.

    Parameters
    ----------
    exe_desc : str
        See :func:`get_prms_exe_name`.
    force : bool
        Passed through to :func:`compile_prms`.  Forces recompilation even
        when the binary already exists.
    compile_source : str or None
        Override the source version to compile.  When ``None`` the source
        version is inferred from *exe_desc*.

    Returns
    -------
    pathlib.Path
        Absolute path to the executable.

    Raises
    ------
    FileNotFoundError
        If the binary does not exist and cannot be compiled.
    """
    exe_path = get_prms_exe_path(exe_desc)

    if exe_path.exists() and not force:
        return exe_path

    # Determine which source to compile
    source = compile_source
    if source is None:
        source = _get_source_for_exe_desc(exe_desc)

    if source is not None:
        return compile_prms(source=source, force=force)

    raise FileNotFoundError(
        f"Executable not found and no compilable source is available for "
        f"exe_desc='{exe_desc}': {exe_path}\n"
        "GSFLOW source is not part of this repository; its binaries are "
        "checked in to bin/."
    )
