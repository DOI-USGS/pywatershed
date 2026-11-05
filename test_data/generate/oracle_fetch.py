#!/usr/bin/env python
"""Download the pinned pws-prms-oracle answer sets into test_data/.oracle/.

    [GITHUB_TOKEN=...] python test_data/generate/oracle_fetch.py \\
        [--domain hru_1 ...] [--repo OWNER/NAME | --source DIR]

autotest/prms_oracle.toml pins one inputs key per domain and, per
control file, one outputs key per platform. For each pinned asset of
this platform that is not already present at its pinned key, the asset
is downloaded from the oracle's GitHub release (tag = domain name) and
unpacked into test_data/.oracle/<domain>/, which is laid out like the
oracle's domains/ and is what oracle_populate.py reads when
PWS_ORACLE_DIR is unset. Unpacked files are made read-only (0o444), as
in the oracle.

The oracle repository is private (GitHub "internal"), so GITHUB_TOKEN
must hold a token that can read it; assets are then fetched through the
GitHub API. Without a token the public release URL is tried, which works
only if the repository is public. --source names a directory holding
the assets instead (for testing, or an offline copy).
"""

import argparse
import json
import os
import pathlib as pl
import platform
import shutil
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request

REPO_ROOT = pl.Path(__file__).parents[2]
PINS = REPO_ROOT / "autotest" / "prms_oracle.toml"
ORACLE_DIR = REPO_ROOT / "test_data" / ".oracle"
REPO = "DOI-USGS/pws-prms-oracle"
# what this script has unpacked into a domain: {"inputs": ikey,
# "outputs": {control: okey}}; removed with the domain when the inputs
# key changes
FETCHED = ".fetched.json"


def platform_tag() -> str:
    """The oracle's platform tag (prms_oracle/build.py platform_tag,
    replicated -- the oracle is not a dependency): linux-x86_64 /
    macos-arm64 / windows-x86_64."""
    system = {"linux": "linux", "darwin": "macos", "win32": "windows"}[
        sys.platform
    ]
    machine = platform.machine().lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "arm64"}.get(
        machine, machine
    )
    return f"{system}-{arch}"


def read_fetched(dom_dir: pl.Path) -> dict:
    path = dom_dir / FETCHED
    if path.exists():
        return json.loads(path.read_text())
    return {"inputs": None, "outputs": {}}


def write_fetched(dom_dir: pl.Path, fetched: dict) -> None:
    (dom_dir / FETCHED).write_text(json.dumps(fetched, indent=1) + "\n")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Source:
    """Where assets come from: a local directory, or a GitHub release of
    repo -- through the API with GITHUB_TOKEN, else the public URL."""

    def __init__(self, repo: str, directory: str | None):
        self.repo = repo
        self.directory = pl.Path(directory).expanduser() if directory else None
        # empty counts as unset: a fork PR's CI step maps a secret it
        # cannot see to ""
        self.token = os.environ.get("GITHUB_TOKEN") or None
        self._releases: dict[str, dict] = {}

    def _api(self, url: str, accept: str) -> urllib.request.Request:
        return urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def _release_assets(self, domain: str) -> dict[str, str]:
        """asset name -> API url, for the release tagged domain."""
        if domain not in self._releases:
            url = (
                f"https://api.github.com/repos/{self.repo}"
                f"/releases/tags/{domain}"
            )
            req = self._api(url, "application/vnd.github+json")
            with urllib.request.urlopen(req) as resp:
                release = json.load(resp)
            self._releases[domain] = {
                a["name"]: a["url"] for a in release["assets"]
            }
        return self._releases[domain]

    def get(self, domain: str, asset: str, dest: pl.Path) -> None:
        if self.directory is not None:
            shutil.copyfile(self.directory / asset, dest)
            return
        if self.token is None:
            url = (
                f"https://github.com/{self.repo}/releases/download"
                f"/{domain}/{asset}"
            )
            print(f"  download {url}")
            try:
                urllib.request.urlretrieve(url, dest)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise SystemExit(
                        f"{url}: not found. If {self.repo} is private, "
                        "set GITHUB_TOKEN to a token that can read it."
                    ) from e
                raise
            return
        assets = self._release_assets(domain)
        if asset not in assets:
            raise SystemExit(
                f"{asset}: not in release {domain} of {self.repo}"
            )
        print(f"  download {assets[asset]}")
        # the asset url answers 302 to a signed storage url, which
        # rejects a request that also carries Authorization; follow the
        # redirect by hand, without the token
        req = self._api(assets[asset], "application/octet-stream")
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(req) as resp:
                dest.write_bytes(resp.read())
                return
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                raise
            location = e.headers["Location"]
        urllib.request.urlretrieve(location, dest)


def unpack(archive: pl.Path, domain: str) -> None:
    """Unpack <domain>/... members into ORACLE_DIR, replacing whatever
    the archive's top-level entries (output dirs, CBH files) already
    hold, then make every unpacked file read-only."""
    dom_dir = ORACLE_DIR / domain
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        rels = [pl.Path(m.name).relative_to(domain) for m in members]
        for top in sorted({r.parts[0] for r in rels if r.parts}):
            existing = dom_dir / top
            if existing.is_dir():
                shutil.rmtree(existing)
            elif existing.exists():
                existing.unlink()
        tar.extractall(ORACLE_DIR, filter="data")
        for m in members:
            if m.isfile():
                (ORACLE_DIR / m.name).chmod(0o444)


def fetch_domain(domain: str, pins: dict, source: Source) -> None:
    plat = platform_tag()
    ikey = pins["inputs"][domain]
    dom_dir = ORACLE_DIR / domain
    fetched = read_fetched(dom_dir)
    if fetched["inputs"] != ikey and dom_dir.exists():
        print(f"{domain}: inputs key changed, removing {dom_dir}")
        shutil.rmtree(dom_dir)
        fetched = read_fetched(dom_dir)

    wanted = {"inputs": (f"inputs_{domain}_{ikey}.tar.gz", ikey)}
    for key, by_platform in pins["outputs"].items():
        dom, control = key.split("/")
        if dom != domain:
            continue
        if plat not in by_platform:
            raise SystemExit(f"{key}: no pinned key for platform {plat}")
        okey = by_platform[plat]
        wanted[control] = (
            f"outputs_{domain}_{control}_{plat}_{okey}.tar.gz",
            okey,
        )

    with tempfile.TemporaryDirectory() as tmp:
        for what, (asset, key) in wanted.items():
            current = (
                fetched["inputs"]
                if what == "inputs"
                else fetched["outputs"].get(what)
            )
            if current == key:
                print(f"{domain}:{what}: current ({key})")
                continue
            print(f"{domain}:{what}: fetching {asset}")
            archive = pl.Path(tmp) / asset
            source.get(domain, asset, archive)
            ORACLE_DIR.mkdir(parents=True, exist_ok=True)
            unpack(archive, domain)
            if what == "inputs":
                fetched["inputs"] = key
            else:
                fetched["outputs"][what] = key
            write_fetched(dom_dir, fetched)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--domain", action="append", help="default: every pinned domain"
    )
    parser.add_argument(
        "--repo", default=REPO, help="GitHub repository holding the releases"
    )
    parser.add_argument(
        "--source", help="a directory holding the assets, instead of --repo"
    )
    args = parser.parse_args(argv)
    pins = tomllib.loads(PINS.read_text())
    source = Source(args.repo, args.source)
    for domain in args.domain or sorted(pins["inputs"]):
        if domain not in pins["inputs"]:
            raise SystemExit(f"{domain}: not pinned in {PINS}")
        fetch_domain(domain, pins, source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
