#!/usr/bin/env python3
"""Print the site-packages directory of a non-editable install of a package.

Run on the remote host's venv interpreter. Prints nothing if the package
isn't installed, or is installed editable (editable installs already point
at the synced source tree, so no overlay is needed).
"""

import importlib.metadata as m
import json
import sys


def main() -> None:
    repo_name = sys.argv[1]

    dist = None
    for candidate in (repo_name, repo_name.replace("-", "_")):
        try:
            dist = m.distribution(candidate)
            break
        except m.PackageNotFoundError:
            continue
    if dist is None:
        return

    editable = False
    try:
        raw = dist.read_text("direct_url.json")
        if raw:
            editable = bool(json.loads(raw).get("dir_info", {}).get("editable", False))
    except Exception:
        pass
    if editable:
        return

    top_level = dist.read_text("top_level.txt")
    pkg = top_level.strip().splitlines()[0] if top_level else dist.metadata["Name"].replace("-", "_")
    print(str(dist.locate_file(pkg)))


if __name__ == "__main__":
    main()
