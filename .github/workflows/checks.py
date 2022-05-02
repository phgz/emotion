#!/usr/bin/env python
import os
import re
import subprocess
import sys
from pathlib import Path

import git
import toml

repo = git.Repo(".", search_parent_directories=True)
project_root = repo.working_tree_dir
index = [item for item in repo.index.diff("HEAD")]

# ------------------------------------------------------------------------------#
#                            Check for breakpoints                              #
# ------------------------------------------------------------------------------#

for item in index:
    code = repo.git.show(item.a_blob)
    match = re.search(r"breakpoint\(\)", code, re.S)

    if match:
        print("Found the presence of breakpoints in the code.")
        sys.exit(1)

# ------------------------------------------------------------------------------#
#                              Check for version                                #
# ------------------------------------------------------------------------------#

metadata_files = ["setup.py", "pyproject.toml"]
path = lambda file: os.path.join(project_root, file)
current_metadata_file = next(
    filter(lambda p: os.path.exists(p), map(path, metadata_files))
)
metadata_name = current_metadata_file.split("/")[-1]

origin_default = next(ref for ref in repo.refs if ref.name == "origin/main")
default_branch = origin_default.name.split("/")[-1]
default_branch_metadata_file = default_branch.commit.tree[metadata_name]

with open(current_metadata_file, "r") as f:
    current_content = f.read()

default_content = repo.git.show(default_branch_metadata_file)

not_staged = [item.a_path for item in repo.index.diff(None)]

find_version = dict(
    zip(
        metadata_files,
        [
            lambda s: re.search(r"version\s*=\s*[\'|\"](.*)[\"|\']", s).group(1),
            lambda s: toml.loads(s)["tool"]["poetry"]["version"],
        ],
    )
)

versions = [
    find_version[metadata_name](content)
    for content in [current_content, default_content]
]

parsed_current, parsed_remote_master = [list(map(int, v.split("."))) for v in versions]

if parsed_current <= parsed_remote_master or metadata_name in not_staged:
    print(repo.active_branch.name + "'s version is not greater than remote master's.")
    sys.exit(1)

# ------------------------------------------------------------------------------#
#      Check for python unused imports/variables and expand star imports       #
# ------------------------------------------------------------------------------#

autoflake_opts = [
    "--in-place",
    "--recursive",
    "--expand-star-imports",
    "--remove-all-unused-imports",
    "--ignore-init-module-imports",
    "--remove-unused-variables",
    "--verbose",
]

to_process = [
    item.a_path
    for item in index
    if item.a_path.endswith(".py") and Path(item.a_path).exists()
]

if to_process:
    out = subprocess.run(
        ["autoflake"] + autoflake_opts + to_process,
        check=True,
        cwd=project_root,
        capture_output=True,
    )

    if out.stderr:
        mod_files = [file.split()[1] for file in out.stderr.decode().split("\n")[:-1]]

        print(
            "Autoflakes found unused imports or variables in the code. Check these files:"
        )
        print("\n".join(mod_files) + "\n")

        subprocess.run(
            ["git", "diff"],
            check=True,
            cwd=project_root,
        )

        sys.exit(1)
