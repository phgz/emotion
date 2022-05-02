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

setup_files = ["setup.py", "pyproject.toml"]
path = lambda file: os.path.join(project_root, file)
current_setup_file = next(filter(lambda p: os.path.exists(p), map(path, setup_files)))
setup_name = current_setup_file.split("/")[-1]

HEAD = next(ref for ref in repo.remotes.origin.refs if ref.name == "origin/HEAD")
default_branch = HEAD.ref.name.split("/")[-1]

remote_master_setup_file = repo.remotes.origin.fetch(default_branch)[0].commit.tree[
    setup_name
]

with open(current_setup_file, "r") as f:
    current_content = f.read()

remote_master_content = repo.git.show(remote_master_setup_file)

not_staged = [item.a_path for item in repo.index.diff(None)]

find_version = dict(
    zip(
        setup_files,
        [
            lambda s: re.search(r"version\s*=\s*[\'|\"](.*)[\"|\']", s).group(1),
            lambda s: toml.loads(s)["tool"]["poetry"]["version"],
        ],
    )
)

versions = [
    find_version[setup_name](content)
    for content in [current_content, remote_master_content]
]

parsed_current, parsed_remote_master = [list(map(int, v.split("."))) for v in versions]

if parsed_current <= parsed_remote_master or setup_name in not_staged:
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
