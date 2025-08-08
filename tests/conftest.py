#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 11:48:24 2025

@author: dlvilla
"""

import pytest
import re
import os
import logging

# conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--keep-files",
        action="store_true",
        default=False,
        help="Do not delete generated files after tests",
    )
    
@pytest.fixture(scope="session")
def keep_files(request):
    return request.config.getoption("--keep-files")


#Written by AI.
def delete_matching_files_in_dir(dir_path: str, regex_patterns: list[str]):
    """
    Delete files in the given directory that match any regex pattern.
    Does not recurse into subdirectories.

    Parameters:
    - dir_path: Path to the directory to clean
    - regex_patterns: List of regex patterns as strings

    Returns:
    - List of deleted file paths
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a valid directory")

    compiled_patterns = [re.compile(p) for p in regex_patterns]
    deleted_files = []

    for entry in os.scandir(dir_path):
        if entry.is_file():
            for pattern in compiled_patterns:
                if pattern.fullmatch(entry.name):
                    try:
                        os.remove(entry.path)
                        deleted_files.append(entry.path)
                        print(f"Deleted: {entry.path}")
                    except Exception as e:
                        print(f"Error deleting {entry.path}: {e}")
                    break  # Stop after first pattern match

    return deleted_files



def pytest_sessionfinish(session, exitstatus):
    print(">>> [HOOK] pytest_sessionfinish called")
    print(f">>> Exit status: {exitstatus}")
    if not keep_files:
        print("All output files are deleted by default use 'pytest --keep-files to not delete them.")
        test_dir = os.path.dirname(__file__)
        patterns = [".*\.png",".*\.csv",".*\.log",".*\.txt"]
        delete_matching_files_in_dir(test_dir,patterns)
        # Add final cleanup or summary reporting here
