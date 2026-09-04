# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> three_dimensions -> depth -> dvlt -> dvlt_runtime -> src -> dvlt -> util -> download.py functionality."""

import os

import requests


def _offline_mode_enabled():
    return os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {
        "1",
        "on",
        "true",
        "yes",
    }


def download_file_from_url(url, filename):
    """
    Downloads a file from a URL, handling redirects.

    Args:
        url: URL of the file to download
        filename: Local path to save the downloaded file
    """
    if _offline_mode_enabled():
        raise RuntimeError(
            f"Cannot download {url}: offline mode is enabled by HF_HUB_OFFLINE."
        )

    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
