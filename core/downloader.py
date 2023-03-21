"""Zindi data downloader
https://zindi.africa/learn/how-to-download-data-files-from-zindi-to-colab
"""

import requests
from tqdm.auto import tqdm


def zindi_data_downloader(url: str, token:str, file_name):
    """Function to download project data from zindi website."""
    # Get the competition data
    competition_data = requests.post(url=data_url, data=token, stream=True)

    # Progress bar monitor download
    pbar = tqdm(desc=file_name,
                total=int(competition_data.headers.get('content-length', 0)),
                unit='B',
                unit_scale=True,
                unit_divisor=512)

    handle = open(file_name, "wb")
    for chunk in competition_data.iter_content(chunk_size=512):  # Download the data in chunks
        if chunk:  # filter out keep-alive new chunks
            handle.write(chunk)
        pbar.update(len(chunk))
    handle.close()
    pbar.close()
