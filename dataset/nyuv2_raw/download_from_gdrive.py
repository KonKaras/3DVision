# Source: https://stackoverflow.com/a/39225039

import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        content_size = int(response.headers.get("content-length"))
        print(f"Content size {content_size}")
        with open(destination, "wb") as f, tqdm(
            total=content_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=CHUNK_SIZE,
            desc=destination,
        ) as bar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    size = f.write(chunk)
                    bar.update(size)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    print("Saving response content")
    save_response_content(response, destination)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2]
        download_file_from_google_drive(file_id, destination)
