from dotenv import load_dotenv
load_dotenv()

import os

from langchain_google_community import GoogleDriveLoader

loader = GoogleDriveLoader(
    folder_id=os.environ["GOOGLE_DRIVE_FOLDER_ID"],
    recursive=False  # Optional: Fetch files from subfolders recursively. Defaults to False.
)

docs = loader.load()
