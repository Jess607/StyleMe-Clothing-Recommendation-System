from googleapiclient.http import MediaIoBaseDownload
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import numpy as np
from PIL import Image
from io import BytesIO



def get_images_from_folder(folder_id):
    # Replace 'folder_id' with the ID of the folder you want to access
    # Set the path to your Google Drive API credentials JSON file
    credentials_path = 'clothing-app-397708-5788022966e2.json'

# Initialize the Google Drive API
    creds = None
    if os.path.exists(credentials_path):
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )

    drive_service = build('drive', 'v3', credentials=creds)
    list_images=[]
    list_file_id=[]

    # List the files and subfolders in the folder
    results = drive_service.files().list(q=f"'{folder_id}' in parents and trashed=false",fields="files(name, id, mimeType)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found in the folder.')
    else:
        print('Files and subfolders in the folder:')
        for item in items:
            print(f"{item['name']} {item['id']}")
            list_file_id.append(item['id'])

    for i in range(len(items)):
        request = drive_service.files().get_media(fileId=items[i]['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        try:
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")

            print("Download complete.")
        except Exception as e:
            print(f"An error occurred during the download: {str(e)}")


        downloaded_content = fh.getvalue()
        img = Image.open(BytesIO(downloaded_content))
        list_images.append(img)
    return list_images, list_file_id   


def get_images_from_drive(list_file_id):
    credentials_path = 'clothing-app-397708-5788022966e2.json'
    list_images=[]

# Initialize the Google Drive API
    creds = None
    if os.path.exists(credentials_path):
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )

    drive_service = build('drive', 'v3', credentials=creds)

    for i in list_file_id:
        request = drive_service.files().get_media(fileId=i)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        try:
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")

            print("Download complete.")
        except Exception as e:
            print(f"An error occurred during the download: {str(e)}")
        

        downloaded_content = fh.getvalue()
        img = Image.open(BytesIO(downloaded_content))
        list_images.append(img)
    return list_images, list_file_id


