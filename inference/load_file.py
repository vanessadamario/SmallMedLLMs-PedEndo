import os
from os.path import join
import json

def _load_file(bucket_name, blob_name, create=True, local=False):
    if local:
        file_in_bucket = os.listdir(bucket_name) # verify if file exists at location
        if blob_name not in file_in_bucket: # if it does not exist
            if create:
                with open(join(bucket_name, blob_name), "w") as outfile:
                    file_content = {}
                    json.dump(file_content, outfile)
            else:
                raise ValueError(f"File {blob_name} does not exist in folder {bucket_name}.")
        else: # it exists, we load it
            with open(join(bucket_name, blob_name), "r") as f:
                file_content = json.load(f)
        return None, file_content

    else:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blobs = bucket.list_blobs()
        blob_names = [blob.name for blob in blobs]
        if blob_name not in blob_names:
            if create:
                blob = bucket.blob(blob_name)
                with blob.open("w") as outfile:
                    file_content = {}
                    json.dump(file_content, outfile)
            else:
                raise ValueError(f"Blob {blob_name} does not exist in bucket {bucket_name}.")
        else:
            blob = bucket.blob(blob_name)
            with blob.open("r") as f:
                file_content = json.load(f)

        return bucket, file_content


def load_esap(local=False):
    if local: 
        folder_name = '/home/vdamario/src/LlamaT1D/inference'
    else:
        folder_name = 'data_guidelines'

    _, file_content = _load_file(bucket_name=folder_name, blob_name="Ped-ESAP.json", create=False, local=local)
    return file_content