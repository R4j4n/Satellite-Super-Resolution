import os
import gdown
import zipfile
from pathlib import Path


# get the root path of the project
root_pth = Path(__file__).parent.parent

# create a data directory
Path(root_pth).joinpath("data").mkdir(exist_ok=True, parents=True)

file_id = "1URWgsLOI8zAXJ31E7H3Jo-lMnOf5mhA6"

# Create the download URL
download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Download the file
output = "train.zip"
final_pth = str(Path(root_pth).joinpath(output))
gdown.download(download_url, str(Path(root_pth).joinpath(output)), quiet=False)

# Extract the downloaded zip file
with zipfile.ZipFile(final_pth, "r") as zip_ref:
    zip_ref.extractall(str(Path(root_pth).joinpath("data")))


# delete the zip file
os.remove(str(final_pth))

print("Download and extraction complete.")
