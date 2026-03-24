import os
import kagglehub
import shutil

os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\Dex\.kaggle'

os.environ['KAGGLEHUB_CACHE'] = r'D:\TRY\MLPROJECT\temp_cache'

print("Downloading directly to D: drive... please wait.")

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

if not os.path.exists("data"):
    source = os.path.join(path, "chest_xray")
    if os.path.exists(source):
        shutil.move(source, "data")
        print("Success! 'data' folder is ready on D: drive.")
    else:
        shutil.move(path, "data")
        print("Data moved to D: drive.")
