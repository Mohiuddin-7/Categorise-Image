from huggingface_hub import snapshot_download

# Dataset repository name (from the URL)
repo_id = "YashJain/UI-Elements-Detection-Dataset"

# Download entire dataset
snapshot_download(repo_id, repo_type="dataset", local_dir="UI_Dataset")
