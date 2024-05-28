import argparse
import os
import subprocess
import time

files = [
    "dataset_stats.pkl",
    "train_val_kl_seed_42.png",
    "train_val_l1_seed_42.png",
    "train_val_loss_seed_42.png",
    "policy_best.ckpt",
    "policy_last.ckpt",
    "policy_epoch_1900_seed_42.ckpt",
    "policy_epoch_2900_seed_42.ckpt",
    "policy_epoch_3900_seed_42.ckpt",
    "policy_epoch_4900_seed_42.ckpt",
    "policy_epoch_5900_seed_42.ckpt",
    "policy_epoch_6900_seed_42.ckpt",
    "policy_epoch_7900_seed_42.ckpt"
]

def download_file(file: str) -> bool:
    try:
        source_path = os.path.join(options.src, file)
        url = f"{options.server}:{source_path}"
        command = f"scp {url} {options.dest}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Downloaded: {file}")
            return False
        else:
            print(f"Failed to download {url}")
            return True
    except Exception as e:
        print(f"Caught exception trying to download with scp: {e}")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser("fetch_results")
    parser.add_argument("--server", metavar="dstack_server", action="store", type=str, required=True, help="Dstack server name (e.g., funny-tiger-1)")
    parser.add_argument("--src", metavar="path", action="store", type=str, default="/workflow/checkpoints", help="Path on server (e.g., /workflow/checkpoints)")
    parser.add_argument("--dest", metavar="path", action="store", type=str, required=True, help="Local directory to write to")
    options = parser.parse_args()

    files_remaining = files.copy()

    while len(files_remaining) > 0:
        print(f"Downloading {len(files_remaining)} files...")
        for file in files_remaining.copy():
            error = download_file(file=file)
            if not error:
                files_remaining.remove(file)
        print("\n")
        time.sleep(60)

