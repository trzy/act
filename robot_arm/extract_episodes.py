#
# extract_episodes.py
# Bart Trzynadlowski
#
# Given recorded episodes produced by the robot-arm server, this script copies out the hdf5 files
# into a sequentially numbered series of episode files (episode_1.hdf5, episode_2.hdf5, ...) as
# expected by imitate_episodes.py.
#

import argparse
import os

import h5py
import numpy as np

def pad(dataset, new_length):
  length = dataset.shape[0]
  last_element = dataset[-1]
  pad_values = np.repeat(last_element[np.newaxis, :], new_length - length, axis=0)
  dataset = dataset[:]
  dataset.resize((new_length, *dataset.shape[1:]))
  dataset[length:] = pad_values
  return dataset

if __name__ == "__main__":
  parser = argparse.ArgumentParser("extract_episodes")
  parser.add_argument("--dir", metavar="path", action="store", required=True, type=str, help="Directory of robot-arm data recording")
  options = parser.parse_args()

  print("Analyzing episodes...")
  dirs = [ os.path.join(options.dir, dir) for dir in os.listdir(options.dir) if dir.startswith("example-") and os.path.isdir(os.path.join(options.dir, dir)) ]
  files = []
  max_length = 0
  for i in range(len(dirs)):
    dir = dirs[i]
    file = os.path.join(dir, "data.h5")
    files.append(file)
    with h5py.File(name=file, mode="r") as fp:
      max_length = max(max_length, len(fp["/action"]))
  print(f"  - Maximum episode length: {max_length}")

  print(f"Extracting and padding episodes to {max_length} observations...")
  for i in range(len(files)):
    file = files[i]
    dest_file = os.path.join(options.dir, f"episode_{i}.hdf5")
    with h5py.File(name=file, mode="r") as fp:
      current_length = fp["/action"].shape[0]
      actions = pad(dataset=fp["/action"], new_length=max_length)
      qpos = pad(dataset=fp["/observations/qpos"], new_length=max_length)
      qvel = pad(dataset=fp["/observations/qvel"], new_length=max_length)
      images = pad(dataset=fp["/observations/images/top"], new_length=max_length)
      with h5py.File(name=dest_file, mode="w", rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False   # TODO: is this needed?
        follower = root.create_group("observations")
        qpos = follower.create_dataset(name="qpos", shape=qpos.shape)
        qvel = follower.create_dataset(name="qvel", shape=qvel.shape)
        camera_images = follower.create_group("images")
        camera_images.create_dataset(name="top", shape=images.shape, dtype="uint8", chunks=(1, *images.shape[1:]))
        action = root.create_dataset(name="action", shape=actions.shape)
        root["/observations/qpos"][...] = qpos
        root["/observations/qvel"][...] = qvel
        root["/observations/images/top"][...] = images
        root["/action"][...] = actions
        print(root["/action"].shape)
        print(f"  - Copied: {file} -> {dest_file} and extended from {current_length} -> {max_length} observations")