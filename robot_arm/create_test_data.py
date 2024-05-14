#
# create_test_data.py
#
# Creates an h5py file with mock data to test ACT model with modified input dimensions.
#
# How record_sim_episodes.py works:
#
#   - End-effector simulation is generated.
#   - For each step in episode:
#       - Produce end effector "action", which consists of 8 numbers for each arm:
#           x, y, z:        End effector position
#           qx, qy, qz, qw: End effector rotation (quaternion)
#           g:              Gripper position (0, open, to 1, closed)
#         This is produced from a fixed trajectory. 
#       - This action is appled to a simulator. Presumably, the results of this are the joint
#         positions at the start of the *next* timestep. For each arm:
#           qpos:           6 absolute joint positions plus 1 gripper position
#           qvel:           6 absolute joint velocities plus 1 gripper velocity (-: closing, +: opening)
#           images:         RGB images
#   - The result is an "episode" array of length episode_len+1. Why +1? Because each time step
#     produces an action to generate the joint positions at the next time step. So the action at
#     index (episode_len-1) produces one more output at index episode_len.
#
#   - Produces joint_traj, which is just qpos for each element in "episode", the actual joint
#     positions. Also replaces the gripper pose in this array with the gripper control value.
#     Remember: joint_traj are the motor outputs from the end effector simulation, the ground truth
#     actions in motor (joint) space, apparently. I am not sure, but I think that the end effector
#     simulation might not apply physics the same way as the subsequent simulation.
#
#   - Joint position simulation is generated.
#   - For each element in joint_traj:
#       - Input "action" is the joint_traj element, the motor positions from the initial simulation.
#         Presumably, these represent "leader" arm values.
#       - Output are positions, velocities, and images as before. Presumably these represent
#         "follower" arms.
#   - The result is an "episode_replay" array, which is episode_len+2 long. Because episode_len+1
#     joint positions were simulated, an extra output at the end was produced.
#
#   - joint_traj and episode_replay are truncated by removing the last element of each, bringing
#     them to:
#       - joint_traj -> episode_len
#       - episode_replay -> episode_len+1
#
#   - joint_traj becomes the /actions list in the final h5py file.
#   - episode_replay becomes /observations/qpos, /observations/qvel, and /observations/images/{cam_name}.
#     Note that only the first episode_len elements are used (not the very last one).
#
#   - It therefore appears the format should be:
#       - /observations -- observed follower position at time step
#       - /actions -- leader arm position input at time step (which produces observation at next
#         time step)
#
#   - Prediction: given FOLLOWER OBSERVATIONS -> PREDICT LEADER ACTIONS. Leader arm may end up in
#     positions follower cannot (e.g., operator gripping more tightly), and will ultimately be
#     reflected to follower anyway.
# 


#TODO next: capture some actual data! then compare losses and try to perform forward simulation

import argparse

import h5py

NUM_MOTORS = 5          # 4 joints plus gripper
NUM_TIME_STEPS = 100    # arbitrary number of time steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser("create_test_data")
    parser.add_argument("file", nargs=1)
    options = parser.parse_args()

    output_filename = options.file[0]

    with h5py.File(name=output_filename, mode="w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False   # TODO: is this needed?

        # Follower data, including RGB images from camera, goes under /observations
        follower = root.create_group("observations")
        qpos = follower.create_dataset(name="qpos", shape=(NUM_TIME_STEPS, NUM_MOTORS))
        qvel = follower.create_dataset(name="qvel", shape=(NUM_TIME_STEPS, NUM_MOTORS))
        camera_images = follower.create_group("images")
        camera_images.create_dataset(name="top", shape=(NUM_TIME_STEPS, 480, 640, 3), dtype="uint8", chunks=(1, 480, 640, 3))
        
        # Leader data, the motor commands at each time step given the 
        action = root.create_dataset(name="action", shape=(NUM_TIME_STEPS, NUM_MOTORS))


