import asyncio
from dataclasses import dataclass

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

@dataclass
class Observation:
    qpos: np.ndarray
    image: np.ndarray

async def main(args, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
    try:
        set_seed(1)
        # command line parameters
        ckpt_dir = args['ckpt_dir']
        policy_class = args['policy_class']
        onscreen_render = False
        task_name = args['task_name']
        batch_size_train = args['batch_size']
        batch_size_val = args['batch_size']

        # get task parameters
        is_sim = False
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]

        #dataset_dir = task_config['dataset_dir']
        #num_episodes = task_config['num_episodes']
        episode_len = task_config['episode_len']
        camera_names = task_config['camera_names']

        # fixed parameters
        state_dim = 5 #14
        backbone = 'resnet18'
        if policy_class == 'ACT':
            enc_layers = 4
            dec_layers = 7
            nheads = 8
            policy_config = {
                'num_queries': args['chunk_size'],
                'kl_weight': args['kl_weight'],
                'hidden_dim': args['hidden_dim'],
                'dim_feedforward': args['dim_feedforward'],
                'backbone': backbone,
                'enc_layers': enc_layers,
                'dec_layers': dec_layers,
                'nheads': nheads,
                'camera_names': camera_names,
            }
        else:
            raise NotImplementedError

        config = {
            'ckpt_dir': ckpt_dir,
            'episode_len': episode_len,
            'state_dim': state_dim,
            'policy_class': policy_class,
            'onscreen_render': onscreen_render,
            'policy_config': policy_config,
            'task_name': task_name,
            'seed': args['seed'],
            'temporal_agg': args['temporal_agg'],
            'camera_names': camera_names,
            'real_robot': not is_sim
        }

        await infer_loop(config, "policy_best.ckpt", input_queue=input_queue, output_queue=output_queue)
    except Exception as e:
        import traceback
        print(f"Exception caught: {e}")
        traceback.print_exc()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def _prepare_image(frame: np.ndarray) -> torch.Tensor:
    frame = frame.transpose([2, 0, 1])  # (height,width,channels) -> (channels,height,width)
    frame = np.stack([ frame ], axis=0)
    return torch.from_numpy(frame / 255.0).float().cuda().unsqueeze(0)


async def infer_loop(config, ckpt_name, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

    with torch.inference_mode():
        t = 0
        while True:
            # Get observation
            observation = await input_queue.get()
            if not isinstance(observation, Observation):
                # Reset state
                t = 0
                if temporal_agg:
                    all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
                continue
            qpos_numpy = observation.qpos
            curr_image = _prepare_image(frame=observation.image)

            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            ### query policy
            if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    print(f"t={t} infer (query_frequency={query_frequency})")
                    all_actions = policy(qpos, curr_image)
                    # print("Inference:")
                    # for action in all_actions:
                    #     print(post_process(action.cpu()))
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
                    print(f"t={t} sample {t%query_frequency}")
            elif config['policy_class'] == "CNNMLP":
                raw_action = policy(qpos, curr_image)
            else:
                raise NotImplementedError

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            target_qpos = action

            # Send
            await output_queue.put(target_qpos)

            # Next
            t += 1


####################################################################################################
# Server
####################################################################################################

from annotated_types import Len
import base64
from typing import Annotated, Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from robot_arm.networking import handler, MessageHandler, Session, TCPServer

# HelloMessage is also used to reset inference process
class HelloMessage(BaseModel):
    message: str

class InferenceRequestMessage(BaseModel):
    motor_radians: Annotated[List[float], Len(min_length=5, max_length=5)]
    frame: bytes

class InferenceResponseMessage(BaseModel):
    target_motor_radians: Annotated[List[float], Len(min_length=5, max_length=5)]
      
class InferenceServer(MessageHandler):
    def __init__(self, port: int, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        super().__init__()
        self.sessions = set()
        self._server = TCPServer(port=port, message_handler=self)
        self._input_queue = input_queue
        self._output_queue = output_queue

    async def run(self):
        await asyncio.gather(self._server.run(), self._send_results())
        
    async def _send_results(self):
        while True:
            target_motor_radians = await self._output_queue.get()
            msg = InferenceResponseMessage(target_motor_radians=target_motor_radians)
            for session in self.sessions:
                await session.send(msg)

    async def on_connect(self, session: Session):
        print("Connection from: %s" % session.remote_endpoint)
        await session.send(HelloMessage(message = "Hello from ACT inference server"))
        self.sessions.add(session)
    
    async def on_disconnect(self, session: Session):
        print("Disconnected from: %s" % session.remote_endpoint)
        self.sessions.remove(session)
    
    @handler(HelloMessage)
    async def handle_HelloMessage(self, session: Session, msg: HelloMessage, timestamp: float):
        print("Hello received: %s" % msg.message)
        await self._input_queue.put(msg)
    
    @handler(InferenceRequestMessage)
    async def handle_InferenceRequestMessage(self, session: Session, msg: InferenceRequestMessage, timestamp: float):
        frame = np.frombuffer(buffer=base64.b64decode(msg.frame), dtype=np.uint8).reshape((480, 640, 3))
        motor_radians = np.array(msg.motor_radians)
        await self._input_queue.put(Observation(qpos=motor_radians, image=frame))


####################################################################################################
# Program Entry Point
####################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, default='ACT', help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, default='sim_bottlecap_desk', help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=100, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # Unused but some modules depend on it
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)

    options = parser.parse_args()

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    tasks = []
    server = InferenceServer(port=8000, input_queue=input_queue, output_queue=output_queue)
    loop = asyncio.new_event_loop()
    tasks.append(loop.create_task(server.run()))
    tasks.append(loop.create_task(main(vars(options), input_queue=input_queue, output_queue=output_queue)))
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except asyncio.exceptions.CancelledError:
        print("\nExited normally")
    except:
        print("\nExited due to uncaught exception")