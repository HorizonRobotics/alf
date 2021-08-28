# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np
from time import time
import igibson
import pybullet as p

from alf.environments.igibson.igibson_env import iGibsonCustomEnv

# env params
yaml_filename = 'turtlebot_clip.yaml'
mode = 'gui'
action_timestep = 1.0 / 10.0
physics_timestep = 1.0 / 120.0
device_idx = 1
fov = 75

# data saving
trial_name = '0630_1'
if not os.path.isdir('manual_control_images'):
    os.mkdir('manual_control_images')
data_dir = os.path.join('manual_control_images', trial_name)
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# keys to action
robot_keys_to_actions = {
    'w': [0.5, 0.5],
    's': [-0.5, -0.5],
    'a': [0.2, -0.2],
    'd': [-0.2, 0.2],
    'f': [0, 0]
}


def main():
    config_filename = os.path.join(igibson.vlnav_config_path, yaml_filename)
    env = iGibsonCustomEnv(
        config_file=config_filename,
        mode=mode,
        action_timestep=action_timestep,
        physics_timestep=physics_timestep,
        device_idx=device_idx)
    env.reset()
    robot = env.config.get('robot')
    if robot == 'Turtlebot' or robot == 'Locobot':
        keys_to_actions = robot_keys_to_actions
    else:
        raise ValueError(f'Unknown robot: {robot}')

    categories = list(env.scene.objects_by_category.keys())
    categories_fname = os.path.join(
        data_dir, f'{env.config.get("scene_id")}_categories.txt')
    with open(categories_fname, 'w') as output:
        for name in categories:
            output.write(name + '\n')

    def get_key_pressed():
        pressed_keys = []
        events = p.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            pressed_keys.append(key)
        return pressed_keys

    def get_robot_cam_frame():
        frames = env.simulator.renderer.render_robot_cameras(modes=('rgb'))
        if len(frames) > 0:
            frame = cv2.cvtColor(
                np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)
            return frame
        return None

    running = True
    while running:
        # detect pressed keys
        pressed_keys = get_key_pressed()
        if ord('r') in pressed_keys:
            print('Reset the environment...')
            env.reset()
            pressed_keys = []
        if ord('p') in pressed_keys:
            print('Shutting down the environment...')
            env.close()
            running = False
            pressed_keys = []
        if ord('c') in pressed_keys:
            print('Saving an image from RobotView...')
            img_path = os.path.join(data_dir, f'img_{int(time())}.png')
            frame = get_robot_cam_frame()
            assert frame is not None
            frame = (frame[:, :, :3] * 255).astype(np.uint8)
            cv2.imwrite(img_path, frame)
            pressed_keys = []
        if ord('8') in pressed_keys:
            print('Forward...')
            env.step(keys_to_actions['w'])
            # pressed_keys = []
        if ord('5') in pressed_keys:
            print('Backward...')
            env.step(keys_to_actions['s'])
            # pressed_keys = []
        if ord('6') in pressed_keys:
            print('Left...')
            env.step(keys_to_actions['a'])
            # pressed_keys = []
        if ord('4') in pressed_keys:
            print('Right...')
            env.step(keys_to_actions['d'])
            # pressed_keys = []
        if ord('2') in pressed_keys:
            print('Staying still...')
            env.step(keys_to_actions['f'])
            pressed_keys = []


if __name__ == "__main__":
    main()
