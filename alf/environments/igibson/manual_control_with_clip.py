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
import igibson
import pybullet as p
from alf.environments.igibson.igibson_env import iGibsonCustomEnv

import clip
import torch
from PIL import Image
from matplotlib import pyplot as plt
import gym
from scipy.stats import entropy
from alf.utils.video_recorder import VideoRecorder

# clip params
scene_id = "Rs_int"
use_colors = False
target_name = 'grey bed' if use_colors else 'bed'
record_file = f'/home/junyaoshi/Desktop/CLIP_videos/{scene_id}_{target_name.replace(" ", "_")}' \
              f'{"_with_colors_" if use_colors else "_"}0718_3.mp4'
fps = 10
colors = [
    "blue", "yellow", "green", "white", "brown", "grey", "purple", "red",
    "orange", "pink", "black"
]

# env params
yaml_filename = 'turtlebot_clip.yaml'
mode = 'gui'
action_timestep = 1.0 / 10.0
physics_timestep = 1.0 / 120.0
device_idx = 1
fov = 75

# keys to action
robot_keys_to_actions = {
    'w': [0.5, 0.5],
    's': [-0.5, -0.5],
    'a': [0.2, -0.2],
    'd': [-0.2, 0.2],
    'f': [0, 0]
}


class ClipRenderWrapper(gym.Wrapper):
    """This wrapper makes the env render CLIP statistics real time

    The render() function renders the current image frame on the left, and classes with
    best CLIP values, along with their associated values, on the right.
    The plot is then converted into a numpy array for video logging.
    """

    def __init__(self, env, classes, target_name):
        super().__init__(env)
        self.env = env
        self.metadata['render.modes'] = ['rgb_array', 'human']
        self.classes = classes
        self.target_name = target_name
        self.best_values = None
        self.best_indices = None
        self.frame_array = None
        self.target_value = None

    def render(self, mode='rgb_array', fontsize=40, **kwargs):
        assert self.best_values is not None and self.best_indices is not None and self.frame_array is not None
        assert self.target_value is not None and self.target_name is not None
        entropy_val = entropy(self.best_values.cpu().numpy()) / np.log(
            len(self.best_values.cpu().numpy()))
        fig = plt.figure(figsize=(50, 30))

        # plot current image frame and entropy
        plt.subplot(1, 2, 1)
        plt.imshow(self.frame_array)
        plt.axis("off")
        plt.title(
            f'Entropy: {entropy_val:.4f} | {target_name}: {100. * self.target_value.item():.4f}%',
            fontsize=fontsize)

        # plot top classes and values determined by CLIP
        y = np.arange(self.best_values.shape[-1])
        plt.subplot(1, 2, 2)
        plt.grid()
        plt.barh(y, self.best_values.cpu().numpy())
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(
            y, [
                self.classes[index] if len(self.classes[index]) < 25 else
                self.classes[index][:20] + '\n' + self.classes[index][20:]
                for index in self.best_indices
            ],
            fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.xlim([0., 1.])
        plt.xlabel("probability", fontsize=fontsize)

        plt.tight_layout(rect=[0.01, 0.01, 0.01, 0.01])
        plt.subplots_adjust(wspace=0.5)

        # convert plot into numpy array
        fig.canvas.draw()
        output = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        output = output.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        return output

    def set_clip_infos(self, best_values, best_indices, frame_array,
                       target_value):
        """Set CLIP related infos"""
        self.best_values = best_values
        self.best_indices = best_indices
        self.frame_array = frame_array
        self.target_value = target_value


def main():
    # load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # load iGibson env
    config_filename = os.path.join(igibson.vlnav_config_path, yaml_filename)
    env = iGibsonCustomEnv(
        config_file=config_filename,
        mode=mode,
        action_timestep=action_timestep,
        physics_timestep=physics_timestep,
        device_idx=device_idx)

    # load and process classes
    classes = list(env.scene.objects_by_category.keys())
    if use_colors:
        classes = [f'{color} {label}' for color in colors for label in classes]
    target_index = classes.index(target_name)
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    env = ClipRenderWrapper(env, classes, target_name=target_name)

    env.reset()
    robot = env.config.get('robot')
    if robot == 'Turtlebot' or robot == 'Locobot':
        keys_to_actions = robot_keys_to_actions
    else:
        raise ValueError(f'Unknown robot: {robot}')

    recorder = VideoRecorder(
        env, append_blank_frames=0, frames_per_sec=fps, path=record_file)

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

    def process_pressed_keys(pressed_keys, running):
        if ord('r') in pressed_keys:
            print('Reset the environment...')
            env.reset()
            pressed_keys = []
        if ord('p') in pressed_keys:
            print('Shutting down the environment...')
            env.close()
            running = False
            pressed_keys = []
        if ord('8') in pressed_keys:
            print('Forward...')
            env.step(keys_to_actions['w'])
        if ord('5') in pressed_keys:
            print('Backward...')
            env.step(keys_to_actions['s'])
        if ord('6') in pressed_keys:
            print('Left...')
            env.step(keys_to_actions['a'])
        if ord('4') in pressed_keys:
            print('Right...')
            env.step(keys_to_actions['d'])
        if ord('2') in pressed_keys:
            print('Staying still...')
            env.step(keys_to_actions['f'])
        return running

    running = True
    while running:

        # get and process image input
        frame = get_robot_cam_frame()
        assert frame is not None
        frame = (frame[:, :, :3] * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame, 'RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_value, best_index = similarity[0].topk(1)
        best_values, best_indices = similarity[0].topk(5)
        target_value = similarity[:, target_index]

        print(f'{target_name} (Target): {100. * target_value.item():.4f} | '
              f'{classes[best_index]} (Best): {100. * best_value.item():.4f}')

        env.set_clip_infos(
            best_values=best_values,
            best_indices=best_indices,
            frame_array=frame,
            target_value=target_value)
        recorder.capture_frame()

        # detect and process pressed keys
        pressed_keys = get_key_pressed()
        running = process_pressed_keys(pressed_keys, running)


if __name__ == "__main__":
    main()
