# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl import flags
from absl import logging
import sys
import numpy as np
import pprint
from suite_robomaster import RobomasterEnv, QRCodeRewardFunction

flags.DEFINE_string('robot_ip', None, "IP address of the robot")
#"192.168.86.25"
FLAGS = flags.FLAGS


def play():
    import pygame
    import pygame.locals as K
    pygame.init()
    freq = 10
    env = RobomasterEnv(robot_ip=FLAGS.robot_ip, freq=freq, dead_band=0.1)
    #reward_function=QRCodeRewardFunction())
    print("observation_space: %s" % pprint.pformat(env.observation_space))
    print("action_space: %s" % pprint.pformat(env.action_space))
    print("Keyboard control:" + """
    W       : accelarate
    S       : decelarate
    A       : turn left
    D       : turn right
    U       : move arm forward
    J       : move arm back
    I       : move arm up
    K       : move arm down
    LEFT    : close gripper
    RIGHT   : open gripper
    ESC     : quit
    """)

    clock = pygame.time.Clock()
    action = np.zeros(env.action_space.shape[0])

    speed_delta = 0.1
    turn_delta = 0.1
    gripper_delta = 0.1
    arm_delta = 0.1

    stopping = False
    pygame.key.set_repeat(200, 200)
    while not stopping:
        clock.tick_busy_loop(freq)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stopping = True
                break
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == K.K_ESCAPE:
                stopping = True
                break
            elif event.key == K.K_w:
                action[0] += speed_delta
            elif event.key == K.K_s:
                action[0] -= speed_delta
            elif event.key == K.K_a:
                action[1] -= turn_delta
            elif event.key == K.K_d:
                action[1] += turn_delta
            elif event.key == pygame.K_u:
                action[2] += arm_delta
            elif event.key == pygame.K_j:
                action[2] -= arm_delta
            elif event.key == pygame.K_i:
                action[3] += arm_delta
            elif event.key == pygame.K_k:
                action[3] -= arm_delta
            elif event.key == K.K_LEFT:
                action[4] += gripper_delta
            elif event.key == K.K_RIGHT:
                action[4] -= gripper_delta

        action = np.clip(action, -1, 1)
        ret = env.step(action)
        env.render('human')

    env.close()


import atexit


def myexit():
    import traceback
    traceback.print_stack()


if __name__ == "__main__":
    FLAGS(sys.argv)
    logging.use_absl_handler()
    logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().use_absl_log_file(log_dir="C:\\Users\\xw\\tmp")
    logging.info("test log")
    # atexit.register(myexit)
    play()
