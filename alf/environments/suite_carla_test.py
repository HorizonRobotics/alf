# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import pprint
import sys
import torch

import alf
from alf.environments import suite_carla

flags.DEFINE_bool('manual', False, "Manual control")
FLAGS = flags.FLAGS


class SuiteCarlaTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_carla.is_available():
            self.skipTest('suite_carla is not available.')

    def test_carla(self):
        alf.config('suite_carla.Player', with_bev_sensor=True)

        env = suite_carla.CarlaEnvironment(4, 'Town01')
        logging.info(
            "observation_spec: %s" % pprint.pformat(env.observation_spec()))
        logging.info(
            "observation_desc: %s" % pprint.pformat(env.observation_desc()))
        logging.info("action_spec: %s" % pprint.pformat(env.action_spec()))
        logging.info("action_desc: %s" % pprint.pformat(env.action_desc()))
        action_spec = env.action_spec()

        try:
            for _ in range(10):
                action = action_spec.sample([env.batch_size])
                logging.info("action: %s" % action)
                action[:, 2] = 0
                for _ in range(10):
                    time_step = env.step(action)
                    logging.debug("goal: %s, gnss: %s reward=%s" %
                                  (time_step.observation['goal'][0],
                                   time_step.observation['gnss'][0],
                                   float(time_step.reward[0][0])))
        finally:
            env.close()


def play(env):
    logging.info(
        "observation_spec: %s" % pprint.pformat(env.observation_spec()))
    logging.info(
        "observation_desc: %s" % pprint.pformat(env.observation_desc()))
    logging.info("action_spec: %s" % pprint.pformat(env.action_spec()))
    logging.info("action_desc: %s" % pprint.pformat(env.action_desc()))
    logging.info("Keyboard control:" + """
    W/UP         : throttle
    S/DOWN       : brake
    A/LEFT       : steer left
    D/RIGHT      : steer right
    SPACE        : steer ahead
    Q            : toggle reverse
    ESC          : quit
    """)

    action = env.action_spec().zeros([env.batch_size])
    THROTTLE = 0
    STEER = 1
    BRAKE = 2
    REVERSE = 3

    import pygame
    import pygame.locals as K
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    steer = 0.

    stopping = False
    while not stopping:
        clock.tick_busy_loop(1000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stopping = True
                break
            if event.type != pygame.KEYUP:
                continue
            if event.key == K.K_ESCAPE:
                stopping = True
                break
            if event.key == K.K_q:
                action[:, REVERSE] = 1 - action[:, REVERSE]
            if event.key == K.K_SPACE:
                steer = 0.

        keys = pygame.key.get_pressed()
        if keys[K.K_UP] or keys[K.K_w]:
            action[:, THROTTLE] = torch.min(action[:, THROTTLE] + 0.01,
                                            torch.tensor(1.))
        else:
            action[:, THROTTLE] = 0

        if keys[K.K_DOWN] or keys[K.K_s]:
            action[:, BRAKE] = torch.min(action[:, BRAKE] + 0.2,
                                         torch.tensor(1.))
        else:
            action[:, BRAKE] = 0

        steer_increment = 0.005
        if keys[K.K_LEFT] or keys[K.K_a]:
            if steer > 0:
                steer = 0
            else:
                steer -= steer_increment
        elif keys[K.K_RIGHT] or keys[K.K_d]:
            if steer < 0:
                steer = 0
            else:
                steer += steer_increment
        steer = min(0.7, max(-0.7, steer))
        action[:, STEER] = steer

        time_step = env.step(action)
        if time_step.step_type[0] == alf.data_structures.StepType.LAST:
            action = env.action_spec().zeros([env.batch_size])
            steer = 0.

        env.render("human")


def main():
    FLAGS(sys.argv)
    if not FLAGS.manual:
        return False

    logging.use_absl_handler()
    logging.set_verbosity(logging.INFO)

    alf.config('suite_carla.Player', with_bev_sensor=True)
    env = suite_carla.CarlaEnvironment(
        batch_size=1,
        map_name='Town01',
        num_other_vehicles=20,
        num_walkers=20,
        day_length=100)
    try:
        play(env)
    finally:
        env.close()
    return True


if __name__ == '__main__':
    if not main():
        alf.test.main()
