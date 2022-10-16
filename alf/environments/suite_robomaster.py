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
from absl import logging
import cv2
from functools import partial
import gym
from gym.spaces import Box
import numpy as np
import random
from unittest.mock import Mock

try:
    import robomaster
    import robomaster.robot
    from robomaster import chassis, protocol, util, gripper, robotic_arm, logger
except ImportError:
    robomaster = Mock()


def gripper_async_open(gripper, power):
    proto = protocol.ProtoGripperCtrl()
    proto._control = 1
    proto._power = util.GRIPPER_POWER_CHECK.val2proto(power)
    return gripper._send_async_proto(proto, protocol.host2byte(3, 6))


def gripper_async_close(gripper, power):
    proto = protocol.ProtoGripperCtrl()
    proto._control = 2
    proto._power = util.GRIPPER_POWER_CHECK.val2proto(power)
    return gripper._send_async_proto(proto, protocol.host2byte(3, 6))


def gripper_async_pause(gripper):
    proto = protocol.ProtoGripperCtrl()
    proto._control = 0
    proto._power = 0
    return gripper._send_async_proto(proto, protocol.host2byte(3, 6))


def send_async_action(self, action):
    action._action_id = action._get_next_action_id()
    proto = action.encode()
    proto._action_id = action._action_id
    action_msg = protocol.Msg(self._client.hostbyte, action.target, proto)
    self._client.send_msg(action_msg)


def robotic_arm_async_move(self, x, y):
    action = robotic_arm.RoboticArmMoveAction(x, y, z=0, mode=0)
    send_async_action(self, action)


def robotic_arm_async_moveto(self, x, y):
    action = robotic_arm.RoboticArmMoveAction(x, y, z=0, mode=1)
    send_async_action(self, action)


def chassis_auto_stop_timer(self, api="drive_speed"):
    if api == "drive_speed":
        logger.info("Chassis: drive_speed timeout, auto stop!")
        self.drive_speed(0, 0, 0)
    elif api == "drive_wheels":
        logger.info("Chassis: drive_wheels timeout, auto stop!")
        self.drive_wheels(0, 0, 0, 0)
    else:
        logger.warning("Chassis: unsupported api:{0}".format(api))


"""robomaster SDK bug
Servo.get_angle(): extra print(proto)
Chassis._auto_stop_timer(): mising one argument for drive_wheels()
"""

chassis.Chassis._auto_stop_timer = chassis_auto_stop_timer
gripper.Gripper.async_open = gripper_async_open
gripper.Gripper.async_close = gripper_async_close
gripper.Gripper.async_pause = gripper_async_pause
robotic_arm.RoboticArm.async_move = robotic_arm_async_move
robotic_arm.RoboticArm.async_moveto = robotic_arm_async_moveto


class RewardFunction(object):
    def __init__(self):
        self.task_space = None

    def __call__(self, obs):
        return 0, None


class RobotBatteryLevelLow(Exception):
    pass


class RobomasterEnv(gym.Env):
    """Gym environment for Robomaster EP robot.

    action:
    0: body forward/backword speed
    1: body turn speed
    2: arm x
    3: arm y
    4: gripper

    observation['state']:
    0-1:  arm x, arm y
    2:    gripper status
    3-5:  yaw, pitch, roll
    6-8:  x, y, z
    9-14: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

    Args:
        max_move_speed: forward speed, m/s
        max_turn_speed: turn speed, degree/s
        max_servo_speed: servo turning speed, degrees/s
        max_arm_x: horizontal movement, m
        max_arm_y: vertical movement, m
        max_gripper_power: positive for close, negative for open
        min_battery_level: stop if battery falls below this percent
    """

    def __init__(
            self,
            robot_ip: str,
            max_move_speed: float = 0.2,
            max_turn_speed: float = 20,
            max_servo_speed: float = 6,
            max_arm_x: float = 0.01,
            max_arm_y: float = 0.01,
            max_gripper_power: float = 100,
            use_servo_control: bool = True,
            dead_band=0.5,
            freq=10,
            min_battery_level=5,
            reward_function=RewardFunction(),
    ):

        self._robot = robomaster.robot.Robot()
        if robot_ip is not None:
            robomaster.config.ROBOT_IP_STR = robot_ip
            self._robot.initialize(conn_type="sta", proto_type="tcp")
        else:
            self._robot.initialize()
        robomaster.logger.error("test error")
        self._max_move_speed = max_move_speed
        self._max_turn_speed = max_turn_speed
        self._max_servo_speed = max_servo_speed
        self._max_arm_x = max_arm_x
        self._max_arm_y = max_arm_y
        self._max_gripper_power = max_gripper_power
        self._use_servo_control = use_servo_control
        self._dead_band = dead_band
        self._min_battery_level = min_battery_level

        width = 640
        height = 360

        self._robot.camera.start_video_stream(display=False, resolution="360p")

        self.observation_space = gym.spaces.Dict({
            "image":
                Box(low=0, high=255, shape=(width, height, 3), dtype=np.uint8),
            "state":
                Box(low=-1e10, high=1e10, shape=(15, ), dtype=np.float)
        })
        if reward_function.task_space is not None:
            self.observation_space['task'] = reward_function.task_space

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(5, ))

        def _subvec(subfunc, offset, dim):
            return subfunc(
                freq=freq,
                callback=partial(self._subvec, begin=offset, end=offset + dim))

        def _subscalar(subfunc, dim, mapping=None):
            return subfunc(
                freq=freq,
                callback=partial(self._subscalar, dim=dim, mapping=mapping))

        # arm x, arm y
        _subvec(self._robot.robotic_arm.sub_position, 0, 2)
        # gripper status
        _subscalar(self._robot.gripper.sub_status, 2, {
            "closed": -1,
            "open": 1,
            "normal": 0
        })
        # yaw, pitch, roll
        _subvec(self._robot.chassis.sub_attitude, 3, 3)
        # x, y, z
        _subvec(self._robot.chassis.sub_position, 6, 3)
        # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        _subvec(self._robot.chassis.sub_imu, 9, 6)
        # vgx, vgy, vgz, vbx, vby, vbz
        # vg: velocity in the coordinate system at power-on)
        # vb: velocity in ego-centric coordinate system)
        _subvec(self._robot.chassis.sub_velocity, 15, 6)
        # 0: static_flag, 1: up_hill, 2: down_hill, 3: on_slope, 4: pick_up,
        # 5: slip_flag, 6: impact_x, 7: impact_y, 8: impact_z, 9: roll_over,
        # 10: hill_static
        _subvec(self._robot.chassis.sub_status, 21, 11)

        self._robot.battery.sub_battery_info(
            freq=freq, callback=self._subbattery)

        self._state = np.zeros(32)
        self._reward_function = reward_function
        self._surface = None
        self._font = None
        self._battery_level = 100

    def _subvec(self, info, begin, end):
        self._state[begin:end] = info

    def _subscalar(self, info, dim, mapping):
        if mapping is not None:
            info = mapping[info]
        self._state[dim] = info

    def _subbattery(self, info):
        self._battery_level = info

    def step(self, action):
        if self._battery_level < self._min_battery_level:
            self._close()
            raise RobotBatteryLevelLow()

        dead_band = self._dead_band
        speed = 0

        def _calc_control(a, max_control):
            a = np.clip(a, -1, 1)
            if a < -dead_band:
                return max_control * (a + dead_band) / (1 - dead_band)
            elif a > dead_band:
                return max_control * (a - dead_band) / (1 - dead_band)
            else:
                return 0.

        speed = _calc_control(action[0], self._max_move_speed)
        turn_speed = _calc_control(action[1], self._max_turn_speed)
        if speed == 0 and turn_speed == 0:
            self._robot.chassis.drive_wheels(0, 0, 0, 0)
        else:
            self._robot.chassis.drive_speed(
                x=speed, y=0, z=turn_speed, timeout=0.2)

        if self._use_servo_control:
            servo1_speed = _calc_control(action[2], self._max_turn_speed)
            servo2_speed = _calc_control(action[3], self._max_turn_speed)
            # The speed unit for drive_speed is rpm
            self._robot.servo.drive_speed(1, servo1_speed / 6)
            self._robot.servo.drive_speed(2, servo2_speed / 6)
        else:
            arm_x = _calc_control(action[2], self._max_arm_x)
            arm_y = _calc_control(action[3], self._max_arm_y)
            if speed == 0 and turn_speed == 0:
                # robomaster will stop moving if trying to move the robotic arm
                # So we only allow moving the robotic arm when the robot is not moving
                # The unit for API is mm.
                self._robot.robotic_arm.async_move(
                    x=1000 * arm_x, y=1000 * arm_y)

        gripper_power = _calc_control(action[4], self._max_gripper_power)
        if gripper_power < 0:
            self._robot.gripper.async_open(-gripper_power)
        elif gripper_power > 0:
            self._robot.gripper.async_close(gripper_power)
        else:
            self._robot.gripper.async_pause()

        img = self._robot.camera.read_cv2_image(strategy="newest")
        # img = np.zeros((360, 640, 3), dtype=np.uint8)
        # [width, height, channels]
        img = np.transpose(img[:, :, ::-1], (1, 0, 2))

        obs = dict(image=img, state=self._state)
        reward, task = self._reward_function(obs)
        if task is not None:
            obs['task'] = task

        self._current_observation = obs
        self._current_reward = reward
        self._current_action = action

        return obs, reward, False, {}

    def render(self, mode: str):
        import cv2
        import pygame
        height, width = 720, 1280
        if self._surface is None:
            pygame.init()
            pygame.font.init()
            self._clock = pygame.time.Clock()
            if mode == 'human':
                self._surface = pygame.display.set_mode(
                    (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self._surface = pygame.Surface((width, height))

        image = self._current_observation['image']
        if width != image.shape[0] or height != image.shape[1]:
            image = cv2.resize(
                image, dsize=(height, width), interpolation=cv2.INTER_NEAREST)
        surface = pygame.surfarray.make_surface(image)
        self._surface.blit(surface, (0, 0))
        self._clock.tick()

        state = self._current_observation['state']
        np_precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=1)
        info_text = [
            'FPS:      %5.2f' % self._clock.get_fps(),
            'ARM:      (%4.3f, %4.3f)' % tuple(state[0:2]),
            'GRIPPER:  %2.0f' % state[2],
            'ROTATION: (%6.3f, %6.3f, %6.3f)' % tuple(state[3:6]),
            'POSITION: (%6.3f, %6.3f, %6.3f)' % tuple(state[6:9]),
            'IMU ACC:  (%6.3f, %6.3f, %6.3f)' % tuple(state[9:12]),
            'IMU GYRO: (%6.3f, %6.3f, %6.3f)' % tuple(state[12:15]),
            'BATTERY:  %5.2F' % self._battery_level,
            'ACTION:   (%6.3f, %6.3f, %6.3f, %6.3f, %6.3f)' % tuple(
                self._current_action),
            'REWARD:   %3.1f' % self._current_reward,
        ]
        if 'task' in self._current_observation:
            info_text.append(
                'TASK:     %s' % self._current_observation['task'])

        info_text = [info for info in info_text if info != '']
        np.set_printoptions(precision=np_precision)
        self._draw_text(info_text)
        if mode == 'human':
            pygame.display.flip()
        elif mode == 'rbg_array':
            rgb_img = pygame.surfarray.array3d(self._surface).swapaxes(0, 1)
            return rgb_img

    def _draw_text(self, texts):
        import os
        import pygame
        if self._font is None:
            font_name = 'courier' if os.name == 'nt' else 'mono'
            fonts = [x for x in pygame.font.get_fonts() if font_name in x]
            default_font = 'ubuntumono'
            mono = default_font if default_font in fonts else fonts[0]
            mono = pygame.font.match_font(mono)
            self._font = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        info_surface = pygame.Surface((240, 240))
        info_surface.set_alpha(100)
        self._surface.blit(info_surface, (0, 0))
        v_offset = 4
        for item in texts:
            surface = self._font.render(item, True, (255, 255, 255))
            self._surface.blit(surface, (4, v_offset))
            v_offset += 18

    def close(self):
        self._robot.chassis.drive_wheels(0, 0, 0, 0)
        self._robot.servo.drive_speed(1, 0)
        self._robot.servo.drive_speed(2, 0)
        self._robot.gripper.async_pause()
        self._robot.camera.stop_video_stream()
        self._robot.robotic_arm.unsub_position()
        self._robot.gripper.unsub_status()
        self._robot.chassis.unsub_attitude()
        self._robot.chassis.unsub_position()
        self._robot.chassis.unsub_imu()
        self._robot.close()


class QRCodeRewardFunction(RewardFunction):
    def __init__(self):
        self._reader = cv2.QRCodeDetector()
        self._words = ['apple', 'orange']
        self._current_word_id = random.randint(0, 1)
        self.task_space = Box(0, 1, shape=(), dtype=np.int64)

    def __call__(self, obs):
        image = obs['image']
        w, h, _ = image.shape
        center = obs['image'][w // 4:w * 3 // 4, h // 4:h * 3 // 4, :]
        center = np.transpose(center, (1, 0, 2))
        reward = 0
        try:
            data, bbox, qrimg = self._reader.detectAndDecode(center)
            if data == self._words[self._current_word_id]:
                self._current_word_id = 1 - self._current_word_id
                reward = 1
        except cv2.error:
            # OpenCV(4.6.0) floodfill.cpp:522: error: (-211:One of the arguments'
            # values is out of range) Seed point is outside of image in function
            # 'cv::floodFill'
            pass
        return reward, np.int64(self._current_word_id)
