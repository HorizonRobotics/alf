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

import pickle
import os
import sys, math
from copy import copy, deepcopy
from pdb import set_trace

import Box2D
import cv2
#import pandas as pd
import numpy as np
from Box2D import b2Vec2
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape,
                      revoluteJointDef, contactListener)
from PIL import Image

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl
"""File adapted from https://raw.githubusercontent.com/NotAnyMike/gym/master/gym/envs/box2d/car_racing.py
Several differences with the original file:
1. When computing reward, separately return the obstacle and out-of-track info.
2. Fix the num_obstacles bug.
3. No early termination (done=False always)
4. Obstacles placed more densely.
"""

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discreet control is reasonable in this environment as well, on/off discretisation is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles in track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position, gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_H = 700
WINDOW_W = int(WINDOW_H * 1.5)

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50
ZOOM = 2.7  # Camera zoom, 0.25 to take screenshots, default 2.7
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
NUM_TILES_FOR_AVG = 5  # The number of tiles before and after to takeinto account for angle
TILES_IN_SCREEN = 20  # The number of tiles that fit in a screen (arange as track)
HARD_NEG_REWARD = 100  # For actions like going out or timeout
SOFT_NEG_REWARD = 0.1  # For actions like living
MIN_SEGMENT_LENGHT = 8

ROAD_COLOR = [0.4, 0.4, 0.4]

OBSTACLE_NAME = 'obstacle'
OBSTACLE_VALUE = -10
TILE_NAME = 'tile'
BORDER_NAME = 'border'
GRASS_NAME = 'grass'

# Debug actions
SHOW_NEXT_N_TILES = 0  # Show the next N tiles
SHOW_ENDS_OF_TRACKS = 0  # Shows with red dots the end of track
SHOW_START_OF_TRACKS = 0  # Shows with green dots the end of track
SHOW_INTERSECTION_POINTS = 0  # Shows with yellow dots the intersections of main track
SHOW_GROUP_INTERSECTIONS = 0  # Shows each group of intersections in its own hippie color
SHOW_XT_JUNCTIONS = 0  # Shows in dark and light green the t and x junctions
SHOW_JOINTS = 0  # Shows joints in white
SHOW_TURNS = 0  # Shows the 10 hardest turns
SHOW_AXIS = 0  # Draws two lines where the x and y axis are
ZOOM_OUT = 0  # Shows maps in general and does not do zoom
if ZOOM_OUT: ZOOM = 0.25  # Complementary to ZOOM_OUT
# in lane -> 0: left, 1: right

# Warning, Not optimized for training
SHOW_BETA_PI_ANGLE = 0  # Shows the direction of the beta+pi/2 angle in each tile

# This will forbid predicting turns that has the oposite direction of the
# do not modify it if you dont know what you are doing, this can have slight changes
# in the behaviour of the game and you will not realise
FORBID_HARD_TURNS_IN_INTERSECTIONS = False


def key_press_example(k, mod):
    """
    Example callback function
    """
    if k == key.LEFT:
        # change a value or print something
        pass


def key_release_example(k, mod):
    """
    Example callback function
    """
    if k == key.LEFT:
        # change a value or print something
        pass


def original_reward_callback(env):
    reward, done = -0.1, False

    left = env.info['count_left_delay'] > 0
    right = env.info['count_right_delay'] > 0
    not_visited = env.info['visited'] == False

    visited_count = env.info['visited'].sum()

    count = ((left | right) & not_visited).sum()
    reward += 1000 / len(env.track) * count

    env.info['visited'][left | right] = True
    env.info['count_right_delay'] = env.info['count_right']
    env.info['count_left_delay'] = env.info['count_left']

    # if outside the map
    x, y = env.car.hull.position
    if not done and abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
        done = True
        reward -= HARD_NEG_REWARD

    if env.reward > 1000 or env.reward < -1000:
        # if too good or too bad
        done = True
    else:
        if not env.allow_outside:
            reward, done = env.check_outside(reward, done)
        reward, done = env.check_timeout(reward, done)

        if visited_count == len(env.track):
            done = True
        # if outside the map
        x, y = env.car.hull.position
        if not done and abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
            done = True
            reward -= HARD_NEG_REWARD

    return reward, reward, done, {}


def default_reward_callback(env):
    reward = -SOFT_NEG_REWARD

    left = env.info['count_left_delay'] > 0
    right = env.info['count_right_delay'] > 0
    track0 = env.info['track'] == 0
    track1 = env.info['track'] == 1
    not_visited = env.info['visited'] == False

    # To allow changes of lane in intersections without lossing points
    if (left & right & track0).sum() > 0 and (left & right & track1).sum() > 0:
        factor = 3
    elif (left & right & track1).sum() > 0 and ((
        (left | right) & track0).sum() == 0):
        factor = 3
    elif (left & right & track0).sum() > 0 and ((
        (left | right) & track1).sum() == 0):
        factor = 3
    else:
        factor = 1

    # Positive reward
    reward += (((left | right) & not_visited).sum() / factor)
    env.tile_visited_count += (left | right).sum()

    # Negative reward
    reward += env.check_obstacles_touched()

    full_reward = reward
    reward = np.clip(reward, env.min_step_reward, env.max_step_reward)

    env.info['visited'][left | right] = True
    env.info['count_right_delay'] = env.info['count_right']
    env.info['count_left_delay'] = env.info['count_left']

    env._update_obstacles_info()

    done = False

    if env.reward > 1000 or env.reward < -200:
        # if too good or too bad
        done = True
    else:
        if not env.allow_outside:
            reward, done = env.check_outside(reward, done)
        reward, done = env.check_timeout(reward, done)
        reward, done = env.check_unvisited_tiles(reward, done)

        # if outside the map
        x, y = env.car.hull.position
        if not done and abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
            done = True
            reward -= HARD_NEG_REWARD

    return reward, full_reward, done, {}


def safe_reward_callback(env):
    reward = 0.

    left = env.info['count_left_delay'] > 0
    right = env.info['count_right_delay'] > 0
    not_visited = env.info['visited'] == False

    # Positive reward
    count = ((left | right) & not_visited).sum()
    reward += 1000. / len(env.track) * count

    #reward += (( (left | right) & not_visited).sum() / factor)
    new_tile_visited = (left | right).sum()
    env.tile_visited_count += (left | right).sum()

    # Negative reward
    obstacles_reward = env.check_obstacles_touched()
    obstacles_reward = -(obstacles_reward != 0)

    full_reward = reward
    reward = np.clip(reward, env.min_step_reward, env.max_step_reward)

    env.info['visited'][left | right] = True
    env.info['count_right_delay'] = env.info['count_right']
    env.info['count_left_delay'] = env.info['count_left']

    env._update_obstacles_info()

    tiles_remaining = len(env.track) - env.info['visited'].sum()
    done = (tiles_remaining == 0)

    prev_out_of_track = env.prev_out_of_track
    _, out = env.check_outside(reward, False)
    out_of_track_reward = 0.
    if env.num_obstacles == 0 and not prev_out_of_track:
        # only enforce out-of-track when there is no obstacle
        out_of_track_reward = -float(out)

    return reward, full_reward, done, {
        'obstacles_reward': obstacles_reward,
        'out_of_track_reward': out_of_track_reward,
        "success": done
    }


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile: return

        if tile.typename != OBSTACLE_NAME:
            tile.color[0] = ROAD_COLOR[0]
            tile.color[1] = ROAD_COLOR[1]
            tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__: return

        if begin:
            if tile.typename == TILE_NAME:
                self.env.add_current_tile(tile.id, tile.lane)
                obj.tiles.add(tile)

                #self.env.reward_tiles.add(tile)
                if tile.lane == 1:
                    self.env.info['count_right'][tile.id] += 1
                    self.env.info['count_right_delay'][tile.id] += 1
                else:
                    self.env.info['count_left'][tile.id] += 1
                    self.env.info['count_left_delay'][tile.id] += 1
            elif tile.typename == OBSTACLE_NAME:
                #self.env.reward_tiles.add(tile)
                self.env.obstacle_contacts['count'][tile.id] += 1
                self.env.obstacle_contacts['count_delay'][tile.id] += 1
        else:
            if tile.typename == TILE_NAME:
                obj.tiles.remove(tile)
                self.env.remove_current_tile(tile.id, tile.lane)

                if tile.lane == 1:
                    self.env.info['count_right'][tile.id] -= 1
                else:
                    self.env.info['count_left'][tile.id] -= 1

                # Registering last contact with track
                self.env.update_contact_with_track()
            elif tile.typename == OBSTACLE_NAME:
                self.env.obstacle_contacts['count'][tile.id] -= 1


class CarRacing(gym.Env, EzPickle):
    '''
    Controls some attributes of the game, such as the number of tracks (num_tracks)
    which is a proxy to control the complexity of map, the number of lanes (num_lanes_changes) and
    the probability of finding an obstacle (prob_osbtacle).
    Only call this method once to set the parameters and do not called outside this env, only use
    make method (i.e. init)

    num_tracks:        (int 1)       Number of tracks, in {1,2}, 1: simple, 2: complex, you can
                                     modify the code to allow more than one track, but good beheviour
                                     is not garanted

    num_lanes:         (int 1)       Number of lanes in track, > 0 ({1,2})

    num_lanes_changes  (int 0)       Number of changes from 2 to 1 or viceversa, this
                                     is ultimately transform as a probability over the
                                     total number of points in track

    num_obstacles      (flt 0)       The probability of finding an obstacle a point of
                                     the track, [0,1]

    max_single_lane    (int 0)       The maximum number of tiles that a single lane road
                                     can have before becoming two lanes again

    allow_reverse      (bool 0)      Allow the car going in reverse, if true key.DOWN goes
                                     backwards and action_space changes

    max_time_out       (flt 2.0)     Max time that car is allowed to be outside of track or stoped
                                     before reseting the env (sending done=True), if
                                     max_time_out == 0 then car can be outside without any problem.
                                     max_time_out is in "seconds" but given that in training time
                                     goes faster it is really in 1xFPS (max_time_out = 1/FPS means
                                     the car is allow only to be outside the track for one frame.
                                     Usually FPS is 59, see the constant in this file, but in
                                     playing time (runing this file) max_time_out is approximately
                                     in seconds to allow you have a sense of the magnitude
                                     This is not necessary time of not earning rewards, only time
                                     outside the track or without moving

    grayscale          (bool 0)      Whether or not use grayscale for the state representation,
                                     the state will be 96x96 of values between 0,255 as rbg state

    show_info_panel    (bool 1)      Whether or not show the info black panel at the bottom of the state

    frames_per_state   (int 1)       The number of concatenated frames for the state, =3 means
                                     that the state will be the last 3 frames of the env

    discretize_actions (str "hard")  How to discretize the action space, a value in {None,"soft",
                                     "hard"}. WARNING "soft" method is not implemented yet
                                     - None means space is continuous
                                     - "hard" actions are 4 [NOTHING, LEFT, RIGHT, ACCELERATE, BREAK]
                                     - "soft" actions are 7 [NOTHING, SOFT_LEFT, HARD_LEFT, SOFT_RIGHT,
                                     HARD_RIGHT, SOFT_STRAIGHT, HARD_STRAIGHT, SOFT_BREAK,
                                     HARD_BREAK]

    min_step_reward (flt -inf)    To limit the min reward the agent can have in an episode, -np.inf means no limit
                                     it is good to control the gradient
                                     Having max and min values for reward makes learning more stable (i.e. less
                                     variance) but so far it does not make it learn faster

    max_step_reward (flt +inf)    To limit the max reward the agent can have in an episode, +np.inf means no limit.
                                     It is good to control the learned speed of the car, to avoid high speeds 1 is
                                     ok. It is also good to control the gradient.
                                     Having max and min values for reward makes learning more stable (i.e. less
                                     variance) but so far it does not make it learn faster

    reward_fn          (function)    The default value is default_reward_callback. This
                                     paramter is a function which can replace the reward
                                     function, it has to take 4 variables as inputs.
                                     tile,begin,local_vars,global_vars. Tile is the current tile,
                                     begin is True if the contact with tile has just started,
                                     local_vars and global_vars are locals() and globals()
                                     respectively. The function should manually set the reward
                                     on local_vars['env'].reward. see the function
                                     default_reward_callback for an example of how a
                                     reward function works

    key_press_fn        (function)   A function that will be called each time a key is pressed in the
                                     window (viewer). See key_press_example(k, mod) function to see
                                     an example of how this can work. The default is None

    key_release_fn      (function)   A function that willl be called each time a key is release in
                                     window (viewer). See key_release_example(k, mod) function to see
                                     an example of how this can work. The default is None

    verbose             (int 1)      1: Print useful information, such as fail creating track msg, etc

    random_obstacle_x_position (True)Whether or not to have the obstacles in a random x position or
                                     at the begining of each size of the track

    load_tracks_from    (str None)   The folder from where the env can read and load the tracks,
                                     there must be a id.pkl for each track containing the info, track
                                     and tracks from the env. There also has to be a file called
                                     list.csv containing the ids with bool columns x,t,obstacles

    allow_outside       (bool True)  Whether or not to allow the car going outside or not

    '''
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, **kwargs):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.full_reward = 0.0
        self.prev_reward = 0.0
        self.highest_reward = 0.0
        self._current_nodes = {
        }  # A dict of dicts, dict[id][lane]=direction you can be in more than one tile at the same time, e.g. intersections
        self._next_nodes = []  # A list of lists of dictionaries
        self.possible_hard_actions = ("NOTHING", "LEFT", "RIGHT", "ACCELERATE",
                                      "BREAK")
        self.possible_soft_actions = (
            "NOTHING", "SOFT_LEFT", "HARD_LEFT", "SOFT_RIGHT", "HARD_RIGHT",
            "SOFT_ACCELERATE", "HARD_ACCELERATE", "SOFT_BREAK",
            "HARD_BREAK")  # Not implemented
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))

        # Config
        self._set_config(**kwargs)
        self._org_config = deepcopy(kwargs)
        self._steps_in_episode = 0

    def _set_config(
            self,
            num_tracks=1,
            num_lanes=1,
            num_lanes_changes=0,
            num_obstacles=0,
            max_single_lane=0,
            max_time_out=2.0,
            grayscale=False,
            observe_car_status=True,
            show_info_panel=False,
            frames_per_state=1,
            discretize_actions='hard',
            allow_reverse=0,
            min_step_reward=-np.inf,
            max_step_reward=+np.inf,
            animate_zoom=False,
            reward_fn=default_reward_callback,
            key_press_fn=None,
            key_release_fn=None,
            verbose=1,
            random_obstacle_x_position=True,
            random_obstacle_shape=True,
            auto_render=False,
            allow_outside=True,
            load_tracks_from=None,
    ):

        self.allow_outside = allow_outside
        self.auto_render = auto_render
        self.reward_fn = reward_fn
        self.random_obstacle_shape = random_obstacle_shape
        self.random_obstacle_x_position = random_obstacle_x_position
        self.verbose = verbose
        self.animate_zoom = animate_zoom

        if load_tracks_from is not None:
            if os.path.isdir(load_tracks_from):
                self.load_tracks_from = load_tracks_from
                self.tracks_df = pd.read_csv(
                    self.load_tracks_from + "/list.csv", index_col=0)
            else:
                raise Exception(
                    "Folder specified in load_tracks_from does not exists")
        else:
            self.load_tracks_from = None

        # Setting key press callback functions
        self.key_press_fn = key_press_fn
        self.key_release_fn = key_release_fn

        # Number of lanes, 1 or 2
        self.num_lanes = num_lanes if num_lanes in [1, 2] else 1

        # Number of tracks, this control the complexity of the map
        self.num_tracks = num_tracks if num_tracks > 0 and num_tracks <= 2 else 1

        # Number of obstacles in the track
        self.num_obstacles = num_obstacles if num_obstacles >= 0 else 0

        # Number of points where lanes change from 1 lane to two and viceversa
        self.num_lanes_changes = num_lanes_changes if num_lanes_changes >= 0 else 0

        # Max number of tiles of a single lane road
        self.max_single_lane = max_single_lane if max_single_lane > 10 else 50

        # Allow reverse
        self.allow_reverse = allow_reverse
        min_speed = -1 if self.allow_reverse else 0

        # Max time out of track
        self.max_time_out = max_time_out if max_time_out >= 0 else 2.0

        # Grayscale
        self.grayscale = grayscale
        state_shape = [STATE_H, STATE_W]
        if not self.grayscale:
            state_shape.append(3)
            if frames_per_state > 1:
                print("####################################")
                print("Warning: making frames_per_state = 1")
                print("No support for several frames in RGB")
                frames_per_state = 1

        # Show or not back bottom info panel
        self.show_info_panel = show_info_panel

        # Frames per state
        self.frames_per_state = frames_per_state if frames_per_state > 0 else 1
        if self.frames_per_state > 1:
            state_shape.append(self.frames_per_state)

            lst = list(range(self.frames_per_state))
            self._update_index = [lst[-1]] + lst[:-1]

        # not including "soft" because it is not implemented yet
        self.discretize_actions = discretize_actions if discretize_actions in [
            None, "hard"
        ] else "hard"

        if max_step_reward < min_step_reward:
            raise AttributeError(
                "max_step_reward must be greater than min_step_reward")
        self.max_step_reward = max_step_reward
        self.min_step_reward = min_step_reward

        state_shape = tuple(state_shape)
        # Incorporating reverse now the np.array([-1,0,0]) becomes np.array[-1,-1,0]
        if self.discretize_actions == "soft":
            self.action_space = spaces.Discrete(
                len(self.possible_soft_actions))
        elif self.discretize_actions == "hard":
            self.action_space = spaces.Discrete(
                len(self.possible_hard_actions))
        else:
            self.action_space = spaces.Box(
                np.array([-1, min_speed, 0]),
                np.array([+1, +1, +1]),
                dtype=np.float32)  # steer, gas, brake

        self.observe_car_status = observe_car_status
        self.observation_space = spaces.Box(
            low=0, high=255, shape=state_shape, dtype=np.uint8)
        if observe_car_status:
            self.observation_space = spaces.Dict({
                'rgb':
                    self.observation_space,
                'car':
                    gym.spaces.Box(
                        low=-float('inf'),
                        high=float('inf'),
                        shape=(11, ),
                        dtype=np.float32)
            })

        # Set custom reward function
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World(
            (0, 0), contactListener=self.contactListener_keepref)

    def get_org_config(self):
        return str(self._org_config)

    def check_unvisited_tiles(self, reward, done):
        if self.info['visited'].sum() / self.info.shape[0] > 0.5:
            done = True
        return reward, done

    def check_timeout(self, reward, done):
        if self.t - self.last_touch_with_track > self.max_time_out and \
                self.max_time_out > 0.0:
            # if too many seconds outside the track
            done = True
            if self.verbose > 0:
                print("done by time")
            reward -= HARD_NEG_REWARD
        return reward, done

    def _is_outside(self):
        right = self.info['count_right'] > 0
        left = self.info['count_left'] > 0
        if (left | right).sum() == 0:
            return True
        else:
            return False

    def check_outside(self, reward, done):
        if self._is_outside():
            # In case it is outside the track
            done = True
            self.prev_out_of_track = True
            reward -= HARD_NEG_REWARD
        else:
            self.prev_out_of_track = False
        return reward, done

    def _update_obstacles_info(self):
        different_count = self.obstacle_contacts[
            'count'] != self.obstacle_contacts['count_delay']
        zero_count = self.obstacle_contacts['count'] == 0
        in_contact = self.obstacle_contacts['count'] > 0
        self.obstacle_contacts['visited'][in_contact] = True
        # Next line forgets the obstacles touched but not currently being touch
        self.obstacle_contacts['visited'][(different_count)
                                          & (zero_count)] = False
        self.obstacle_contacts['count_delay'] = self.obstacle_contacts['count']

    def check_obstacles_touched(self, obstacle_value=OBSTACLE_VALUE):
        obs_not_visited = self.obstacle_contacts['visited'] == False
        obs_count = (self.obstacle_contacts[obs_not_visited]['count_delay'] >
                     0).sum()
        obstacle_rwd = 0
        if obs_count > 0:
            obstacle_rwd = obstacle_value
        return obstacle_rwd

    def update_contact_with_track(self):
        self.last_touch_with_track = self.t

    def set_velocity(self, velocity=[0.0, 0.0]):
        self.car.hull.linearVelocity.Set(velocity[0], velocity[1])
        for w in self.car.wheels:
            w.linearVelocity = velocity
            w.omega = np.linalg.norm(velocity)

    def set_speed(self, speed):
        ang = self.car.hull.angle + math.pi / 2
        velocity_x = math.cos(ang) * speed
        velocity_y = math.sin(ang) * speed
        self.set_velocity([velocity_x, velocity_y])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road: return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def place_agent(self, position):
        '''
        position = [beta,x,y]
        '''
        if 'car' in locals() and self.car is not None: self.car.destroy()
        self.car = Car(self.world, *position)

        # This a better way to do it but coordinates changes slightly, Dont know why
        # TODO research this weird behaviour
        #self.car.hull.position.Set(position[1],position[2])#tuple(position[1:])
        #self.car.hull.angle = float(position[0])

    def add_current_tile(self, id, lane):
        ######## Calculating direction
        id_relative = id

        # TODO remove this if and only leave the else
        try:
            if self.info[id]['track'] > 0:
                id_relative -= len(self.tracks[self.info[id]['track'] - 1])
        except Exception as e:
            print(e)
            print("error, see line 512 of car_racing")
            print("info len", str(self.info.shape))
            print("track len", str(self.track.shape))
            return None
        next_id = (id_relative + 1) % len(self.tracks[self.info[id]['track']])
        last_id = (id_relative - 1) % len(self.tracks[self.info[id]['track']])

        keys = self._current_nodes.keys()
        #if id-id_relative+next_id in keys:
        if abs((self.track[id, 0, 1] - self.car.hull.angle + np.pi / 2) %
               (np.pi * 2)) > np.pi:
            direction = -1
        else:
            direction = 1

        ######## Adding it to the current node
        if id in self._current_nodes.keys():
            self._current_nodes[id][lane] = direction
        else:
            self._current_nodes[id] = {lane: direction}
        #######

        self._remove_prediction(id, lane, direction)

        # Cleaning next_nodes from empty lists
        self._next_nodes = [item for item in self._next_nodes if len(item) > 0]

        # Make sure we have SHOW_NEXT_N_TILES predictions
        self._check_predictions()

    def _check_predictions(self):
        ####### predictions
        while len(self._next_nodes) < SHOW_NEXT_N_TILES:
            if len(self._next_nodes) == 0:
                # if there is no prediction
                elems = self._current_nodes
            else:
                # else take the last predictions
                elems = self._next_nodes[-1]
            next_nodes = {}
            for id, vals in elems.items():
                for lane, direction in vals.items():
                    tmp_preds = [
                    ]  # This is only used in the case of direction 0,
                    # direction zero means you can take the two directions, that only
                    # happens at an intersection
                    if direction == 0:
                        tmp_preds.append(self._get_next_node(id, lane, +1))
                        tmp_preds.append(self._get_next_node(id, lane, -1))
                    else:
                        tmp_preds.append(
                            self._get_next_node(id, lane, direction))
                    for tmp_pred in tmp_preds:
                        for k, v in tmp_pred.items():
                            if k not in next_nodes: next_nodes[k] = {}
                            for tmp_lane, tmp_dir in v.items():
                                next_nodes[k][tmp_lane] = tmp_dir

            self._next_nodes.append(next_nodes)

    def _remove_prediction(self, id, lane, direction):
        ###### Removing current new tile from nexts
        if len(self._next_nodes) > 0:
            if id in self._next_nodes[0] and lane in self._next_nodes[0][id] \
                    and self._next_nodes[0][id][lane] == direction:
                # If the current new tile is a prediction, then there is
                # no need to predcit all N next tiles again
                if len(self._next_nodes[0][id]) > 1:
                    del self._next_nodes[0][id][lane]
                else:
                    del self._next_nodes[0][id]
            else:
                # If tile is not the next prediction means that the
                # car is somewhere else and we need to predict all 10 again
                self._next_nodes = []

    def remove_current_tile(self, id, lane):
        if id in self._current_nodes:
            if len(self._current_nodes[id]) > 1:
                del self._current_nodes[id][lane]
            else:
                del self._current_nodes[id]
            if len(self._current_nodes) == 0:
                self._next_nodes = []

    def _update_predictions(self):
        self._trail_nodes = {
            k: v
            for l in self._next_nodes for k, v in l.items()
        }
        self._trail_nodes.update(self._current_nodes)

    def _get_next_node(self, id, lane, direction):
        """
        this will return a dict of elements, elem is a dict of [id][lane] = direction
        """
        # if it is the end of the row or the beginning in the opposite direction
        if (self.info[id]['end'] == True and direction == 1) or \
           (self.info[id]['start'] == True and direction == -1):
            return {}
        else:
            # Else calculate the next tile
            id_relative = id
            if self.info[id]['track'] > 0:
                id_relative -= len(self.tracks[self.info[id]['track'] - 1])

            next_id = (id_relative + direction) % len(
                self.tracks[self.info[id]['track']])  # TODO direction 0

            # If the next tile is in an intersection add all the following intersections
            intersection = self.info[id - id_relative +
                                     next_id]['intersection_id']
            next_nodes = {}
            if intersection != -1:
                for tmp_id in np.where(
                        self.info['intersection_id'] == intersection)[0]:
                    next_nodes[tmp_id] = {}
                    if self.info[tmp_id]['track'] > 0:
                        if self.info[tmp_id]['end']:
                            direction = -1
                        elif self.info[tmp_id]['start']:
                            direction = +1
                        else:
                            if tmp_id != id - id_relative + next_id:
                                direction = 0
                        if any(self.info[list(self._current_nodes.keys())]
                               ['track'] > 0):
                            next_nodes[tmp_id][lane] = direction
                        else:
                            next_nodes[tmp_id][1] = direction
                            next_nodes[tmp_id][0] = direction
                    else:
                        # if it is not end or start but still and intersection then it is in the main track
                        if tmp_id == next_id:
                            # if it is the current one keep the direction and lane
                            next_nodes[tmp_id][lane] = direction
                        else:
                            # add both directions and lanes
                            next_nodes[tmp_id][1] = 0
                            next_nodes[tmp_id][0] = 0
            else:
                if not id - id_relative + next_id in next_nodes.keys():
                    next_nodes[id - id_relative + next_id] = {}
                next_nodes[id - id_relative + next_id][lane] = direction

            # remove nodes that have a complementary directions
            for k, lanes in deepcopy(list(next_nodes.items())):
                for lane, direction in lanes.items():
                    # get direction of tile + (2pi if dir == -1)
                    # if new direction is complementary remove it
                    remove = False
                    if FORBID_HARD_TURNS_IN_INTERSECTIONS:
                        #This will avoid prediction any road that is almost 180 +/- 45 degrees opposite from the direction of the car
                        direction_tmp = self.track[k, 0, 1]
                        if direction == -1:
                            direction_tmp += np.pi
                        direction_tmp %= (2 * np.pi)

                        if (direction_tmp - (self.car.hull.angle + np.pi) +
                                np.pi / 4) % (np.pi * 2) < np.pi / 2:
                            remove = True
                    else:
                        # forbid returning
                        if k in self._current_nodes:
                            remove = True

                    if remove:
                        if len(next_nodes[k]) == 1:
                            del next_nodes[k]
                        else:
                            del next_nodes[k][lane]
            return next_nodes

    def _to_relative(self, id):
        return id - (self.info['track'] < self.info[id]['track']).sum()

    def understand_intersection(self, node_id, direction):
        """
        This method will return a dictionary of the form
        {'right': node_id, 'left': node_id, 'straight': node_id}

        Direction is -1 or 1
        """
        intersection = {'left': None, 'right': None, 'straight': None}
        relative_id = self._to_relative(node_id)
        track_id = self.info[node_id]['track']
        _, angle_org, x, y = self.track[node_id][1]

        angle = (angle_org + direction * np.pi / 2) % (np.pi * 2)

        # Populating straight node
        if self.info[node_id]['t'] and self.info[node_id]['track'] > 0:
            intersection['straight'] = None
        else:
            # Sometimes where the tracks only touch the straight route is in the other track
            # The solution was to not generate touching tracks
            s_node = (relative_id + direction) % len(
                self.tracks[self.info[node_id]['track']])
            s_node = node_id - relative_id + s_node
            intersection['straight'] = [s_node, direction]

        # starting to populate left and right node
        nodes = np.where(
            (self.info['intersection_id'] == self.info[node_id]['intersection_id']) & \
            (self.info['track'] != track_id))[0]

        candidates = []
        # some times the first node of the right turn starts
        # slightly to the left (opposite side) of intersection)
        # that's why there is a secure_candidates
        secure_candidates = []
        for node in nodes:
            if self.info[node]['track'] == 0:
                candidates.append((node + 1) % len(self.tracks[0]))
                secure_candidates.append(
                    (node + int(MIN_SEGMENT_LENGHT / 2)) % len(self.tracks[0]))
                candidates.append((node - 1) % len(self.tracks[0]))
                secure_candidates.append(
                    (node - int(MIN_SEGMENT_LENGHT / 2)) % len(self.tracks[0]))
            else:
                # The not start or not end is in case the node has not broken into two nodes
                # and it is still a continuous road
                if self.info[node]['end'] or not self.info[node]['start']:
                    relative_candidate = self._to_relative(node)
                    next_relative_candidate_safe = (relative_candidate - int(MIN_SEGMENT_LENGHT/2))\
                            %len(self.tracks[self.info[node]['track']])
                    next_relative_candidate = (relative_candidate -1)\
                            %len(self.tracks[self.info[node]['track']])
                    candidates.append(node - relative_candidate +
                                      next_relative_candidate)
                    secure_candidates.append(node - relative_candidate +
                                             next_relative_candidate_safe)
                if self.info[node]['start'] or not self.info[node]['end']:
                    relative_candidate = self._to_relative(node)
                    next_relative_candidate_safe = (relative_candidate + int(MIN_SEGMENT_LENGHT/2))\
                            %len(self.tracks[self.info[node]['track']])
                    next_relative_candidate = (relative_candidate +1)\
                            %len(self.tracks[self.info[node]['track']])
                    candidates.append(node - relative_candidate +
                                      next_relative_candidate)
                    secure_candidates.append(node - relative_candidate +
                                             next_relative_candidate_safe)

        angles = []
        left_dirs = []
        tmp_angles = []
        for tmp_id_true, tmp_id in zip(candidates, secure_candidates):
            beta_tmp, x1, y1 = self.track[tmp_id][0, 1:]
            tmp_angle = math.atan2(y1 - y, x1 - x)
            tmp_angle -= angle
            tmp_angle %= (np.pi * 2)
            tmp_angle -= np.pi
            angles.append(tmp_angle)

            tmp_angle_org = angle_org
            if direction < 0:
                tmp_angle_org = (angle_org + np.pi) % (np.pi * 2)
            tmp_dir = -1 if (beta_tmp - tmp_angle_org) % (
                np.pi * 2) > np.pi else 1
            left_dirs.append(tmp_dir)
            tmp_angles.append(tmp_angle)
            if tmp_angle < 0:
                intersection['left'] = [tmp_id_true, tmp_dir]
                #if self.info[tmp_id]['intersection_id'] == -1:
                #    intersection['right'] = tmp_id
            else:
                tmp_dir *= -1
                intersection['right'] = [tmp_id_true, tmp_dir]
                #if self.info[tmp_id]['intersection_id'] == -1:
                #    intersection['left'] = tmp_id

        angles = np.array(angles)
        if (angles > 0).sum() >= 2 or (angles < 0).sum() >= 2 \
                or (-1 in left_dirs and 1 in left_dirs):
            # TODO consider not using arg min for the direction, use the direction
            # of the angle that most looks like left or right
            angles -= angles.mean()
            left = angles.argmax()
            right = angles.argmin()
            intersection['left'] = [candidates[left], left_dirs[left]]
            intersection['right'] = [candidates[right], -1 * left_dirs[left]]

        return intersection

    def _get_track(self,
                   num_checkpoints,
                   track_rad=900 / SCALE,
                   x_bias=0,
                   y_bias=0):

        #num_checkpoints = 12

        # Create checkpoints
        checkpoints = []
        for c in range(num_checkpoints):
            alpha = 2 * math.pi * c / num_checkpoints + self.np_random.uniform(
                0, 2 * math.pi * 1 / num_checkpoints)
            rad = self.np_random.uniform(track_rad / 3, track_rad)
            if c == 0:
                alpha = 0
                rad = 1.5 * track_rad
            if c == num_checkpoints - 1:
                alpha = 2 * math.pi * c / num_checkpoints
                self.start_alpha = 2 * math.pi * (-0.5) / num_checkpoints
                rad = 1.5 * track_rad
            checkpoints.append((alpha, rad * math.cos(alpha),
                                rad * math.sin(alpha)))

        #print "\n".join(str(h) for h in checkpoints)
        #self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * track_rad, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while 1:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i %
                                                             len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0: break
                if not failed: break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3: beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3: beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4: break
            no_freeze -= 1
            if no_freeze == 0: break
        #print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0: return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[
                i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose > 0:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2,
                                                                  i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        track = [[a, b, x + x_bias * 2, y + y_bias * 2]
                 for a, b, x, y in track]
        track = [[track[i - 1], track[i]] for i in range(len(track))]
        return track

    def _get_possible_candidates_for_obstacles(self):
        return list(range(len(self.track)))

    def _create_obstacles(self):
        # Get random tile, with replacement
        # Create obstacle (red rectangle of random width and position in tile)
        #tiles_idx = np.random.choice(range(len(self.track)), self.num_obstacles, replace=False)

        possible_candidates = self._get_possible_candidates_for_obstacles()
        num_obstacles = int(self.num_obstacles * len(possible_candidates))
        if self.verbose > 0:
            print("num_obstacles: ", num_obstacles)

        # This loop removes any tile close to an
        # intersection from posible candidates
        for inter_tile in np.where(self.info['intersection_id'] != -1)[0]:
            track_len = len(self.tracks[self.info[inter_tile]['track']])
            compl_len = (self.info['track'] <
                         self.info[inter_tile]['track']).sum()

            idx_relative = self._to_relative(inter_tile)

            from_tile = -TILES_IN_SCREEN // 2
            to_tile = +TILES_IN_SCREEN // 2

            if self.info[inter_tile]['start']:
                from_tile = 0
            if self.info[inter_tile]['end']:
                to_tile = 1

            unvalid_candidates = []
            for i in range(from_tile, to_tile):
                unvalid_candidates.append((
                    (idx_relative + i) % track_len) + compl_len)

            possible_candidates = [
                item for item in possible_candidates
                if item not in unvalid_candidates
            ]

            if len(possible_candidates) == 0: break

        # This loop gets a list of tiles where the obstacles will be place
        # the osbtacles can not be near each other, at least +/-TILES_IN_SCREEN
        obstacle_tiles_ids = []
        if len(possible_candidates) > 0:
            for _ in range(num_obstacles):
                idx = np.random.choice(
                    possible_candidates, 1, replace=False)[0]
                obstacle_tiles_ids.append(idx)

                track_len = len(self.tracks[self.info[idx]['track']])
                compl_len = (self.info['track'] <
                             self.info[idx]['track']).sum()

                idx_relative = self._to_relative(idx)

                unvalid_candidates = []
                for i in range(-TILES_IN_SCREEN // 4, +TILES_IN_SCREEN // 4):
                    unvalid_candidates.append((
                        (idx_relative + i) % track_len) + compl_len)

                possible_candidates = [item for item in possible_candidates\
                        if item not in unvalid_candidates]

                if len(possible_candidates) == 0: break

        # This creates the objects for the obstacles
        for count, idx in enumerate(obstacle_tiles_ids):
            if self.random_obstacle_x_position:
                alpha, beta, x, y = self._get_rnd_position_inside_lane(idx)
            else:
                alpha, beta, x, y = self._get_rnd_position_inside_lane(
                    idx, border=False, discrete=True)

            if self.random_obstacle_shape:
                width = max(abs(np.random.normal(1)), 0.1) * TRACK_WIDTH / 4

                p1 = (x - width * math.cos(beta) +
                      TRACK_DETAIL_STEP / 2 * math.sin(beta),
                      y - width * math.sin(beta) -
                      TRACK_DETAIL_STEP / 2 * math.cos(beta))
                p2 = (x + width * math.cos(beta) +
                      TRACK_DETAIL_STEP / 2 * math.sin(beta),
                      y + width * math.sin(beta) -
                      TRACK_DETAIL_STEP / 2 * math.cos(beta))
                p3 = (x + width * math.cos(beta) -
                      TRACK_DETAIL_STEP / 2 * math.sin(beta),
                      y + width * math.sin(beta) +
                      TRACK_DETAIL_STEP / 2 * math.cos(beta))
                p4 = (x - width * math.cos(beta) -
                      TRACK_DETAIL_STEP / 2 * math.sin(beta),
                      y - width * math.sin(beta) +
                      TRACK_DETAIL_STEP / 2 * math.cos(beta))
            else:
                width = TRACK_WIDTH / 2

                p1 = (x + TRACK_DETAIL_STEP / 2 * math.sin(beta),
                      y - TRACK_DETAIL_STEP / 2 * math.cos(beta))
                p2 = (x + 2 * width * math.cos(beta) +
                      TRACK_DETAIL_STEP / 2 * math.sin(beta),
                      y + 2 * width * math.sin(beta) -
                      TRACK_DETAIL_STEP / 2 * math.cos(beta))
                p3 = (x + 2 * width * math.cos(beta) -
                      TRACK_DETAIL_STEP / 2 * math.sin(beta),
                      y + 2 * width * math.sin(beta) +
                      TRACK_DETAIL_STEP / 2 * math.cos(beta))
                p4 = (x - TRACK_DETAIL_STEP / 2 * math.sin(beta),
                      y + TRACK_DETAIL_STEP / 2 * math.cos(beta))

            vertices = [p1, p2, p3, p4]

            # Add it to obstacles
            # Add it to poly_obstacles
            t = self.world.CreateStaticBody(
                fixtures=fixtureDef(shape=polygonShape(vertices=vertices)))
            t.userData = t
            t.color = [0.86, 0.08, 0.23]
            #c = 0.01*(count%3)
            #t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_friction = 1.0
            t.road_visited = True
            t.typename = OBSTACLE_NAME
            t.road_visited = False
            t.id = count
            t.tile_id = idx
            t.fixtures[0].sensor = True
            self.obstacles_poly.append((vertices, t.color))
            self.road.append(t)
            self.info[idx]['obstacles'] = True

    def _create_info(self):
        '''
        Creates the matrix with the information about the track points,
        whether they are at the end of the track, if they are intersections
        '''
        # Get if point is at the end
        info = np.zeros(
            (sum(len(t) for t in self.tracks)),
            dtype=[
                ('track', 'int'),
                ('end', 'bool'),
                ('begining', 'bool'),
                ('intersection', 'bool'),
                ('intersection_id', 'int'),
                ('t', 'bool'),
                ('x', 'bool'),
                ('start', 'bool'),
                ('used', 'bool'),
                ('angle', 'float16'),
                ('ang_class', 'float16'),
                ('lanes', np.ndarray),
                ('count_left', 'int'),
                ('count_right', 'int'),
                ('count_left_delay', 'int'),
                ('count_right_delay', 'int'),
                ('visited', bool),
                #('obstacles',np.ndarray)])
                ('obstacles', bool)
            ])

        info['ang_class'] = np.nan
        info['intersection_id'] = -1
        info['obstacles'] = False

        for i in range(len(info)):
            info[i]['lanes'] = [True, True]

        for i in range(1, len(self.tracks)):
            track = self.tracks[i]
            info[len(self.tracks[i - 1]):len(self.tracks[i - 1]) +
                 len(track)]['track'] = i  # This wont work for num_tracks > 2
            for j in range(len(track)):
                pos = j + len(self.tracks[i - 1])
                p = track[j]
                next_p = track[(j + 1) % len(track)]
                last_p = track[j - 1]
                if np.array_equal(p[1], next_p[0]) == False:
                    # it is at the end
                    info[pos]['end'] = True
                elif np.array_equal(p[0], last_p[1]) == False:
                    # it is at the start
                    info[pos]['start'] = True

        # Trying to get all intersections
        intersections = set()
        if self.num_tracks > 1:
            for pos, point1 in enumerate(self.tracks[0][:, 1, 2:]):
                d = np.linalg.norm(
                    self.track[len(self.tracks[0]):, 1, 2:] - point1, axis=1)
                if d.min() <= 2.05 * TRACK_WIDTH:
                    intersections.add(pos)

            intersections = list(intersections)
            intersections.sort()
            track_len = len(self.tracks[0])

            def backwards():
                me = intersections[-1]
                del intersections[-1]
                if len(intersections) == 0: return [me]
                else:
                    if (me - 1) % track_len == intersections[-1]:
                        return [me] + backwards()
                    else:
                        return [me]

            def forward():
                me = intersections[0]
                del intersections[0]
                if len(intersections) == 0: return [me]
                else:
                    if (me + 1) % track_len == intersections[0]:
                        return [me] + forward()
                    else:
                        return [me]

            groups = []
            tmp_lst = []
            while len(intersections) != 0:
                me = intersections[0]
                tmp_lst = tmp_lst + backwards()
                if len(intersections) != 0:
                    if (me - 1) % track_len == intersections[-1]:
                        tmp_lst = tmp_lst + forward()

                groups.append(tmp_lst)
                tmp_lst = []

            for group in groups:
                min_dist_idx = None
                min_dist = 1e10
                for idx in group:
                    d = np.linalg.norm(
                        self.track[track_len:, 1, 2:] - self.track[idx, 1, 2:],
                        axis=1)
                    if d.min() < min_dist:
                        min_dist = d.min()
                        min_dist_idx = idx

                if min_dist <= TRACK_WIDTH:
                    intersections.append(min_dist_idx)

            info['intersection'][list(intersections)] = True

            # Classifying intersections
            for idx in intersections:
                point = self.track[idx, 1, 2:]
                d = np.linalg.norm(self.track[:, 1, 2:] - point, axis=1)
                argmin = d[info['track'] != 0].argmin()
                filt = np.where(d < TRACK_WIDTH * 2.5)

                # TODO ignore intersections with angles of pi/2

                if info[filt]['start'].sum() - info[filt]['end'].sum() != 0:
                    info[idx]['t'] = True
                    info[argmin + track_len]['t'] = True
                else:
                    # the sum can be zero because second tracks are not cutted in case of x
                    info[idx]['x'] = True
                    info[argmin + track_len]['x'] = True

        # Getting angles of curves
        max_idxs = []
        self.track[:, 0, 1] = np.mod(self.track[:, 0, 1], 2 * math.pi)
        self.track[:, 1, 1] = np.mod(self.track[:, 1, 1], 2 * math.pi)
        for num_track in range(self.num_tracks):

            track = self.tracks[num_track]
            angles = track[:, 0, 1] - track[:, 1, 1]
            inters = np.logical_or(info[info['track'] == num_track]['t'],
                                   info[info['track'] == num_track]['x'])

            track_len_compl = (info['track'] < num_track).sum()
            track_len = len(track)

            while np.abs(angles).max() != 0.0:
                max_rel_idx = np.abs(angles).argmax()

                rel_idxs = [
                    (max_rel_idx + j) % track_len
                    for j in range(-NUM_TILES_FOR_AVG, NUM_TILES_FOR_AVG)
                ]
                idxs_safety = [(max_rel_idx + j) % track_len
                               for j in range(-NUM_TILES_FOR_AVG *
                                              2, NUM_TILES_FOR_AVG * 2)]

                if (inters[idxs_safety] == True).sum() == 0:
                    max_idxs.append(max_rel_idx + track_len_compl)
                    angles[rel_idxs] = 0.0
                else:
                    angles[max_rel_idx] = 0.0

        info['angle'][
            max_idxs] = self.track[max_idxs, 0, 1] - self.track[max_idxs, 1, 1]

        ######### populating intersection_id
        intersection_dict = {}

        # Remove keys which are to close
        intersection_keys = np.where(info['intersection'])[0]
        intersection_vals = np.where((info['x']) | (info['t']))[0]

        for val in intersection_vals:
            tmp = self.track[intersection_keys][:, 1, 2:]
            elm = self.track[val, 1, 2:]
            d = np.linalg.norm(tmp - elm, axis=1)
            if d.min() > TRACK_WIDTH * 2:
                if self.verbose > 0:
                    print("the closest intersection is too far away")
            else:
                k = intersection_keys[d.argmin()]

                if k in intersection_dict.keys(): pass
                else:
                    intersection_dict[k] = []

                intersection_dict[k].append(val)

        self.intersection_dict = intersection_dict

        for k, v in self.intersection_dict.items():
            info['intersection_id'][[k] + v] = k
        del self.intersection_dict
        ##############################################

        self.info = info

    def _set_lanes(self):
        if self.num_lanes_changes > 0 and self.num_lanes > 1:
            rm_lane = 0  # 1 remove lane, 0 keep lane
            lane = 0  # Which lane will be removed
            changes = np.sort(
                self.np_random.randint(0, len(self.track),
                                       self.num_lanes_changes))

            # check in changes work
            # There must be no change at least 50 pos before and end and after a start
            changes_bad = []
            for pos, idx in enumerate(changes):
                start_from = sum(self.info['track'] < self.info[idx]['track'])
                until = sum(self.info['track'] == self.info[idx]['track'])
                changes_in_track = np.subtract(changes, start_from)
                changes_in_track = changes_in_track[(changes_in_track < until)
                                                    * (changes_in_track > 0)]
                idx_relative = idx - start_from

                if sum(((changes_in_track - idx) > 0) * (
                    (changes_in_track - idx) <
                        10)) > 0:  # TODO wont work when at end of track
                    changes_bad.append(idx)
                    next

                track_info = self.info[self.info['track'] == self.info[idx]
                                       ['track']]
                for i in range(50 + 1):
                    if track_info[(idx_relative + i) % len(track_info)][
                            'end'] or track_info[idx_relative - i]['start']:
                        changes_bad.append(idx)
                        break

            if len(changes_bad) > 0:
                changes = np.setdiff1d(changes, changes_bad)

            counter = 0  # in order to avoid more than max number of single lanes tiles
            for i, point in enumerate(self.track):
                change = True if i in changes else False
                rm_lane = (rm_lane + change) % 2

                if change and rm_lane == 1:  # if it is time to change and the turn is to remove lane
                    lane = np.random.randint(0, 2, 1)[0]

                if rm_lane:
                    self.info[i]['lanes'][lane] = False
                    counter += 1
                else:
                    counter = 0

                # Change if end/inter of or if change prob
                if self.info[i]['end'] or self.info[i][
                        'start'] or counter > self.max_single_lane:
                    rm_lane = 0

            # Avoiding any change of lanes in last and beginning part of a track
            for num_track in range(self.num_tracks):
                for lane in range(self.num_lanes):
                    for i in range(10):
                        i %= len(self.tracks[num_track])
                        self.info[self.info['track'] ==
                                  num_track][+i]['lanes'][lane] = True
                        self.info[self.info['track'] ==
                                  num_track][-i]['lanes'][lane] = True

    def _remove_unfinished_roads(self):
        n = 0
        to_remove = set()
        # The problem only appears in track1
        while n < len(self.tracks[0]):
            prev_tile = self.tracks[0][n - 2]
            tile = self.tracks[0][n - 1]
            next_tile = self.tracks[0][n]

            if any(tile[0] != prev_tile[1]) or any(tile[1] != next_tile[0]):
                to_remove.update(n)
                n -= 1
            else:
                n += 1
        self.tracks[0] = np.delete(self.tracks[0], list(to_remove), axis=0)

        if len(self.tracks[1]) < 5:
            self.tracks[1] = np.delete(
                self.tracks[1], range(len(self.tracks[1])), axis=1)

    def _choice_random_track_from_file(self):
        idx = np.random.choice(self.tracks_df.index)
        return idx

    def _generate_track(self):
        if self.load_tracks_from is not None:
            idx = self._choice_random_track_from_file()
            try:
                dic = pickle.load(
                    open(self.load_tracks_from + '/' + str(idx) + ".pkl",
                         'rb'))
            except Exception as e:
                print("######## Error ########")
                print("error loading the track", str(idx))
                print(e)
                return False
            else:
                self.track = dic['track']
                self.tracks = dic['tracks']
                self.info = dic['info']
                self.obstacle_contacts = np.zeros((len(self.obstacles_poly)),
                                                  dtype=[('count', int),
                                                         ('count_delay', int),
                                                         ('visited', bool)])

                self.info[[
                    'count_left', 'count_right', 'count_right_delay',
                    'count_left_delay'
                ]] = 0
                self.info['visited'] = False

                return True
        else:
            tracks = []
            cp = 12
            for _ in range(self.num_tracks):
                # The following variables allow for more complex tracks but, it is also
                # harder to controll their properties and correct behaviour
                track = self._get_track(int(
                    cp * (1**_)))  #,x_bias=-40*_,y_bias=40*_)
                if not track or len(track) == 0: return track
                track = np.array(track)
                if _ > 0 and False:
                    # adding rotation to decrease number of overlaps
                    theta = np.random.uniform() * 2 * np.pi
                    R = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
                    track[:, 0, 2:] = (R @ track[:, 0, 2:].T).T
                    track[:, 1, 2:] = (R @ track[:, 1, 2:].T).T
                    track[:, :2] += theta
                tracks.append(track)

            self.tracks = tracks
            if self.num_tracks > 1:
                if self._remove_roads() == False: return False
                self._remove_unfinished_roads()

            if self.tracks[0].size <= 5:
                return False
            if self.num_tracks > 1:
                if self.tracks[1].size <= 5:
                    return False
                if self.tracks[0].shape[1:] != self.tracks[1].shape[1:]:
                    return False

                self.track = np.concatenate(self.tracks)
            else:
                self.track = np.array(self.tracks[0])

            self._create_info()
            # Avoid lonely tiles at the begining of track
            if self.info[0]['intersection_id'] != -1: return False

            self._set_lanes()

    def _create_track(self):

        Ok = self._generate_track()
        if Ok is False:
            return False

        # Red-white border on hard turns
        borders = []
        if True:
            for track in self.tracks:
                border = [False] * len(track)
                for i in range(1, len(track)):
                    good = True
                    oneside = 0
                    for neg in range(BORDER_MIN_COUNT):
                        beta1 = track[i - neg][1][1]
                        beta2 = track[i - neg][0][1]
                        good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                        oneside += np.sign(beta1 - beta2)
                    good &= abs(oneside) == BORDER_MIN_COUNT
                    border[i] = good
                for i in range(len(track)):
                    for neg in range(BORDER_MIN_COUNT):
                        # TODO ERROR, sometimes list index out of range
                        border[i - neg] |= border[i]
                borders.append(border)

            # Creating borders for printing
            pos = 0
            for j in range(self.num_tracks):
                track = self.tracks[j]
                border = borders[j]
                for i in range(len(track)):
                    alpha1, beta1, x1, y1 = track[i][1]
                    alpha2, beta2, x2, y2 = track[i][0]
                    if border[i]:
                        side = np.sign(beta2 - beta1)

                        c = 1

                        # Addapting border to appear at the right widht when there are different number of lanes
                        if self.num_lanes > 1:
                            if side == -1 and self.info[pos]['lanes'][
                                    0] == False:
                                c = 0
                            if side == +1 and self.info[pos]['lanes'][
                                    1] == False:
                                c = 0

                        b1_l = (x1 + side * TRACK_WIDTH * c * math.cos(beta1),
                                y1 + side * TRACK_WIDTH * c * math.sin(beta1))
                        b1_r = (x1 + side *
                                (TRACK_WIDTH * c + BORDER) * math.cos(beta1),
                                y1 + side *
                                (TRACK_WIDTH * c + BORDER) * math.sin(beta1))
                        b2_l = (x2 + side * TRACK_WIDTH * c * math.cos(beta2),
                                y2 + side * TRACK_WIDTH * c * math.sin(beta2))
                        b2_r = (x2 + side *
                                (TRACK_WIDTH * c + BORDER) * math.cos(beta2),
                                y2 + side *
                                (TRACK_WIDTH * c + BORDER) * math.sin(beta2))
                        self.border_poly.append(
                            ([b1_l, b1_r, b2_r, b2_l],
                             (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
                    pos += 1

        # Create tiles
        for j in range(len(self.track)):
            obstacle = np.random.binomial(1, 0)
            alpha1, beta1, x1, y1 = self.track[j][1]
            alpha2, beta2, x2, y2 = self.track[j][0]

            # drawing angles of old config, the
            # black line is the angle (NOT WORKING)
            if SHOW_BETA_PI_ANGLE:
                if self.track_lanes == None: self.track_lanes = []
                p1x = x1 + np.cos(beta1) * 0.2
                p1y = y1 + np.sin(beta1) * 0.2
                p2x = x1 + np.cos(beta1) * 0.2 + np.cos(beta1 + np.pi / 2) * 2
                p2y = y1 + np.sin(beta1) * 0.2 + np.sin(beta1 + np.pi / 2) * 2
                p3x = x1 - np.cos(beta1) * 0.2 + np.cos(beta1 + np.pi / 2) * 2
                p3y = y1 - np.sin(beta1) * 0.2 + np.sin(beta1 + np.pi / 2) * 2
                p4x = x1 - np.cos(beta1) * 0.2
                p4y = y1 - np.sin(beta1) * 0.2
                self.track_lanes.append([[p1x, p1y], [p2x, p2y], [p3x, p3y],
                                         [p4x, p4y]])

            for lane in range(self.num_lanes):
                if self.info[j]['lanes'][lane]:

                    joint = False  # to differentiate joints from normal tiles

                    r = 1 - ((lane + 1) % self.num_lanes)
                    l = 1 - ((lane + 2) % self.num_lanes)

                    # Get if it is the first or last
                    first = False  # first of lane
                    last = False  # last tile of line

                    if self.info[j]['end'] == False and self.info[j][
                            'start'] == False:

                        # Getting if first tile of lane
                        # if last tile was from the same lane
                        info_track = self.info[self.info['track'] == self.
                                               info[j]['track']]
                        j_relative = j - sum(
                            self.info['track'] < self.info[j]['track'])

                        if info_track[
                                j_relative -
                                1]['track'] == info_track[j_relative]['track']:
                            # If last tile didnt exist
                            if info_track[j_relative -
                                          1]['lanes'][lane] == False:
                                first = True
                        if info_track[(j_relative + 1) % len(info_track)][
                                'track'] == info_track[j_relative]['track']:
                            # If last tile didnt exist
                            if info_track[(j_relative + 1) % len(
                                    info_track)]['lanes'][lane] == False:
                                last = True

                    road1_l = (
                        x1 - (1 - last) * l * TRACK_WIDTH * math.cos(beta1),
                        y1 - (1 - last) * l * TRACK_WIDTH * math.sin(beta1))
                    road1_r = (
                        x1 + (1 - last) * r * TRACK_WIDTH * math.cos(beta1),
                        y1 + (1 - last) * r * TRACK_WIDTH * math.sin(beta1))
                    road2_l = (
                        x2 - (1 - first) * l * TRACK_WIDTH * math.cos(beta2),
                        y2 - (1 - first) * l * TRACK_WIDTH * math.sin(beta2))
                    road2_r = (
                        x2 + (1 - first) * r * TRACK_WIDTH * math.cos(beta2),
                        y2 + (1 - first) * r * TRACK_WIDTH * math.sin(beta2))

                    vertices = [road1_l, road1_r, road2_r, road2_l]

                    if self.info[j]['end'] == True or self.info[j][
                            'start'] == True:

                        points = []  # to store the new points
                        p3 = [
                        ]  # in order to save all points 3 to create joints
                        for i in [0, 1]:  # because there are two point to do
                            # Get the closest point to a line make by the continuing trend of the original road points, the points will be the points
                            # under a radius r from line to avoid taking points far away in the other extreme of the track
                            # Remember the distance from a point p3 to a line p1,p2 is d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
                            # p1=(x1,y1)+sin/cos, p2=(x2,y2)+sin/cos, p3=points in poly
                            if self.info[j]['end']:
                                p1 = road1_l if i == 0 else road1_r
                                p2 = road2_l if i == 0 else road2_r
                            else:
                                p1 = road1_l if i == 0 else road1_r
                                p2 = road2_l if i == 0 else road2_r

                            if len(p3) == 0:
                                max_idx = sum(
                                    sum(
                                        self.info[self.info['track'] == 0]
                                        ['lanes'], [])
                                )  # this will work because only seconday tracks have ends
                                p3_org = sum(
                                    [x[0] for x in self.road_poly[:max_idx]],
                                    [])
                                # filter p3 by distance to p1 < TRACK_WIDTH*2
                                distance = TRACK_WIDTH * 2
                                not_too_close = np.where(
                                    np.linalg.norm(
                                        np.subtract(p3_org, p1), axis=1) >=
                                    TRACK_WIDTH / 3)[0]
                                while len(p3) == 0 and distance < PLAYFIELD:
                                    close = np.where(
                                        np.linalg.norm(
                                            np.subtract(p3_org, p1), axis=1) <=
                                        distance)[0]
                                    p3 = [
                                        p3_org[i] for i in np.intersect1d(
                                            close, not_too_close)
                                    ]
                                    distance += TRACK_WIDTH

                            if len(p3) == 0:
                                raise RuntimeError('p3 lenght is zero')

                            d = (np.cross(
                                np.subtract(p2, p1), np.subtract(
                                    p1, p3)))**2 / np.linalg.norm(
                                        np.subtract(p2, p1))
                            points.append(p3[d.argmin()])

                        if self.info[j]['start']:
                            vertices = [points[0], points[1], road1_r, road1_l]
                        else:
                            vertices = [road2_r, road2_l, points[0], points[1]]
                        joint = True

                    test_set = set([tuple(p) for p in vertices])
                    if len(test_set) >= 3:
                        # TODO CHECK IF THIS AVOID THE ERROR OF ASSERTION COUNT >= 3
                        # TODO remove this try and find a way of really catching the errer
                        #try:
                        self.fd_tile.shape.vertices = vertices
                        t = self.world.CreateStaticBody(fixtures=self.fd_tile)
                        #except AssertionError as e:
                        #print(str(e))
                        #print(vertices)
                        #return False
                        t.userData = t
                        i = 0
                        # changing the following i for j achives different colors when visited tiles
                        c = 0.01 * (i % 3)
                        if joint and SHOW_JOINTS:
                            t.color = [1, 1, 1]
                        else:
                            #t.color = [ROAD_COLOR[0], ROAD_COLOR[1], ROAD_COLOR[2]]
                            t.color = [
                                ROAD_COLOR[0] + c, ROAD_COLOR[1] + c,
                                ROAD_COLOR[2] + c
                            ]
                        t.road_visited = False
                        t.typename = TILE_NAME
                        t.road_friction = 1.0
                        t.id = j
                        t.lane = lane
                        t.fixtures[0].sensor = True
                        self.road_poly.append((vertices, t.color, t.id,
                                               t.lane))
                        self.road.append(t)
                    else:
                        print("saved from error")

        self._create_obstacles()

        self.obstacle_contacts = np.zeros((len(self.obstacles_poly)),
                                          dtype=[('count', int),
                                                 ('count_delay', int),
                                                 ('visited', bool)])

        return True

    def reset(self):
        '''
        car_position [angle float, x float, y float]
                     Position of the car
                     Default: first tile of principal track
        '''
        self._destroy()
        self.reward = 0.0
        self.full_reward = 0.0
        self.highest_reward = 0.0
        self.last_touch_with_track = 0.0
        self.prev_reward = 0.0
        self.prev_out_of_track = False
        self.tile_visited_count = 0
        self.t = 0.0
        self._current_nodes = {}
        self._next_nodes = []
        self.road_poly = []
        self.border_poly = []
        self.obstacles_poly = []
        self.track = []
        self.tracks = []
        self.info = []
        self.road = []
        self.track_lanes = None
        self.human_render = False
        self.state = np.zeros(self.observation_space.shape)
        self._steps_in_episode = 0

        while True:
            success = self._create_track()

            if success:
                if self._position_car_on_reset() is not False:
                    break

            if self.verbose > 0:
                print(
                    "retry to generate track (normal if there are not many of this messages)"
                )

        # there are 20 frames of noise at the begining
        #for _ in range(self.frames_per_state+20):
        #    obs = self.step(None)[0]
        obs = self.step(None)[0]
        return obs

    def _position_car_on_reset(self):
        """
        This function takes care of placing the car in a position
        at every reset call. This function should be modify to have
        the desired behaviour of where the car appears, do not
        re-spawn the car after the reset function has been called
        """
        self.place_agent(self.get_rnd_point_in_track())

    def _update_state(self, new_frame):
        if self.frames_per_state > 1:
            self.state[:, :, -1] = new_frame
            self.state = self.state[:, :, self._update_index]
        else:
            self.state = new_frame

    def _transform_action(self, action):
        if self.discretize_actions == "soft":
            raise NotImplementedError
        elif self.discretize_actions == "hard":
            # ("NOTHING", "LEFT", "RIGHT", "ACCELERATE", "BREAK")
            # angle, gas, break
            if action == 0: action = [0, 0, 0.0]  # Nothing
            if action == 1: action = [-1, 0, 0.0]  # Left
            if action == 2: action = [+1, 0, 0.0]  # Right
            if action == 3: action = [0, +1, 0.0]  # Accelerate
            if action == 4: action = [0, 0, 0.8]  # break

        return action

    def step(self, action):
        action = self._transform_action(action)

        if action is not None:
            self._steps_in_episode += 1
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        # self.state = self.render("state_pixels") # Old code, only one frame
        self._update_state(self.render("state_pixels"))

        step_reward = 0
        full_step_reward = 0
        done = False

        step_reward, full_step_reward, done, info = self.reward_fn(self)

        self.car.fuel_spent = 0.0

        self.reward += step_reward
        self.full_reward += full_step_reward

        if self.auto_render:
            self.render()

        obs = self.state
        if self.observe_car_status:
            obs = {'rgb': self.state, 'car': self.get_car_status()}
        return obs, step_reward, done, info

    def _render_additional_objects(self):
        """
        This function has the only objective to be a way for the classes inhereting
        from this class to add its own obects to render, and avoiding overwriting
        this function
        """
        pass

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                'Score: 0000',
                font_size=40,
                x=20,
                y=WINDOW_H * 1.5 / 40.00,
                anchor_x='left',
                anchor_y='center',
                color=(255, 255, 255, 255))
            self.full_score_label = pyglet.text.Label(
                'Full Score: 0000',
                font_size=20,
                x=250,
                y=WINDOW_H * 1.5 / 40.00,
                anchor_x='left',
                anchor_y='center',
                color=(255, 255, 255, 255))
            self.speed_label = pyglet.text.Label(
                'Speed: 000',
                font_size=20,
                x=20,
                y=WINDOW_H * 3.7 / 40.00,
                anchor_x='left',
                anchor_y='center',
                color=(255, 255, 255, 255))
            self.angle_label = pyglet.text.Label(
                'Angle: 000',
                font_size=20,
                x=250,
                y=WINDOW_H * 3.7 / 40.00,
                anchor_x='left',
                anchor_y='center',
                color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

            self.viewer.window.on_key_press = self._key_press
            self.viewer.window.on_key_release = self._key_release
        if "t" not in self.__dict__: return  # reset() not called yet

        if self.animate_zoom:
            zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(
                self.t, 1)  # Animate zoom first second
        else:
            zoom = ZOOM * SCALE

        zoom_state = ZOOM * SCALE * STATE_W / WINDOW_W
        zoom_video = ZOOM * SCALE * VIDEO_W / WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        # The angle is the same as the car, not as the speed
        #if np.linalg.norm(vel) > 0.5:
        #    angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        if ZOOM_OUT:
            self.transform.set_translation(WINDOW_W / 2, WINDOW_H / 2)
            self.transform.set_rotation(0)
        else:
            self.transform.set_translation(
                WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) -
                                scroll_y * zoom * math.sin(angle)),
                WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) +
                                scroll_y * zoom * math.cos(angle)))
            self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window

        # To allow listening to keys during training
        win.switch_to()
        win.dispatch_events()
        if mode == "rgb_array" or mode == "state_pixels" or mode == "HD":
            win.clear()
            t = self.transform
            if mode == 'rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            elif mode == 'HD':
                VP_H = WINDOW_H
                VP_W = WINDOW_W
            else:
                VP_W = STATE_W
                VP_H = STATE_H
            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            if self.show_info_panel:
                self.render_indicators(
                    WINDOW_W, WINDOW_H)  # TODO: find why 2x needed, wtf
            self._render_additional_objects()
            image_data = pyglet.image.get_buffer_manager().get_color_buffer(
            ).get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]
            if self.grayscale and mode != "rgb_array":
                arr_bw = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
                arr = arr_bw

        if mode == "rgb_array" and not self.human_render:  # agent can call or not call env.render() itself when recording video.
            win.flip()

        if mode == 'human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)
            self._render_additional_objects()
            win.flip()

        self.viewer.onetime_geoms = []
        return arr

    def _key_press(self, k, mod):
        from pyglet.window import key
        if k == key.S:  # S from Show
            self.auto_render = not self.auto_render
        if k == key.T:  # T from Take screnshot
            self.screenshot()

        if self.key_press_fn is not None:
            self.key_press_fn(k, mod)

    def _key_release(self, k, mod):
        if self.key_release_fn is not None:
            self.key_release_fn(k, mod)

    def screenshot(self, dest="./", name=None, quality='low'):
        '''
        Saves the current state, quality 'low','medium' or 'high', low will save the
        current state if the quality is low, otherwise will save the current frame
        '''
        if quality == 'low':
            state = self.state
        elif quality == 'medium':
            state = self.render('rgb_array')
        else:
            state = self.render("HD")
        if state is not None:
            for f in range(self.frames_per_state):

                if self.frames_per_state == 1 or quality is not 'low':
                    frame_str = ""
                    frame = state
                else:
                    frame_str = "_frame%i" % f
                    frame = state[:, :, f]

                if self.grayscale:
                    frame = np.stack([frame, frame, frame], axis=-1)

                frame = frame.astype(np.uint8)
                im = Image.fromarray(frame)
                if name == None: name = "screenshot_%0.3f" % self.t
                im.save("%s/%s%s.jpeg" % (dest, name, frame_str))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _remove_roads(self):

        if self.num_tracks > 1:

            def _get_section(first, last, track):
                sec = []
                pos = 0
                found = False
                while 1:
                    point = track[pos % track.shape[0], :, 2:]
                    if np.linalg.norm(point[1] - first) <= TRACK_WIDTH / 2:
                        found = True
                    if found:
                        sec.append(point)
                        if np.linalg.norm(point[1] - last) <= TRACK_WIDTH / 2:
                            break
                    pos = pos + 1
                    if pos / track.shape[0] >= 2: break
                if sec == []: return False
                return np.array(sec)

            THRESHOLD = TRACK_WIDTH * 2

            track1 = np.array(self.tracks[0])
            track2 = np.array(self.tracks[1])

            points1 = track1[:, :, [2, 3]]
            points2 = track2[:, :, [2, 3]]

            inter2 = np.array([
                x for x in points2
                if (np.linalg.norm(points1[:, 1, :] - x[1:], axis=1) <=
                    TRACK_WIDTH / 3.5).sum() >= 1
            ])

            intersections = []
            for i in range(inter2.shape[0]):
                if np.array_equal(
                        inter2[i - 1, 1, :],
                        inter2[i, 0, :]) == False or np.array_equal(
                            inter2[i, 1, :],
                            inter2[((i + 1) % len(inter2)), 0, :]) == False:
                    intersections.append(inter2[i])
            intersections = np.array(intersections)

            # For each point in intersection
            # > get section of both roads
            # > For each point in section in second road
            # > > get min distance
            # > get max of distances
            # if max dist < threshold remove
            removed_idx = set()
            intersection_keys = []
            intersection_vals = []
            sec1_closer_to_center = None
            for i in range(intersections.shape[0]):
                _, first = intersections[i - 1]
                last, _ = intersections[i]

                sec1 = _get_section(first, last, track1)
                sec2 = _get_section(first, last, track2)

                sec1_distance_to_center = np.mean(
                    np.linalg.norm(sec1[2:], axis=1))
                sec2_distance_to_center = np.mean(
                    np.linalg.norm(sec2[2:], axis=1))

                if sec1 is not False and sec2 is not False:

                    remove = False
                    if sec1_distance_to_center > sec2_distance_to_center:
                        # sec1 is outside
                        if sec1_closer_to_center is False:
                            remove = True
                        else:
                            sec1_closer_to_center = False
                    else:
                        # sec1 is inside
                        if sec1_closer_to_center is True:
                            remove = True
                        else:
                            sec1_closer_to_center = True

                    if remove is False:
                        max_min_d = 0
                        remove = False
                        min_distances = []
                        for point in sec1[:, 1]:
                            dist = np.linalg.norm(
                                sec2[:, 1] - point, axis=1).min()
                            min_distances.append(dist)
                            #min_d = dist if max_min_d < dist else max_min_d

                        min_distances = np.array(min_distances)

                        # if the max minimal distance is too small
                        if min_distances.max() < THRESHOLD * 2:
                            remove = True
                            # if the middle tiles of segment are too close to main track
                        elif len(min_distances) > 25 and (
                                min_distances[10:-10].min() < TRACK_WIDTH * 3):
                            remove = True
                        # if the segment is smaller than MIN_SEGMENT_LENGHT
                        elif len(min_distances) < MIN_SEGMENT_LENGHT:
                            remove = True
                        # if there are more than 50 tiles very close to main track
                        elif len(min_distances) > 50 and (
                                min_distances < TRACK_WIDTH * 2).sum() > 50:
                            remove = True

                    # Removing tiles
                    if remove:
                        for point in sec2:
                            idx = np.all(
                                track2[:, :, [2, 3]] == point, axis=(1, 2))
                            removed_idx.update(np.where(idx)[0])
                    else:
                        key = np.where(
                            np.all(
                                track1[:, :, [2, 3]] == sec1[0], axis=(1,
                                                                       2)))[0]
                        val = np.where(
                                np.all(track2[:,:,[2,3]] == sec2[0], axis=(1,2)))[0]\
                                        + len(track1)
                        intersection_keys.append(key[0])
                        intersection_vals.append(val[0])

                        key = np.where(
                            np.all(
                                track1[:, :, [2, 3]] == sec1[-1], axis=(1,
                                                                        2)))[0]
                        val = np.where(
                                np.all(track2[:,:,[2,3]] == sec2[-1], axis=(1,2)))[0]\
                                        + len(track1)
                        intersection_keys.append(key[0])
                        intersection_vals.append(val[0])

            track2 = np.delete(
                track2, list(removed_idx),
                axis=0)  # efficient way to delete them from np.array

            self.intersections = intersections

            if len(track1) == 0 or len(track2) == 0:
                return False

            self.tracks[0] = track1
            self.tracks[1] = track2

            return True

    def _render_tiles(self):
        '''
        Can only be called inside a glBegin
        '''
        # drawing road old way
        for poly, color, id, lane in self.road_poly:
            if SHOW_NEXT_N_TILES > 0 and id in self._trail_nodes and lane in self._trail_nodes[
                    id]:
                #if (hasattr(self,'_objective') and self._objective == id) or id in self._current_nodes.keys() \
                #or (hasattr(self,'_neg_objectives') and id in self._neg_objectives):
                #if id in self.predictions_id:
                color = [c / 2 for c in color]

            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)

    def _render_obstacles(self):
        '''
        Can only be called inside a glBegin
        '''
        # drawing road old way
        for poly, color in self.obstacles_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)

        # Ploting axis
        if SHOW_AXIS:
            # x-axis
            gl.glColor4f(0, 0, 0, 1)
            gl.glVertex3f(-PLAYFIELD, 2, 0)
            gl.glVertex3f(+PLAYFIELD, 2, 0)
            gl.glVertex3f(+PLAYFIELD, -2, 0)
            gl.glVertex3f(-PLAYFIELD, -2, 0)

            # y-axis
            gl.glVertex3f(+2, -PLAYFIELD, 0)
            gl.glVertex3f(+2, +PLAYFIELD, 0)
            gl.glVertex3f(-2, +PLAYFIELD, 0)
            gl.glVertex3f(-2, -PLAYFIELD, 0)

        self._update_predictions()
        self._render_tiles()
        self._render_obstacles()
        self._render_road_lines()
        self.render_debug_clues()

        gl.glEnd()

    def _render_road_lines(self):
        if SHOW_BETA_PI_ANGLE:
            for block in self.track_lanes:
                gl.glColor4f(1, 1, 1, 0.8)
                for x, y in block:
                    gl.glVertex3f(x, y, 0)

    def render_debug_clues(self):

        if SHOW_ENDS_OF_TRACKS:
            for x, y in self.track[self.info['end']][:, 1, 2:]:
                gl.glColor4f(1, 0, 0, 1)
                gl.glVertex3f(x + 2, y + 2, 0)
                gl.glVertex3f(x - 2, y + 2, 0)
                gl.glVertex3f(x - 2, y - 2, 0)
                gl.glVertex3f(x + 2, y - 2, 0)

        if SHOW_START_OF_TRACKS:
            for x, y in self.track[self.info['start']][:, 1, 2:]:
                gl.glColor4f(0, 1, 0, 1)
                gl.glVertex3f(x + 2, y + 2, 0)
                gl.glVertex3f(x - 2, y + 2, 0)
                gl.glVertex3f(x - 2, y - 2, 0)
                gl.glVertex3f(x + 2, y - 2, 0)

        if SHOW_INTERSECTION_POINTS:
            for x, y in self.track[self.info['intersection']][:, 1, 2:]:
                gl.glColor4f(1, 1, 0, 1)
                gl.glVertex3f(x + 1, y + 1, 0)
                gl.glVertex3f(x - 1, y + 1, 0)
                gl.glVertex3f(x - 1, y - 1, 0)
                gl.glVertex3f(x + 1, y - 1, 0)

        if SHOW_GROUP_INTERSECTIONS:
            ids = set(self.info['intersection_id'])
            ids.remove(-1)

            for id in ids:
                np.random.seed(id)
                r = np.random.uniform(size=3)
                for elem in self.track[self.info['intersection_id'] == id]:
                    x, y = elem[1, 2:]
                    gl.glColor4f(r[0], r[1], r[2], 1)
                    gl.glVertex3f(x + 1, y + 1, 0)
                    gl.glVertex3f(x - 1, y + 1, 0)
                    gl.glVertex3f(x - 1, y - 1, 0)
                    gl.glVertex3f(x + 1, y - 1, 0)

        if SHOW_XT_JUNCTIONS:
            for x, y in self.track[self.info['t']][:, 1, 2:]:
                gl.glColor4f(0, 0.4, 0, 1)
                gl.glVertex3f(x + 1, y + 1, 0)
                gl.glVertex3f(x - 1, y + 1, 0)
                gl.glVertex3f(x - 1, y - 1, 0)
                gl.glVertex3f(x + 1, y - 1, 0)
            for x, y in self.track[self.info['x']][:, 1, 2:]:
                gl.glColor4f(6, 0.8, 0.18, 1)
                gl.glVertex3f(x + 1, y + 1, 0)
                gl.glVertex3f(x - 1, y + 1, 0)
                gl.glVertex3f(x - 1, y - 1, 0)
                gl.glVertex3f(x + 1, y - 1, 0)

        if SHOW_TURNS:
            for x, y in self.track[np.abs(
                    self.info['angle']).argsort()[-10:]][:, 1, 2:]:
                gl.glColor4f(1, 0, 0, 1)
                gl.glVertex3f(x + 1, y + 1, 0)
                gl.glVertex3f(x - 1, y + 1, 0)
                gl.glVertex3f(x - 1, y - 1, 0)
                gl.glVertex3f(x + 1, y - 1, 0)

    def get_speed(self):
        return np.sqrt(
            np.square(self.car.hull.linearVelocity[0]) +
            np.square(self.car.hull.linearVelocity[1]))

    def get_car_status(self):
        abs_sensors, wheel_angles = [], []
        for i in range(4):
            abs_sensors.append(self.car.wheels[i].omega)
            wheel_angles.append(self.car.wheels[i].joint.angle)
        abs_sensors = np.array(abs_sensors, dtype=np.float32)
        wheel_angles = np.array(wheel_angles, dtype=np.float32)
        speed = np.array(self.get_speed(), dtype=np.float32)
        angular_vel = np.array(self.car.hull.angularVelocity, dtype=np.float32)
        tiles_remaining = np.array(
            [(len(self.track) - self.info['visited'].sum()) / len(self.track)],
            dtype=np.float32)
        return np.r_[abs_sensors, wheel_angles, speed, angular_vel,
                     tiles_remaining]

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W / 40.0
        h = H / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h, 0)
            gl.glVertex3f((place + 0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)

        true_speed = self.get_speed()
        vertical_ind(21, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(22, 0.01 * self.car.wheels[0].omega,
                     (0.0, 0, 1))  # ABS sensors
        vertical_ind(23, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(24, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(25, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(30, -5.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(36, -0.4 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "Score: %04i" % self.reward
        self.full_score_label.text = "Full Score: %04i" % self.full_reward
        self.speed_label.text = "Speed: %0.2f" % np.linalg.norm(
            self.car.hull.linearVelocity)
        self.angle_label.text = "Angle: %0.2f" % self.car.wheels[0].joint.angle
        self.score_label.draw()
        #self.full_score_label.draw()
        #self.speed_label.draw()
        #self.angle_label.draw()

    def get_rnd_point_in_track(self, border=True):
        '''
        returns a random point in the track with the angle equal
        to the tile of the track, the x position can be randomly
        in the x (relative) axis of the tile, border=True make
        sure the x position is enough to make the car fit in
        the track, otherwise the point can be in the extreme
        of the track and two wheels will be outside the track
        -----
        Returns: [beta, x, y]
        '''
        direction = 1 if np.random.uniform() > 0.5 else -1
        idx = self.np_random.randint(0, len(self.track))
        _, beta, x, y = self._get_rnd_position_inside_lane(
            idx, border=border, direction=direction)
        return [beta, x, y]

    def _get_rnd_position_inside_lane(self,
                                      idx,
                                      border=True,
                                      direction=1,
                                      discrete=False):
        '''
        idx of tile
        direction: -1 if want it going in the opposite direction,
        discrete=True means the random position is either 0 or 1, i.e. in the
        beginning or the end of the position in the x-relative coordinate
        '''
        h = np.random.uniform(0, 1)
        h = 1 if h >= 0.5 else 0
        return self._get_position_inside_lane(
            idx, h, border=border, direction=direction, discrete=discrete)

    def _get_position_inside_lane(self,
                                  idx,
                                  x_pos,
                                  border=True,
                                  direction=1,
                                  discrete=False):
        '''
        x_pos in [0,1] meaning the position in the x axis relative to the direction
        '''
        alpha, beta, x, y = self.track[idx, 1, :]
        if direction == -1:
            alpha += np.pi
            beta += np.pi
        from_val, to_val = self._get_extremes_of_position(idx, border)
        if discrete:
            # it is 1-border in becase -TRACK_WIDTH when border=True
            # makes h always equal to -3.3333 because it is taking
            # into account the border twice
            x_pos = from_val * x_pos + (1 - x_pos) * (to_val - TRACK_WIDTH *
                                                      (1 - border))
        else:
            x_pos = from_val * x_pos + (1 - x_pos) * to_val
        x += x_pos * math.cos(beta)
        y += x_pos * math.sin(beta)
        return [alpha, beta, x, y]

    def _get_extremes_of_position(self, idx, border):
        """
        Get the extreme points (x axis not converted) for a position
        in the track
        """
        r, l = True, True
        if self.num_lanes > 1:
            l, r = self.info[idx]['lanes']
        from_val = -TRACK_WIDTH * l + border * TRACK_WIDTH / 2
        to_val = +TRACK_WIDTH * r - border * TRACK_WIDTH / 2
        return from_val, to_val

    def get_position_near_junction(self, type_junction, tiles_before):
        '''
        type_junction (str) : 't', 'x' or 'xt' so far
        tiles_before  (int) : number of tiles before the t junction, can be
                              negative as well
        '''
        if type_junction == 'xt':
            if self.info['x'].sum() > 0:
                # To give priority to the rarely found x intersections
                filter = self.info['x']
            else:
                filter = (self.info['x'] == True) | (self.info['t'] == True)
        else:
            filter = self.info[type_junction] == True
        if filter.sum() > 0:
            beta, x, y, _ = self._get_tile_near_tile_in_filter(
                filter, tiles_before)
            return [beta, x, y]
        else:
            return False

    def _get_tile_near_tile_in_filter(self, filter, tiles_before=8):
        """
        this helper function receives a filter and a number of tiles, and will return
        a position tiles_before tiles before a position in the filter
        Have in mind that tiles_before can be negative
        """
        idx = np.random.choice(np.where(filter)[0])
        idx_relative = idx - (self.info['track'] <
                              self.info[idx]['track']).sum()
        track = self.track[self.info['track'] == self.info[idx]['track']]
        idx_general = (idx_relative + tiles_before) % len(track) + (
            self.info['track'] < self.info[idx]['track']).sum()
        _, beta, x, y = self._get_rnd_position_inside_lane(idx_general)
        if tiles_before > 0: beta += math.pi
        return beta, x, y, idx_general

    def get_position_near_obstacle(self, tiles_before=8):
        if self.num_obstacles > 0:
            filter = self.info['obstacles'] == True
            beta, x, y, idx = self._get_tile_near_tile_in_filter(
                filter, tiles_before=tiles_before)
            return beta, x, y, idx
        return False

    def get_position_outside(self, distance):
        '''
        Returns a random position outside the track with random angle

        bear in mind that distance can be negative
        '''
        idx = np.random.randint(0, len(self.track))
        angle = np.random.uniform(0, 2 * math.pi)
        _, beta, x, y = self.track[idx, 1, :]
        r, l = True, True
        if self.num_lanes > 1:
            l, r = self.info[idx]['lanes']
        if distance > 0:
            x = x + (r * TRACK_WIDTH + distance) * math.cos(beta)
            y = y + (r * TRACK_WIDTH + distance) * math.sin(beta)
        else:
            x = x - (l * TRACK_WIDTH + abs(distance)) * math.cos(beta)
            y = y - (l * TRACK_WIDTH + abs(distance)) * math.sin(beta)

        return [angle, x, y]

    def switch_intersection_points(self):
        global SHOW_INTERSECTION_POINTS
        SHOW_INTERSECTION_POINTS += 1
        SHOW_INTERSECTION_POINTS %= 2

    def switch_intersection_groups(self):
        global SHOW_GROUP_INTERSECTIONS
        SHOW_GROUP_INTERSECTIONS += 1
        SHOW_GROUP_INTERSECTIONS %= 2

    def switch_xt_intersections(self):
        global SHOW_XT_JUNCTIONS
        SHOW_XT_JUNCTIONS += 1
        SHOW_XT_JUNCTIONS %= 2

    def switch_end_of_track(self):
        global SHOW_ENDS_OF_TRACKS
        SHOW_ENDS_OF_TRACKS += 1
        SHOW_ENDS_OF_TRACKS %= 2

    def switch_start_of_track(self):
        global SHOW_START_OF_TRACKS
        SHOW_START_OF_TRACKS += 1
        SHOW_START_OF_TRACKS %= 2

    def change_zoom(self):
        global ZOOM_OUT, ZOOM
        ZOOM_OUT = (ZOOM_OUT + 1) % 2
        if ZOOM_OUT: ZOOM = 0.25
        else: ZOOM = 2.7

    # TODO delete if there is not error by commenting this out
    #def set_press_fn(self,key_press_fn):
    #self.key_press_fn = key_press_fn
    #if self.viewer is not None:
    #if self.key_press_fn is not None:
    #self.viewer.window.on_key_press = self.key_press_fn
    #def set_release_fn(self,key_release_fn):
    #self.key_release_fn = key_release_fn
    #if self.viewer is not None:
    #if self.key_release_fn is not None:
    #self.viewer.window.on_key_release = self.key_release_fn


def play(env):
    """
    run this function in order to create a window and be able to play
    this environment.

    env:        CarRacing env
    """
    from pyglet.window import key
    discretize = env.discretize_actions
    if discretize == None:
        a = np.array([0.0, 0.0, 0.0])
    else:
        a = np.array([0])

    def key_press(k, mod):
        global restart
        if discretize == None:
            if k == 0xff0d: restart = True
            if k == key.LEFT: a[0] = -1.0
            if k == key.RIGHT: a[0] = +1.0
            if k == key.UP: a[1] = +1.0
            if k == key.DOWN: a[1] = -1.0
            if k == key.SPACE:
                a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
        elif discretize == "hard":
            if k == 0xff0d: restart = True
            if k == key.LEFT: a[0] = 1
            if k == key.RIGHT: a[0] = 2
            if k == key.UP: a[0] = 3
            if k == key.SPACE: a[0] = 4

    def key_release(k, mod):
        if discretize == None:
            if k == key.LEFT and a[0] == -1.0: a[0] = 0
            if k == key.RIGHT and a[0] == +1.0: a[0] = 0
            if k == key.UP: a[1] = 0
            if k == key.DOWN: a[1] = 0
            if k == key.SPACE: a[2] = 0
        else:
            a[0] = 0
        if k == key.D: set_trace()
        if k == key.R: env.reset()
        if k == key.Z: env.change_zoom()
        if k == key.G: env.switch_intersection_groups()
        if k == key.I: env.switch_intersection_points()
        if k == key.X: env.switch_xt_intersections()
        if k == key.E: env.switch_end_of_track()
        if k == key.S: env.switch_start_of_track()
        if k == key.T: env.screenshot('./')
        if k == key.Q: sys.exit()

    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.key_press_fn = key_press
    env.key_release_fn = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False

        while True:
            if discretize != None: a_tmp = a[0]
            else: a_tmp = a
            s, r, done, info = env.step(a_tmp)
            total_reward += r
            if steps % 200 == 0 or done:
                #print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(
                    steps, total_reward))
                steps += 1
                print("step {} total_reward {:+0.2f}".format(
                    steps, total_reward))
            steps += 1
            if not record_video:  # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break

    env.close()


from gym.envs.registration import register

for i, obs_level in enumerate([0., 0.1, 0.3]):
    register(
        id=f"SafeCarRacing{i}-v0",
        entry_point=
        'alf.environments.safe_car_racing.car_racing_hardcore:CarRacing',
        kwargs={
            'num_obstacles': obs_level,
            'discretize_actions': None,
            'verbose': 1,
            'show_info_panel': False,
            'observe_car_status': True,
            'reward_fn': safe_reward_callback
        })
