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
"""A utility to watch the vehicles in a simulation.

A typical scenario is that you have an on-going Carla training session and you
want to see what the training vehicles are doing. You can use this utility to do
this:

.. code-block:: bash

    python carla_spectator --port 2000 --host localhost

If you only have one training session going on, the port is 2000 by default.
You can use `ps aux | grep Carla` to find out `--carla-rpc-port` and use it to
replace 2000.

After carla_spectator starts, you can use TAB key to switch to different vehicles
and ESC key to quit the program.
"""

from absl import app
from absl import logging
from absl import flags
import carla
import functools
import math
import numpy as np
import os

from alf.environments.suite_carla import CameraSensor

flags.DEFINE_integer('port', 2000, "Carla server RPC port")
flags.DEFINE_string('host', "localhost", "Carla server address")
FLAGS = flags.FLAGS


def _transform_to_ego(actor, target):
    trans = actor.get_transform()
    yaw = math.radians(trans.rotation.yaw)
    target = np.array([target.x, target.y, target.z], dtype=np.float32)
    cos, sin = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos, sin, 0.], [-sin, cos, 0.], [0., 0., 1.]])
    target = np.matmul(rot, target).astype(np.float32).tolist()
    return carla.Location(target[0], target[1], target[2])


def main(_):
    import pygame
    pygame.init()
    pygame.font.init()

    width = 640
    height = 320
    display = pygame.display.set_mode((width, height),
                                      pygame.HWSURFACE | pygame.DOUBLEBUF)

    font_name = 'courier' if os.name == 'nt' else 'mono'
    fonts = [x for x in pygame.font.get_fonts() if font_name in x]
    default_font = 'ubuntumono'
    mono = default_font if default_font in fonts else fonts[0]
    mono = pygame.font.match_font(mono)
    font = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
    info_surface = pygame.Surface((240, 220))
    info_surface.set_alpha(100)
    display.blit(info_surface, (0, 0))

    client = carla.Client(FLAGS.host, FLAGS.port, worker_threads=1)
    world = client.get_world()
    settings = world.get_settings()
    # For some unknown reason, world.get_actors() returns empty list without
    # calling world.apply_settings() first.
    world.apply_settings(settings)
    actors = world.get_actors()
    vehicles = []
    for actor in actors:
        if actor.type_id.startswith('vehicle'):
            vehicles.append(actor)
            logging.info("id: %s type_id: %s" % (actor.id, actor.type_id))
    logging.info("Found %s vechiles." % len(vehicles))
    if len(vehicles) == 0:
        logging.info("There is no vehicles in the simulation. Quit")
        return

    vehicle_id = 0
    xyz = [(1.6, 0., 1.7), (0., 0., 50.)]
    pyr = [(0., 0., 0.), (-90., 0., 0.)]
    view = 0

    def _make_camera(vehicle, view):
        return CameraSensor(
            vehicle,
            image_size_x=width,
            image_size_y=height,
            xyz=xyz[view],
            pyr=pyr[view])

    camera = _make_camera(vehicles[vehicle_id], view)
    clock = pygame.time.Clock()

    def on_tick(timestamp):
        camera.render(display)
        vehicle = vehicles[vehicle_id]
        v = _transform_to_ego(vehicle, vehicle.get_velocity())
        a = _transform_to_ego(vehicle, vehicle.get_acceleration())
        loc = vehicle.get_location()
        clock.tick_busy_loop(1000)
        info_text = [
            'FPS: %6.2f' % clock.get_fps(),
            'Location: (%6.3f, %6.3f, %6.3f)' % (loc.x, loc.y, loc.z),
            'Velocity: (%5.2f, %5.2f, %5.2f) m/s' % (v.x, v.y, v.z),
            'Acceleration: (%5.2f, %5.2f, %5.2f) m/s^2' % (a.x, a.y, a.z),
        ]
        v_offset = 4
        for item in info_text:
            surface = font.render(item, True, (255, 255, 255))
            display.blit(surface, (8, v_offset))
            v_offset += 18

        pygame.display.flip()

    callback_id = world.on_tick(on_tick)

    logging.info("Keyboard control:" + """
    ESC          : quit
    TAB          : switch vehicle
    SPACE        : switch between ego view and birdeye view
    """)
    stopping = False
    import pygame.locals as K
    try:
        while not stopping:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stopping = True
                    break
                if event.type != pygame.KEYUP:
                    continue
                if event.key == K.K_ESCAPE:
                    stopping = True
                    break
                if event.key == K.K_TAB:
                    client.apply_batch_sync(camera.destroy(), True)
                    vehicle_id = (vehicle_id + 1) % len(vehicles)
                    vehicle = vehicles[vehicle_id]
                    camera = _make_camera(vehicle, view)
                    logging.info("Switched to id: %s type_id: %s" %
                                 (vehicle.id, vehicle.type_id))
                if event.key == K.K_SPACE:
                    view = 1 - view
                    client.apply_batch_sync(camera.destroy(), True)
                    vehicle = vehicles[vehicle_id]
                    camera = _make_camera(vehicle, view)
    finally:
        world.remove_on_tick(callback_id)
        client.apply_batch_sync(camera.destroy(), True)
        pygame.quit()


if __name__ == '__main__':
    app.run(main)
