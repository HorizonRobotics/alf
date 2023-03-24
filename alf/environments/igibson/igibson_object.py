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
import igibson
from igibson.objects.articulated_object import ArticulatedObject


class iGibsonObject(ArticulatedObject):
    """iGibson dataset object from igibson/data/ig_dataset/objects"""

    def __init__(self, name, scale=1.):
        """
        Args:
            name (str): name of the object, corresponding to the directory
                igibson/data/ig_dataset/objects/'name'
            scale (float): scale of object
        """
        dirname = os.path.join(igibson.ig_dataset_path, 'objects', name)
        object_dir = [f.path for f in os.scandir(dirname) if f.is_dir()][0]
        object_name = os.path.basename(object_dir)
        filename = os.path.join(object_dir, f'{object_name}.urdf')
        super(iGibsonObject, self).__init__(filename, scale)

    def load(self):
        """Load the object into pybullet.

        _load() will be implemented in the subclasses
        Returns:
            int: PyBullet object unique id
        """
        if self.loaded:
            return self.body_id
        self.body_id = self._load()
        self.loaded = True

        return self.body_id
