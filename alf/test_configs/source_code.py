# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import alf


@alf.configurable
def test(a, b=123):
    return a, b


@alf.configurable(blacklist=['b'])
def test_func(a, b=100, c=200):
    return a, b, c


@alf.configurable(blacklist=['b'])
def test_func2(a, b=100, c=200):
    return a, b, c


@alf.configurable(blacklist=['b'])
def test_func3(a, b=100, c=200):
    return a, b, c


@alf.configurable("Test.FancyTest")
def test_func4(arg=10):
    return arg


def test_func5(arg=10):
    return arg


def test_func6(arg=10):
    return arg


def test_func7(arg=10):
    return arg


def test_func8(a=1, b=2, c=3):
    return a, b, c


def test_func9(a=1, b=2, c=3):
    return a, b, c


def test_func10(a=1, b=2, c=3):
    return a, b, c


def test_func11(a=1, b=2, c=3):
    return a, b, c


@alf.configurable
class Test(object):
    def __init__(self, a, b, c=10):
        self._a = a
        self._b = b
        self._c = c

    @alf.configurable(whitelist=['c'])
    def func(self, a, b=10, c=100):
        return a, b, c

    def __call__(self):
        return self._a, self._b, self._c


@alf.configurable
class Test2(Test):
    def __init__(self, a, b):
        super().__init__(a, b, 5)

    def func(self, a):
        return a

    def __call__(self):
        return self._a + 1, self._b + 2, self._c + 3


@alf.configurable
class Test3(Test):
    def func(self, a):
        return a

    def __call__(self):
        return self._a - 1, self._b - 2, self._c - 3


@alf.repr_wrapper
class MyClass(object):
    def __init__(self, a, b, c=100, d=200):
        pass


@alf.repr_wrapper
class MySubClass(MyClass):
    def __init__(self, x):
        super().__init__(3, 5)
