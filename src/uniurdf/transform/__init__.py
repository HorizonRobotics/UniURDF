# Project uniurdf
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""URDF transform stages.

Each stage mutates an ``ElementTree`` ``<robot>`` element in place and reports
what it changed. They are internal seams: callers go through
:mod:`uniurdf.robot` rather than composing stages by hand.

Nothing is re-exported here on purpose. ``_linalg`` sits at the bottom of the
import graph and ``camera`` sits near the top (it needs the spec types), so
eagerly importing the stages from this package's ``__init__`` would make any
import of ``_linalg`` pull in the whole graph and cycle back on itself. Import
each stage from its own module.
"""

__all__: list[str] = []
