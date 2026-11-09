# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

from numpydantic import NDArray
from pydantic import BaseModel, Field, field_serializer

from .embodiment_tags import EmbodimentTag

# Common schema


class RotationType(Enum):
    """Type of rotation representation"""

    AXIS_ANGLE = "axis_angle"
    QUATERNION = "quaternion"
    ROTATION_6D = "rotation_6d"
    MATRIX = "matrix"
    EULER_ANGLES_RPY = "euler_angles_rpy"
    EULER_ANGLES_RYP = "euler_angles_ryp"
    EULER_ANGLES_PRY = "euler_angles_pry"
    EULER_ANGLES_PYR = "euler_angles_pyr"
    EULER_ANGLES_YRP = "euler_angles_yrp"
    EULER_ANGLES_YPR = "euler_angles_ypr"


# LeRobot schema


# Dataset schema (parsed from LeRobot schema and simplified)


class DatasetStatisticalValues(BaseModel):
    max: NDArray = Field(..., description="Maximum values")
    min: NDArray = Field(..., description="Minimum values")
    mean: NDArray = Field(..., description="Mean values")
    std: NDArray = Field(..., description="Standard deviation")
    q01: NDArray = Field(..., description="1st percentile values")
    q99: NDArray = Field(..., description="99th percentile values")

    @field_serializer("*", when_used="json")
    def serialize_ndarray(self, v: NDArray) -> list[float]:
        return v.tolist()  # type: ignore


class DatasetStatistics(BaseModel):
    state: dict[str, DatasetStatisticalValues] = Field(..., description="Statistics of the state")
    action: dict[str, DatasetStatisticalValues] = Field(..., description="Statistics of the action")


class VideoMetadata(BaseModel):
    """Metadata of the video modality"""

    resolution: tuple[int, int] = Field(..., description="Resolution of the video")
    channels: int = Field(..., description="Number of channels in the video", gt=0)
    fps: float = Field(..., description="Frames per second", gt=0)


class StateActionMetadata(BaseModel):
    absolute: bool = Field(..., description="Whether the state or action is absolute")
    rotation_type: RotationType | None = Field(None, description="Type of rotation, if any")
    shape: tuple[int, ...] = Field(..., description="Shape of the state or action")
    continuous: bool = Field(..., description="Whether the state or action is continuous")


class DatasetModalities(BaseModel):
    video: dict[str, VideoMetadata] = Field(..., description="Metadata of the video")
    state: dict[str, StateActionMetadata] = Field(..., description="Metadata of the state")
    action: dict[str, StateActionMetadata] = Field(..., description="Metadata of the action")


class DatasetMetadata(BaseModel):
    """Metadata of the trainable dataset

    Changes:
        - Update to use the new RawCommitHashMetadataMetadata_V1_2
    """

    statistics: DatasetStatistics = Field(..., description="Statistics of the dataset")
    modalities: DatasetModalities = Field(..., description="Metadata of the modalities")
    embodiment_tag: EmbodimentTag = Field(..., description="Embodiment tag of the dataset")
