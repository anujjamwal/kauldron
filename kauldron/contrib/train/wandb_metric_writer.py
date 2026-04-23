# Copyright 2026 The kauldron Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WandbMetricWriter."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Mapping, Optional, Sequence

from etils import epy
from kauldron import konfig
from kauldron.train import metric_writer
from kauldron.typing import Array, Float, Scalar  # pylint: disable=g-multiple-import
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member
import numpy as np

with epy.lazy_imports(
    error_callback="must `pip install wandb` to use WandbMetricWriter"
):
  import wandb  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


def _float_img_to_uint8(img: np.ndarray) -> np.ndarray:
  if np.issubdtype(img.dtype, np.floating):
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
  if img.dtype != np.uint8:
    return img.astype(np.uint8)
  return img


def _video_thwc_to_tchw_uint8(video: np.ndarray) -> np.ndarray:
  # (T, H, W, C) -> (T, C, H, W) uint8 for wandb.Video.
  return np.transpose(_float_img_to_uint8(video), (0, 3, 1, 2))


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class WandbMetricWriter(metric_writer.KDMetricWriter):
  """Weights & Biases integration for Kauldron Trainer.

  Logs scalars, images, histograms, videos, audios, point clouds, and text
  summaries to a W&B run. Only the lead host opens a run; `wandb.init` is
  invoked lazily on the first write. Setting `pass_to_default_writer=True`
  (the default) will also forward every write to the parent `KDMetricWriter`
  so that TensorBoard / logging output continues to be produced alongside W&B.

  Requires `pip install wandb` (declared in the `wandb` optional extra).

  Example usage in a Trainer config:

  ```
  from kauldron.contrib.train import WandbMetricWriter

  def get_config():
      cfg = kd.train.Trainer(...)
      cfg.writer = WandbMetricWriter(
          project="my-project",
          run_name="mnist-smoke",
          pass_to_default_writer=True,
      )
      return cfg
  ```
  """

  project: Optional[str] = None
  entity: Optional[str] = None
  run_name: Optional[str] = None
  group: Optional[str] = None
  tags: Optional[Sequence[str]] = None
  mode: Optional[str] = None  # "online" | "offline" | "disabled"
  notes: Optional[str] = None
  run_id: Optional[str] = None
  resume: Optional[str | bool] = None

  pass_to_default_writer: bool = True

  @functools.cached_property
  def _run(self):
    if not status.is_lead_host:
      return None
    return wandb.init(
        project=self.project,
        entity=self.entity,
        name=self.run_name,
        group=self.group,
        tags=list(self.tags) if self.tags else None,
        mode=self.mode,
        notes=self.notes,
        id=self.run_id,
        resume=self.resume,
        dir=str(self.workdir) if self.workdir else None,
        reinit=False,
    )

  def _key(self, name: str) -> str:
    return f"{self.collection}/{name}"

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]) -> None:
    if status.is_lead_host and self._run is not None:
      wandb.log(
          {self._key(k): float(v) for k, v in scalars.items()},
          step=step,
      )
    if self.pass_to_default_writer:
      return super().write_scalars(step, scalars)

  def write_images(
      self,
      step: int,
      images: Mapping[str, Array["n h w c"]],
  ) -> None:
    if status.is_lead_host and self._run is not None:
      payload = {}
      for name, arr in images.items():
        a = _float_img_to_uint8(np.asarray(arr))
        payload[self._key(name)] = [wandb.Image(a[i]) for i in range(a.shape[0])]
      wandb.log(payload, step=step)
    if self.pass_to_default_writer:
      return super().write_images(step, images)

  def write_histograms(
      self,
      step: int,
      arrays: Mapping[str, Array],
      *,
      num_buckets: Mapping[str, int] | None = None,
  ) -> None:
    if status.is_lead_host and self._run is not None:
      payload = {}
      for name, arr in arrays.items():
        bins = 64 if num_buckets is None else num_buckets.get(name, 64)
        payload[self._key(name)] = wandb.Histogram(
            sequence=np.asarray(arr).reshape(-1),
            num_bins=int(bins),
        )
      wandb.log(payload, step=step)
    if self.pass_to_default_writer:
      return super().write_histograms(step, arrays, num_buckets=num_buckets)

  def write_pointcloud(
      self,
      step: int,
      point_clouds: Mapping[str, Array],
      *,
      point_colors: Optional[Mapping[str, Array]] = None,
      configs: Optional[Mapping[str, Any]] = None,
  ) -> None:
    if status.is_lead_host and self._run is not None:
      payload = {}
      for name, pts in point_clouds.items():
        pts_np = np.asarray(pts)
        if point_colors is not None and point_colors.get(name) is not None:
          pts_np = np.concatenate(
              [pts_np, np.asarray(point_colors[name])], axis=-1
          )
        payload[self._key(name)] = wandb.Object3D(pts_np)
      wandb.log(payload, step=step)
    if self.pass_to_default_writer:
      return super().write_pointcloud(
          step, point_clouds, point_colors=point_colors, configs=configs
      )

  def write_videos(
      self,
      step: int,
      videos: Mapping[str, Array["n t h w c"]],
  ) -> None:
    if status.is_lead_host and self._run is not None:
      payload = {}
      for name, arr in videos.items():
        a = np.asarray(arr)  # (N, T, H, W, C)
        payload[self._key(name)] = [
            wandb.Video(
                _video_thwc_to_tchw_uint8(a[i]), fps=4, format="mp4"
            )
            for i in range(a.shape[0])
        ]
      wandb.log(payload, step=step)
    if self.pass_to_default_writer:
      return super().write_videos(step, videos)

  def write_audios(
      self,
      step: int,
      audios: Mapping[str, Float["n t c"]],
      *,
      sample_rate: int,
  ) -> None:
    if status.is_lead_host and self._run is not None:
      payload = {}
      for name, arr in audios.items():
        a = np.asarray(arr)  # (N, T, C)
        payload[self._key(name)] = [
            wandb.Audio(a[i], sample_rate=sample_rate)
            for i in range(a.shape[0])
        ]
      wandb.log(payload, step=step)
    if self.pass_to_default_writer:
      return super().write_audios(step, audios, sample_rate=sample_rate)

  def write_texts(self, step: int, texts: Mapping[str, str]) -> None:
    if status.is_lead_host and self._run is not None:
      wandb.log(
          {self._key(k): wandb.Html(f"<pre>{v}</pre>") for k, v in texts.items()},
          step=step,
      )
    if self.pass_to_default_writer:
      return super().write_texts(step, texts)

  def write_hparams(self, hparams: Mapping[str, Any]) -> None:
    if status.is_lead_host and self._run is not None:
      wandb.config.update(dict(hparams), allow_val_change=True)
    if self.pass_to_default_writer:
      return super().write_hparams(hparams)

  def write_config(self, config: konfig.ConfigDict) -> None:
    if (
        status.is_lead_host
        and self._run is not None
        and config is not None
    ):
      wandb.config.update(
          {"kd_config": config.to_json()}, allow_val_change=True
      )
    if self.pass_to_default_writer:
      return super().write_config(config)

  def write_param_overview(self, step: int, params) -> None:
    if status.is_lead_host and self._run is not None:
      wandb.log(
          {self._key("params"): wandb.Html(f"<pre>{params!r}</pre>")},
          step=step,
      )
    if self.pass_to_default_writer:
      return super().write_param_overview(step, params)

  def write_element_spec(self, step: int, element_spec) -> None:
    if status.is_lead_host and self._run is not None:
      wandb.log(
          {
              self._key("element_spec"): wandb.Html(
                  f"<pre>{element_spec!r}</pre>"
              )
          },
          step=step,
      )
    if self.pass_to_default_writer:
      return super().write_element_spec(step, element_spec)

  def close(self) -> None:
    if status.is_lead_host and self._run is not None:
      wandb.finish()
    if self.pass_to_default_writer:
      super().close()
