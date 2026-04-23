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

"""Tests for WandbMetricWriter."""

from __future__ import annotations

import sys
import types
from unittest import mock

from kauldron.contrib.train import wandb_metric_writer
import numpy as np
import pytest


class _Tagged:
  """Captures constructor args passed to fake `wandb.{Image,Video,...}`."""

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs


class _FakeConfig:

  def __init__(self):
    self.update_calls = []

  def update(self, data, allow_val_change=False):
    self.update_calls.append((dict(data), allow_val_change))


def _make_fake_wandb():
  fake = types.ModuleType("wandb")
  fake.init = mock.MagicMock(return_value=mock.MagicMock(name="WandbRun"))
  fake.log = mock.MagicMock()
  fake.finish = mock.MagicMock()
  fake.config = _FakeConfig()
  fake.Image = mock.MagicMock(side_effect=_Tagged)
  fake.Video = mock.MagicMock(side_effect=_Tagged)
  fake.Audio = mock.MagicMock(side_effect=_Tagged)
  fake.Histogram = mock.MagicMock(side_effect=_Tagged)
  fake.Object3D = mock.MagicMock(side_effect=_Tagged)
  fake.Html = mock.MagicMock(side_effect=_Tagged)
  return fake


@pytest.fixture
def fake_wandb(monkeypatch):
  fake = _make_fake_wandb()
  monkeypatch.setitem(sys.modules, "wandb", fake)
  # The writer module's `wandb` symbol is an epy lazy proxy; point it at the
  # fake module so attribute access returns our mocks.
  monkeypatch.setattr(wandb_metric_writer, "wandb", fake)
  return fake


def _make_writer(**kwargs):
  defaults = dict(
      project="p",
      run_name="r",
      mode="disabled",
      pass_to_default_writer=False,
      collection="train",
  )
  defaults.update(kwargs)
  return wandb_metric_writer.WandbMetricWriter(**defaults)


# ---------- helpers ----------


def test_float_img_to_uint8_converts_float():
  img = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
  out = wandb_metric_writer._float_img_to_uint8(img)
  assert out.dtype == np.uint8
  np.testing.assert_array_equal(out, np.array([[0, 127, 255]], dtype=np.uint8))


def test_float_img_to_uint8_clips_out_of_range():
  img = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
  out = wandb_metric_writer._float_img_to_uint8(img)
  np.testing.assert_array_equal(out, np.array([0, 127, 255], dtype=np.uint8))


def test_float_img_to_uint8_passthrough_uint8():
  img = np.array([0, 128, 255], dtype=np.uint8)
  out = wandb_metric_writer._float_img_to_uint8(img)
  assert out is img  # same array, no copy needed


def test_video_thwc_to_tchw_uint8_shape():
  video = np.random.rand(3, 4, 5, 3).astype(np.float32)  # (T, H, W, C)
  out = wandb_metric_writer._video_thwc_to_tchw_uint8(video)
  assert out.shape == (3, 3, 4, 5)  # (T, C, H, W)
  assert out.dtype == np.uint8


# ---------- write_scalars ----------


def test_write_scalars_prefixes_collection_and_casts_float(fake_wandb):
  writer = _make_writer()
  writer.write_scalars(step=5, scalars={"loss": 0.25, "acc": np.float32(0.9)})
  assert fake_wandb.log.call_count == 1
  payload, kwargs = fake_wandb.log.call_args.args, fake_wandb.log.call_args.kwargs
  assert payload[0] == {"train/loss": 0.25, "train/acc": pytest.approx(0.9)}
  assert kwargs == {"step": 5}
  assert all(isinstance(v, float) for v in payload[0].values())


# ---------- write_images ----------


def test_write_images_per_image_wrapping_and_uint8(fake_wandb):
  writer = _make_writer()
  imgs = np.random.rand(2, 4, 4, 3).astype(np.float32)
  writer.write_images(step=10, images={"preds": imgs})

  # Image should be called once per image in the batch.
  assert fake_wandb.Image.call_count == 2
  for call in fake_wandb.Image.call_args_list:
    (arr,) = call.args
    assert arr.dtype == np.uint8
    assert arr.shape == (4, 4, 3)

  # Logged payload should hold a list of length-2 under the prefixed key.
  (payload,), kwargs = (
      fake_wandb.log.call_args.args,
      fake_wandb.log.call_args.kwargs,
  )
  assert list(payload.keys()) == ["train/preds"]
  assert len(payload["train/preds"]) == 2
  assert kwargs == {"step": 10}


# ---------- write_histograms ----------


def test_write_histograms_uses_num_buckets(fake_wandb):
  writer = _make_writer()
  arrays = {"w": np.arange(100).reshape(10, 10)}
  writer.write_histograms(step=1, arrays=arrays, num_buckets={"w": 32})

  fake_wandb.Histogram.assert_called_once()
  kwargs = fake_wandb.Histogram.call_args.kwargs
  assert kwargs["num_bins"] == 32
  assert kwargs["sequence"].shape == (100,)
  (payload,), log_kwargs = (
      fake_wandb.log.call_args.args,
      fake_wandb.log.call_args.kwargs,
  )
  assert list(payload.keys()) == ["train/w"]
  assert log_kwargs == {"step": 1}


def test_write_histograms_defaults_to_64_when_missing(fake_wandb):
  writer = _make_writer()
  writer.write_histograms(step=1, arrays={"w": np.arange(50)}, num_buckets=None)
  assert fake_wandb.Histogram.call_args.kwargs["num_bins"] == 64


# ---------- write_videos ----------


def test_write_videos_transposes_and_lists_per_batch(fake_wandb):
  writer = _make_writer()
  vids = np.random.rand(2, 3, 5, 6, 3).astype(np.float32)  # (N,T,H,W,C)
  writer.write_videos(step=7, videos={"vid": vids})

  assert fake_wandb.Video.call_count == 2
  for call in fake_wandb.Video.call_args_list:
    (arr,) = call.args
    assert arr.shape == (3, 3, 5, 6)  # (T, C, H, W)
    assert arr.dtype == np.uint8
    assert call.kwargs == {"fps": 4, "format": "mp4"}

  (payload,), kwargs = (
      fake_wandb.log.call_args.args,
      fake_wandb.log.call_args.kwargs,
  )
  assert list(payload.keys()) == ["train/vid"]
  assert len(payload["train/vid"]) == 2
  assert kwargs == {"step": 7}


# ---------- write_audios ----------


def test_write_audios_passes_sample_rate(fake_wandb):
  writer = _make_writer()
  audios = np.random.rand(2, 16, 1).astype(np.float32)  # (N, T, C)
  writer.write_audios(step=3, audios={"aud": audios}, sample_rate=16_000)

  assert fake_wandb.Audio.call_count == 2
  for call in fake_wandb.Audio.call_args_list:
    (arr,) = call.args
    assert arr.shape == (16, 1)
    assert call.kwargs == {"sample_rate": 16_000}


# ---------- write_pointcloud ----------


def test_write_pointcloud_concats_colors(fake_wandb):
  writer = _make_writer()
  pts = np.zeros((10, 3), dtype=np.float32)
  colors = np.ones((10, 3), dtype=np.float32)
  writer.write_pointcloud(
      step=0, point_clouds={"pc": pts}, point_colors={"pc": colors}
  )
  (arr,) = fake_wandb.Object3D.call_args.args
  assert arr.shape == (10, 6)
  np.testing.assert_array_equal(arr[:, 3:], 1.0)


def test_write_pointcloud_without_colors(fake_wandb):
  writer = _make_writer()
  pts = np.zeros((5, 3), dtype=np.float32)
  writer.write_pointcloud(step=0, point_clouds={"pc": pts})
  (arr,) = fake_wandb.Object3D.call_args.args
  assert arr.shape == (5, 3)


# ---------- write_texts ----------


def test_write_texts_wraps_in_html(fake_wandb):
  writer = _make_writer()
  writer.write_texts(step=2, texts={"t": "hello"})

  fake_wandb.Html.assert_called_once()
  (html_arg,) = fake_wandb.Html.call_args.args
  assert html_arg == "<pre>hello</pre>"
  (payload,), kwargs = (
      fake_wandb.log.call_args.args,
      fake_wandb.log.call_args.kwargs,
  )
  assert list(payload.keys()) == ["train/t"]
  assert kwargs == {"step": 2}


# ---------- write_hparams / write_config ----------


def test_write_hparams_updates_config(fake_wandb):
  writer = _make_writer()
  writer.write_hparams({"lr": 0.1, "bs": 32})
  assert fake_wandb.config.update_calls == [
      ({"lr": 0.1, "bs": 32}, True),
  ]


def test_write_config_stores_json(fake_wandb):
  writer = _make_writer()
  config = mock.MagicMock()
  config.to_json.return_value = '{"x": 1}'
  writer.write_config(config)
  assert fake_wandb.config.update_calls == [
      ({"kd_config": '{"x": 1}'}, True),
  ]


def test_write_config_noop_on_none(fake_wandb):
  writer = _make_writer()
  writer.write_config(None)
  assert fake_wandb.config.update_calls == []


# ---------- close ----------


def test_close_calls_wandb_finish(fake_wandb):
  writer = _make_writer()
  # Trigger a log to materialize the run.
  writer.write_scalars(step=0, scalars={"x": 1.0})
  writer.close()
  fake_wandb.finish.assert_called_once_with()


# ---------- pass_to_default_writer ----------


def test_pass_to_default_writer_false_does_not_invoke_super(fake_wandb):
  writer = _make_writer(pass_to_default_writer=False)
  with mock.patch.object(
      wandb_metric_writer.metric_writer.KDMetricWriter,
      "write_scalars",
  ) as super_scalars:
    writer.write_scalars(step=0, scalars={"x": 1.0})
    super_scalars.assert_not_called()


def test_pass_to_default_writer_true_invokes_super(fake_wandb):
  writer = _make_writer(pass_to_default_writer=True)
  with mock.patch.object(
      wandb_metric_writer.metric_writer.KDMetricWriter,
      "write_scalars",
  ) as super_scalars:
    writer.write_scalars(step=0, scalars={"x": 1.0})
    super_scalars.assert_called_once()


# ---------- lead host gating ----------


def test_non_lead_host_skips_wandb_calls(fake_wandb, monkeypatch):
  # Mirror the `status.is_lead_host` gate by forcing it to False.
  monkeypatch.setattr(
      wandb_metric_writer.status,
      "is_lead_host",
      False,
  )
  writer = _make_writer()
  writer.write_scalars(step=0, scalars={"x": 1.0})
  fake_wandb.init.assert_not_called()
  fake_wandb.log.assert_not_called()
