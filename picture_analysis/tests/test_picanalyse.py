from pathlib import Path

import pytest
from PIL import Image

import sys

# Ensure the project root is importable when tests run from this package context.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import picanalyse


class DummyInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class DummyProcessor:
    def __init__(self):
        self.call_count = 0
        self.last_image_mode = None
        self.decode_calls = 0

    def __call__(self, image, *, return_tensors):
        self.call_count += 1
        self.last_image_mode = image.mode
        assert return_tensors == "pt"
        return DummyInputs()

    def decode(self, ids, *, skip_special_tokens):
        self.decode_calls += 1
        assert skip_special_tokens is True
        return "stub caption"


class DummyModel:
    def __init__(self):
        self.device = "cpu"
        self.to_calls = []
        self.generate_kwargs = None

    def to(self, device):
        self.to_calls.append(device)
        self.device = device
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[101, 102]]


@pytest.fixture()
def repo_root() -> Path:
    return PROJECT_ROOT


def test_can_access_image_rejects_missing_file():
    assert not picanalyse.can_access_image(Path("does_not_exist.jpg"))


def test_can_access_image_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "image.txt"
    bad_file.write_text("not an image")
    assert not picanalyse.can_access_image(bad_file)


def test_can_access_image_accepts_real_image(repo_root):
    image_path = repo_root / "test1.jpeg"
    assert picanalyse.can_access_image(image_path)


def test_describe_image_returns_error_for_unreadable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(picanalyse, "load_model", lambda *args, **kwargs: (None, None))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not really an image")
    assert picanalyse.describe_image(broken) == "unable to read the file"


def test_describe_image_generates_caption_with_supplied_components(tmp_path):
    image_path = tmp_path / "tiny.jpg"
    Image.new("RGB", (5, 5), color="red").save(image_path)

    processor = DummyProcessor()
    model = DummyModel()

    caption = picanalyse.describe_image(
        image_path,
        device="cuda",
        processor=processor,
        model=model,
    )

    assert caption == "stub caption"
    assert processor.call_count == 1
    assert processor.last_image_mode == "RGB"
    assert processor.decode_calls == 1
    assert model.to_calls == ["cuda"]
    assert model.generate_kwargs == {"device": "cuda", "max_new_tokens": 30}
