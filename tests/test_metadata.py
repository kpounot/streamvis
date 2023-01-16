import pytest

import streamvis as sv

test_shapes = [(20, 100), (1, 350), (800, 800)]


def test_danger_with_issue():
    sv_meta = sv.MetadataHandler()
    sv_meta.add_issue("test")
    sv_meta.update({})

    assert len(sv_meta.issues_datatable.source.data["issues"]) == 1


@pytest.mark.parametrize("shape", test_shapes)
def test_check_shape_good(shape):
    sv_meta = sv.MetadataHandler(check_shape=shape)
    sv_meta.update({"shape": shape})

    assert len(sv_meta.issues_datatable.source.data["issues"]) == 0


@pytest.mark.parametrize("shape", test_shapes)
def test_check_shape_bad(shape):
    sv_meta = sv.MetadataHandler(check_shape=shape)
    sv_meta.update({"shape": (42, 42)})

    assert len(sv_meta.issues_datatable.source.data["issues"]) == 1


def test_is_good_frame_good():
    sv_meta = sv.MetadataHandler()
    sv_meta.update({"is_good_frame": 1})

    assert len(sv_meta.issues_datatable.source.data["issues"]) == 0


def test_is_good_frame_bad():
    sv_meta = sv.MetadataHandler()
    sv_meta.update({"is_good_frame": 0})

    assert len(sv_meta.issues_datatable.source.data["issues"]) == 1


def test_saturated_pixels_good():
    sv_meta = sv.MetadataHandler()
    sv_meta.update({"saturated_pixels": 0})

    assert len(sv_meta.issues_datatable.source.data["issues"]) == 0


def test_saturated_pixels_bad():
    sv_meta = sv.MetadataHandler()
    sv_meta.update({"saturated_pixels": 42})

    assert len(sv_meta.issues_datatable.source.data["issues"]) == 1
