import numpy as np
import pytest


def test_polygon_conversion_preserves_separate_crossing_instances():
    from solar.yolo_solar_data import polygon_line
    a=polygon_line([[0,40,100,40,100,50,0,50]],100,100)
    b=polygon_line([[40,0,50,0,50,100,40,100]],100,100)
    assert a.startswith('0 ') and b.startswith('0 ') and a!=b
    assert len(a.split())==9
    assert all(0<=float(x)<=1 for x in a.split()[1:])
    with pytest.raises(ValueError):polygon_line([[0,0,10,10]],100,100)


def test_negative_crop_excludes_all_known_filament_pixels():
    from solar.yolo_solar_data import choose_negative_crop
    rng=np.random.default_rng(42);image=rng.integers(80,180,(1024,1024),dtype=np.uint8)
    forbidden=np.zeros_like(image);forbidden[:512]=1
    bounds=choose_negative_crop(image,forbidden,rng)
    assert bounds is not None
    x0,y0,x1,y1=bounds
    assert not forbidden[y0:y1,x0:x1].any()
    assert y0==512
    assert choose_negative_crop(image,np.ones_like(image),rng) is None
