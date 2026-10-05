from hypothesis import settings

_second = 1000
_base_settings = settings(deadline=0.5 * _second)
BASIC_SETTINGS = settings(_base_settings, max_examples=100)
SAMPLE_SETTINGS = settings(_base_settings, max_examples=50)
ML_SETTINGS = settings(_base_settings, max_examples=3, deadline=60 * _second)
