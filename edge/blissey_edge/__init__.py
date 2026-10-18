"""Blissey edge runtime: CSI packets -> windows -> portable features -> model -> alerts.

Standard library only (Python >= 3.8). No numpy, scikit-learn, pickle or native extensions, so it
runs on any CPython build, including QNX on Raspberry Pi. The same code computes the features used
in training, so training and on-device inference cannot drift apart.
"""
FEATURE_VERSION = 'pv1'
__all__ = ['FEATURE_VERSION']
