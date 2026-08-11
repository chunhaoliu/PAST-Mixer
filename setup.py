"""Portable path defaults for PAST-Mixer experiments."""
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(ROOT_PATH, 'checkpoints')
RESULTS = os.path.join(ROOT_PATH, 'results')
