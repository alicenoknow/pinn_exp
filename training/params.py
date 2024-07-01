from dataclasses import dataclass, fields
from enum import Enum
import json
import os
from typing import Tuple


class LossFunction(Enum):
    BASE = "BASE"
    RELO = "RELO"
    SOFTADAPT = "SOFTADAPT"

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass
class SimulationParameters(metaclass=SingletonMeta):
    _shared_state = {}
    
    RUN_NUM: int = 0
    DIR: str = "results"

    # Model
    LAYERS: int = 12
    NEURONS_PER_LAYER: int = 100
    MODEL_PATH: str = "" # Path to load saved weights
    MESH: str = ""

    # Training
    LOSS: LossFunction = LossFunction.BASE
    EPOCHS: int = 200
    LEARNING_RATE: float = 0.00015
    LBFGS_EPOCHS: int = 0 # After n epochs with adam, use LBFGS for fine tuning

    # Initial condition
    BASE_HEIGHT: float = 0.0  # Base water level
    DECAY_RATE: float = 120   # The rate of decay, how quickly func decreases with distance
    PEAK_HEIGHT: float = 1    # The height of the function's peak
    X_DIVISOR: float = 2      # The divisor used to calculate the x-coord of the center of the function
    Y_DIVISOR: float = 2      # The divisor used to calculate the y-coord of the center of the function

    # Initial weights
    INITIAL_WEIGHT_RESIDUAL: float = 1        # Weight of residual part of loss function
    INITIAL_WEIGHT_INITIAL: float = 1         # Weight of initial part of loss function
    INITIAL_WEIGHT_BOUNDARY: float = 1        # Weight of boundary part of loss function

    # Domain
    XY_DOMAIN: Tuple[float, float] = (0, 1.0)
    T_DOMAIN: Tuple[float, float] = (0, 1.0)
    T_POINTS: int = 40
    INTERIOR_POINTS: int = 40
    INITIAL_POINTS: int = 40
    BOUNDARY_POINTS: int = 40
    POINTS_PLOT: int = 100

    def __init__(self, **kwargs):
        self.__dict__ = self._shared_state

        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_json(self, json_file):
        if json_file is None or not os.path.isfile(json_file):
            return

        with open(json_file, 'r') as f:
            config = json.load(f)

        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def set_run_num(self, run_num):
        self.RUN_NUM = run_num

    def save_params(self):
        """
            Save parameters to run's directory.
        """
        file_dir = os.path.join(self.DIR, f"run_{self.RUN_NUM}")
        file_path = os.path.join(file_dir, "config.json")
        os.makedirs(file_dir, exist_ok=True)
        params_to_save = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Enum):
                value = value.value
        with open(file_path, 'w+') as json_file:
            json.dump(params_to_save, json_file, indent=4)
