from ray.rllib.algorithms.dreamer import DreamerConfig
from gym_auv import DEFAULT_CONFIG

class AuvDreamerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()

        # The line below is needed to be able to pass a custom model config
        self.dreamer_model["dense_size"] = DEFAULT_CONFIG.vessel.dense_observation_size
        self.dreamer_model["lidar_shape"]: DEFAULT_CONFIG.vessel.lidar_shape