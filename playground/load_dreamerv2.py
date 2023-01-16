file_path = "/home/krisbrud/repos/master-thesis/logdir/MovingObstaclesLosRewarder-v0-jan-14/variables.pkl"

# Load the file using pickle

import pickle
with open(file_path, 'rb') as f:
    agent = pickle.load(f)
