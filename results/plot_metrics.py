# %%

# Plot the file "results/metrics/MovingObstaclesLosRewarder-v0-jan-16-4.csv" using seaborn
# Path: results/plot_metrics.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

base_dir = "/home/krisbrud/repos/master-thesis/"
metrics_path = "results/metrics/jan-metrics.jsonl"
path = os.path.join(base_dir, metrics_path)
# metrics = pd.read_csv(path)
metrics = pd.read_json(path, lines=True)
metrics.head()

x_column = "step"
columns_to_plot = [
    # "step",
    "return",
    "length",
    "total_steps",
    "total_episodes",
    "loaded_steps",
    "loaded_episodes",
    "kl_loss",
    "image_loss",
    "dense_loss",
    "reward_loss",
    "discount_loss",
    "model_kl",
    "prior_ent",
    "post_ent",
    "model_loss",
    "model_grad_norm",
    "actor_loss",
    "actor_grad_norm",
    "critic_loss",
    "critic_grad_norm",
    "reward_mean",
    "reward_std",
    "reward_normed_mean",
    "reward_normed_std",
    "critic_slow",
    "critic_target",
    "actor_ent",
    "actor_ent_scale",
    "critic",
    "fps",
]


# Set matplotlib style to ggplot
plt.style.use("ggplot")

# Plot all columns in columns_to_plot against "step"
for column in columns_to_plot:
    plt.figure()
    sns.lineplot(x=x_column, y=column, data=metrics)


# Plot the "Step" and "Value" columns
# sns.lineplot(x="Step", y="Value", data=metrics)

# Set the y label to be "Reward"
# plt.ylabel("Reward")

# %%
