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
# x_column = "total_episodes"
columns_to_plot = [
    # "step",
    "return",
    "length",
    # "total_steps",
    "total_episodes",
    # "loaded_steps",
    # "loaded_episodes",
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
    # "fps",
]

def beautify_column_name(column_name):
    # Replace underscores with spaces, and make first letter uppercase
    return column_name.replace("_", " ").capitalize()

# Set matplotlib style to ggplot
plt.style.use("ggplot")

prefill = 50e3
metrics = metrics[metrics[x_column] > prefill]

# Plot all columns in columns_to_plot against "step"

# columns_to_plot = columns_to_plot[:1]

for column in columns_to_plot:
    plt.figure()
    y_label = beautify_column_name(column)
    x_label = beautify_column_name(x_column)

    smoothed_column = column + "_smoothed"
    metrics[smoothed_column] = metrics[column].dropna().rolling(10).mean()

    # print(smoothed_column)
    # print(metrics[smoothed_column])

    color = (55/255, 75/255, 105/255)


    # sns.lineplot(x=x_column, y=column, data=metrics, alpha=0.5, color="b")
    sns.lineplot(x=x_column, y=column, data=metrics, alpha=0.5, color=color)

    if not metrics[smoothed_column].isnull().all():
        sns.lineplot(x=x_column, y=smoothed_column, data=metrics, color=color)

    legends = ["Raw", "Moving average"]

    plt.ylabel(y_label)
    plt.xlabel(x_label)

# Make pages of subplots with 2 columns and 4 rows each
