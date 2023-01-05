import datetime


def get_now_as_string() -> str:
    """Returns a string of the current time, suitable for directory or file names."""
    now = datetime.datetime.now()
    formatted = now.strftime("%Y%m%d-%H%M%S")
    return formatted


def print_last_reward(progress: dict):
    hist_stats = progress.get("hist_stats")
    if hist_stats is None:
        return

    episode_rewards = hist_stats.get("episode_reward")
    if episode_rewards is None:
        return

    print(f"Last episode reward: {episode_rewards[-1]}")
