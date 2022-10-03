I am currently adapting `Dreamer` for an environment that uses Dict environments, but the Config does not work as intended.

After spending some time stepping into the library with a debugger, I have found that although my argument `preprocessor_pref = None` is correctly propagated into [RolloutWorker](https://github.com/ray-project/ray/blob/master/rllib/evaluation/rollout_worker.py), the value is not actually used to determine if we should disable preprocessing, even though the documentation says so.

However, if I set `_disable_preprocessor_api = True`, this behaviour does not happen (, but we run into another problem:  but this is not clear from the documentation, where I interpret it as setting `preprocessor_pref = None` to be sufficient.

From the [training documenation](https://github.com/ray-project/ray/blob/master/doc/source/rllib/rllib-training.rst#common-parameters):

```
# Whether to use "rllib" or "deepmind" preprocessors by default
# Set to None for using no preprocessor. In this case, the model will have
# to handle possibly complex observations from the environment.
"preprocessor_pref": "deepmind",
```

