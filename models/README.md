# Dreamer

The original implementation of `Dreamer` in `ray.rllib` assumes picture inputs. As these are used in the project and as the source code is somewhat coupled, it was found that the easiest way was to copy the source code and give it support for this.