import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp # for tfp.stats.quantiles
from types import SimpleNamespace
from tqdm import tqdm
import argh
from termcolor import colored # colorful printing
from os import path
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
if path.isfile("../style.mplstyle"):
  from matplotlib import pyplot as plt
  from matplotlib import colors as clrs
  plt.style.use("../style.mplstyle")
  plt.ion()
  import seaborn as sb
  from cmcrameri import cm

import stein

rosenbrock_func = lambda v: -tf.math.log(
  tf.math.reduce_sum(100.0 * tf.math.square(v[..., 1:] - tf.math.square(v[..., :-1])) + tf.math.square(1.0 - v[..., :-1]), axis=-1))

gaussian_func = lambda v: -tf.math.reduce_sum(tf.math.square(v), axis=-1)

functions = {
  "rosenbrock": rosenbrock_func,
  "gaussian": gaussian_func,
}

logit_epsilon = 1e-15
logit = lambda x: tf.where(x > logit_epsilon, tf.where((1.0-x) > logit_epsilon, tf.math.log(tf.math.divide_no_nan(x, 1.0 - x)), tf.math.log(1.0/logit_epsilon - 1.0)), -tf.math.log(1.0/logit_epsilon - 1.0))

def test_stein(func, n_particles=10, n_dims=2, n_components=1, min=-2.0, max=2.0, learning_rate=1e-2, n_iterations=1000, bandwidth=0.1, n_samples=100000, do_newton=False, logit_scale=100.0):
  components = [logit(tf.random.uniform([n_particles, n_dims], 0.0, 1.0)) * logit_scale for c in range(n_components)]

  get_log_likelihood = functions[func]

  def project_params(params):
    inputs = tf.concat(params, axis=-1)
    values = (max-min) * tf.math.sigmoid(inputs / logit_scale) + min
    return values

  @tf.function
  def compute_log_likelihood(params):
    values = project_params(params)
    return get_log_likelihood(values)

  @tf.function
  def evaluate_kernel(a, b):
    #return tf.math.exp(-tf.math.reduce_sum(tf.convert_to_tensor(
    #  [tf.math.reduce_sum(tf.square(a[i] - b[i]), axis=-1) for i in range(n_components)]), axis=0) / (2.0*bandwidth**2))
    return tf.math.exp(-tf.math.reduce_sum(tf.math.square(project_params(a) - project_params(b)), axis=-1) / (2.0*bandwidth**2))

  def report_progress(t, *particles):
    params = project_params(particles)
    if tf.random.uniform([]) < 5e-3:
      if plot(params):
        plt.show(block=True)

  uniform_samples = tf.random.uniform([n_samples, n_dims*n_components], min, max)
  sample_evaluations = tf.math.exp(get_log_likelihood(uniform_samples))

  def plot(samples):
    if n_dims*n_components == 2:
      plt.clf()
      plt.scatter(uniform_samples[:, 0], uniform_samples[:, 1], c=sample_evaluations, s=1, cmap="inferno")
      plt.scatter(samples[:, 0], samples[:, 1])
      return True
    return False

  particles, seconds_passed = stein.flow_particles(components, learning_rate, evaluate_kernel, n_iterations,
    compute_log_likelihood=compute_log_likelihood, report_progress=report_progress, do_newton=do_newton, verbose=True)

  samples = project_params(particles)
  plot(samples)

  real_mean = tf.math.reduce_sum(sample_evaluations[:, None] * uniform_samples, axis=0)
  #real_var = tf.math.reduce_variance(tf.math.exp(sample_evaluations), axis=0)

from multivariate import MultivariateProcess

hierarchical_process = MultivariateProcess.Parameters( # row -> column
  tf.constant([
    [0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0],
    [0.0, 0.0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]),
  tf.constant(1.0),
  tf.constant([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
  tf.constant(0.01),
  tf.constant([0.05]*8),
  tf.constant(0.0)
)

def simulate_hierarchy(n_events=0, n_throwaway=100, output_filename="hierarchical_process.msgpack"): # ensure n_events is an integer
  assert n_events > 0
  event_types, event_times = MultivariateProcess.simulate_events_memoryless(hierarchical_process, n_events, n_throwaway=n_throwaway)
  with open(output_filename, "wb") as f:
    msgpack.dump([event_types.numpy(), event_times.numpy()], f)

argh.dispatch_commands([test_stein, simulate_hierarchy])
