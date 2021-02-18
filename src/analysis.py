import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
import argh
import pickle as pkl
import msgpack # import from Julia simulations
import msgpack_numpy
msgpack_numpy.patch()
import json
import datetime as dt
from os import path
from glob import glob
from scipy.stats.mstats import mquantiles # more granular than np.quantile
if path.isfile("../style.mplstyle"):
  from matplotlib import pyplot as plt
  from matplotlib import colors as clrs
  plt.style.use("../style.mplstyle")
  plt.ion()
  import seaborn as sb
  from cmcrameri import cm
  import pygraphviz as pgv

from multivariate import MultivariateProcess

from scipy.signal import welch

def load(filename):
  filetype = filename.split(".")[-1]
  with open(filename, "rb") as f:
    if filetype == "msgpack":
      return msgpack.load(f)
    elif filetype == "pkl":
      return pkl.load(f)
    else:
      raise ValueError("File type not recognized.")

def load_all_suffixes(filename_prefix):
    matches = sorted(glob("{}*".format(filename_prefix)))
    keys = [key if key[0] != '_' else key[1:] for key in [
      match.split(filename_prefix)[-1].split('.')[0] for match in matches]]
    return {key: load(match) for key, match in zip(keys, matches)}

def trim_sample(samples, serialize=False):
  if serialize:
    return MultivariateProcess.Parameters(
      samples.excitation_coef, samples.excitation_rate[..., 0, 0],
      samples.suppression_coef, samples.suppression_rate[..., 0, 0],
      samples.background, samples.lag[..., 0])
  else:
    return MultivariateProcess.Parameters(
      samples["excitation_coef"], samples["excitation_rate"][..., 0, 0],
      samples["suppression_coef"], samples["suppression_rate"][..., 0, 0],
      samples["background"], samples["lag"][..., 0])

def select_sample(samples, p, serialize=False):
  if serialize:
    return MultivariateProcess.Parameters(
      samples.excitation_coef[p, ...], samples.excitation_rate[p, ...],
      samples.suppression_coef[p, ...], samples.suppression_rate[p, ...],
      samples.background[p, ...], samples.lag[p, ...])
  else:
    return MultivariateProcess.Parameters(
      samples["excitation_coef"][p, ...], samples["excitation_rate"][p, ...],
      samples["suppression_coef"][p, ...], samples["suppression_rate"][p, ...],
      samples["background"][p, ...], samples["lag"][p, ...])

def trim_and_select_sample(samples, p, serialize=False):
  return trim_sample(select_sample(samples, p, serialize), serialize)# designed so that order doesn't matter

@tf.function
def compute_intensities(params, event_types, event_times, batch_size, eval_batch_size, resolution, verbose=False): # resolution is total grid points across entire record
  leftover_excitation = tf.zeros_like(params.excitation_coef)
  leftover_suppression = tf.zeros_like(params.excitation_coef) # this is of the correctly expanded size
  time_of_leftover = event_times[0]
  all_intensities, all_excitations, all_suppressions = [], [], []
  all_eval_times = []
  for t in range(0, len(event_times), batch_size): # not dynamic because of list appending
    batch_types = event_types[t:(t+batch_size+1)]
    batch_times = event_times[t:(t+batch_size+1)]
    n_events_in_batch = tf.cast(len(batch_times) - 1, event_times.dtype)
    eval_length = tf.cast(resolution * (batch_times[-1] - batch_times[0]) / (event_times[-1] - event_times[0]), tf.int32)
    times = tf.linspace(batch_times[0], batch_times[-1], eval_length+1)[:-1]
    all_eval_times.append(times)
    param_shape = tf.shape(params.excitation_coef)[:-2]
    shape = tf.concat([param_shape, [params.n_dims, eval_length]], axis=0)
    intensity = tf.zeros(shape, dtype=event_times.dtype)
    excitation = tf.zeros(shape, dtype=event_times.dtype)
    suppression = tf.zeros(shape, dtype=event_times.dtype)
    next_leftover_excitation = tf.zeros_like(leftover_excitation)
    next_leftover_suppression = tf.zeros_like(leftover_suppression)
    next_time_of_leftover = tf.zeros_like(time_of_leftover)
    for event_type in tf.range(params.n_dims):
      this_shape = tf.concat([param_shape, [eval_length]], axis=0)
      type_intensity = tf.zeros(this_shape, dtype=event_times.dtype)
      type_excitation = tf.zeros(this_shape, dtype=event_times.dtype)
      type_suppression = tf.zeros(this_shape, dtype=event_times.dtype)
      for eval_t in tf.range(0, eval_length, eval_batch_size): # run over same events multiple times to get different parts of the number line
        sub_times = times[eval_t:(eval_t+eval_batch_size)]
        types = tf.fill(tf.shape(sub_times), tf.cast(event_type, batch_types.dtype))
        sub_type_intensity, next_leftover_excitation, next_leftover_suppression, next_time_of_leftover = MultivariateProcess\
          .infer_intensity(params, batch_types, batch_times, types, sub_times, leftover_excitation, leftover_suppression, time_of_leftover)
        sub_type_excitations, _, _ = MultivariateProcess\
          .infer_excitation(params, batch_types, batch_times, types, sub_times, leftover_excitation, time_of_leftover)
        sub_type_suppressions, _, _ = MultivariateProcess\
          .infer_suppression(params, batch_types, batch_times, types, sub_times, leftover_suppression, time_of_leftover)
        sub_type_excitation = tf.math.reduce_sum(sub_type_excitations, axis=-2)
        sub_type_suppression = tf.math.reduce_sum(sub_type_suppressions, axis=-2)
        updates = tf.range(eval_t, tf.minimum(eval_t+eval_batch_size, eval_length))[:, None]
        type_intensity = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(type_intensity), updates, tf.transpose(sub_type_intensity)))
        type_excitation = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(type_excitation), updates, tf.transpose(sub_type_excitation)))
        type_suppression = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(type_suppression), updates, tf.transpose(sub_type_suppression))) # transposes are because tensor_scatter_update allows bulk assignment in the last dims, which we need to make the parameter batches
      selection = tf.one_hot(event_type, params.n_dims, dtype=event_times.dtype)
      intensity += type_intensity[..., None, :] * selection[:, None]
      excitation += type_excitation[..., None, :] * selection[:, None]
      suppression += type_suppression[..., None, :] * selection[:, None] # to allow parameter batching as well
    all_intensities.append(intensity)
    all_excitations.append(excitation)
    all_suppressions.append(suppression)
    leftover_excitation, leftover_suppression, time_of_leftover = next_leftover_excitation, next_leftover_suppression, next_time_of_leftover
    if verbose: tf.print("Completed computation between event times", batch_times[0], "and", batch_times[-1], "with", eval_length, "points, as we approach the finish line of", event_times[-1], "\b.")
  return tf.concat(all_intensities, axis=-1), tf.concat(all_excitations, axis=-1), tf.concat(all_suppressions, axis=-1), tf.concat(all_eval_times, axis=-1)

def obtain_interarrival_intensities(params, event_types, event_times, batch_size, eval_batch_size, resolution, verbose=False): # Instead of looking for t~Exp(1) on each process separately, we look at the aggregated point process as a whole.
  intensities, _, _, times = compute_intensities(params, event_types, event_times, batch_size, eval_batch_size, resolution, verbose)
  labels = np.digitize(times, event_times, False)
  total_intensities = np.sum(intensities, axis=-2)
  intensity_partitions = tf.dynamic_partition(tf.transpose(total_intensities), labels - 1, len(event_times) - 1) # transposed
  dt = times[1] - times[0] # should be the same everywhere
  integrals = np.array([np.sum(partition, axis=0)*dt for partition in intensity_partitions]).T
  return integrals

def get_maximal_likelihood_process(data):
  i = np.argmax(data["test_scores"])
  return MultivariateProcess.deserialize({k: v[i, ...] for k, v in data['sampled_processes'].items()})

@tf.function
def compute_categorical_accuracy(params, event_types, event_times, batch_size, intensity_floor=1e-25): # can (should) be batched by parameters. returns categorical cross-entropy score on event types for each process in the sample, all of which can then be characterized statistically
  if not isinstance(event_types, list):
    event_types, event_times = [event_types], [event_times]
  all_types = tf.range(params.n_dims, dtype=event_types[0].dtype)
  total_score = tf.zeros(tf.shape(params.excitation_coef)[:-2], dtype=MultivariateProcess.dtype)
  for evt_types, evt_times in zip(event_types, event_times):
    event_lags = MultivariateProcess.gather_background(params.lag, evt_types)
    max_event_lag = tf.math.reduce_max(event_lags)
    leftover_excitation = tf.zeros_like(params.excitation_coef)
    leftover_suppression = tf.zeros_like(params.excitation_coef)
    time_of_leftover = evt_times[0]
    for t in range(0, len(evt_times), batch_size):
      batch_types = evt_types[t:(t+batch_size+1)]
      batch_times = evt_times[t:(t+batch_size+1)]
      n_events_in_batch = tf.size(batch_times) - 1
      time_epsilon = 1e-20
      #eval_times = (batch_times + max_event_lag) * (1.0 + time_epsilon) # taken from compute_excess_suppression_penalty(...) Do I evaluate at the next event or right after the previous? # Keep in mind the possibility of duplicate events, and tiny intervals in relation to the absolute units utilized.
      #flat_eval_types = tf.reshape(tf.broadcast_to(all_types[None, :], [n_events_in_batch, params.n_dims]), [n_events_in_batch * params.n_dims])
      #flat_eval_times = tf.reshape(tf.broadcast_to(eval_times[:-1, None], [n_events_in_batch, params.n_dims]), [n_events_in_batch * params.n_dims])
      #flat_intensity, next_leftover_excitation, next_leftover_suppression, next_time_of_leftover = MultivariateProcess.infer_intensity(
      #  params, batch_types, batch_times, flat_eval_types, flat_eval_times, leftover_excitation, leftover_suppression, time_of_leftover)
      #intensity = tf.reshape(flat_intensity, tf.concat([tf.shape(flat_intensity)[:-1], [n_events_in_batch, params.n_dims]], axis=0)) # params_batch x events x types
      #total_intensity = tf.math.reduce_sum(intensity, axis=-1)
      #broad_batch_types = tf.broadcast_to(batch_types, tf.concat([tf.shape(intensity)[:-2], [tf.size(batch_types)]], axis=0)) # expand the batch dims to match up. don't cut off the last entry because we want to be one step ahead. batches having one extra entry saves us from having to stagger intermediate values between them!
      #target_intensity = tf.gather(intensity, broad_batch_types[..., 1:][..., None],
      #  axis=-1, batch_dims=len(tf.shape(intensity)[:-1]))[..., 0] # select along the last axis one element (type) for each event occurrence. batch_types is batched along the events. match each event to the starting intensities at the previous.
      #accuracies = tf.math.log(target_intensity) - tf.math.log(total_intensity)
      #total_score += tf.math.reduce_sum(accuracies, axis=-1)
      eval_times = batch_times[:-1] # for reasons highlighted above, I shall evaluate each prediction *at* the very occurrence of the next event
      flat_eval_types = tf.reshape(tf.broadcast_to(all_types[None, :], [n_events_in_batch, params.n_dims]), [n_events_in_batch * params.n_dims])
      flat_eval_times = tf.reshape(tf.broadcast_to(eval_times[:, None], [n_events_in_batch, params.n_dims]), [n_events_in_batch * params.n_dims])
      flat_intensity, next_leftover_excitation, next_leftover_suppression, next_time_of_leftover = MultivariateProcess.infer_intensity(
        params, batch_types, batch_times, flat_eval_types, flat_eval_times, leftover_excitation, leftover_suppression, time_of_leftover)
      intensity = tf.reshape(flat_intensity, tf.concat([tf.shape(flat_intensity)[:-1], [n_events_in_batch, params.n_dims]], axis=0)) # params_batch x events x types
      total_intensity = tf.math.reduce_sum(intensity, axis=-1)
      broad_batch_types = tf.broadcast_to(batch_types, tf.concat([tf.shape(intensity)[:-2], [tf.size(batch_types)]], axis=0)) # expand the batch dims to match up. don't cut off the last entry because we want to be one step ahead. batches having one extra entry saves us from having to stagger intermediate values between them!
      target_intensity = tf.gather(intensity, broad_batch_types[..., :-1][..., None],
        axis=-1, batch_dims=len(tf.shape(intensity)[:-1]))[..., 0] # select along the last axis one element (type) for each event occurrence. batch_types is batched along the events. match each event to the starting intensities at the previous.
      accuracies = tf.math.log(tf.math.maximum(target_intensity, intensity_floor)) - tf.math.log(total_intensity) # intensity_floor gives a leg up to the occasional white process that predicts 0% chance for some event and would otherwise receive a negatively infinite score
      total_score += tf.math.reduce_sum(accuracies, axis=-1)
      leftover_excitation, leftover_suppression, time_of_leftover = next_leftover_excitation, next_leftover_suppression, next_time_of_leftover
  mean_score = total_score / tf.cast(sum(len(evt_types) for evt_types in event_types), total_score.dtype) # divide by total length
  return tf.math.exp(mean_score)

# One possible thing to do would be to examine the coherence between the intensity in the test events and the intensity in events simulated purely out of the model.
# Also, use the following syntax to pick out a parameter sample i: MultivariateProcess.deserialize({k: v[i,...] for k,v in res['samples'].items()})

# I will not employ np.quantile here because the parameter sample size is small (but of utmost quality) and I need granular control of rounding
# Event-type names can overlap. If they do, they will be drawn as the same node with possibly many edges emanating and incoming
def render_interaction_graph(params, filename, event_type_names, credible_quantile, excitation_threshold, suppression_threshold,
    event_type_color_categories=None, show_names=True, verbose=True):
  import seaborn as sb # in case it hasn't already
  graph = pgv.AGraph(strict=False, directed=True)
  graph.graph_attr["size"] = (2, 4)
  graph.graph_attr["rankdir"] = "TB" # try to make arrows travel from top to bottom
  graph.edge_attr["arrowhead"] = "open"
  if event_type_color_categories is not None:
    graph.node_attr["style"] = "filled"
    distinct_categories = np.unique(event_type_color_categories)
    palette = ["#{0:02x}{1:02x}{2:02x}".format(int(r*255), int(g*255), int(b*255))
      for (r, g, b) in sb.husl_palette(len(distinct_categories), 0.01, 0.95, 0.6)]
  n_event_types = len(event_type_names)
  assert n_event_types == params.n_dims
  n_samples = params.excitation_coef.shape[0]
  assert n_samples == params.suppression_coef.shape[0]
  def set_color(name, category): # use the node's category as its basis for coloring
    node = graph.get_node(name)
    category_index = np.nonzero(category == distinct_categories)[0][0]
    node.attr["fillcolor"] = palette[category_index]
    node.attr["group"] = category_index
    if show_names:
      node.attr["fontsize"] = 5 # make it tiny; for the sake of exploration only
      node.attr["fontname"] = "Helvetica Neue"
    else:
      node.attr["label"] = ""
  for rec in range(n_event_types): # efficiency does not really matter. aim for legibility
    for inf in range(n_event_types):
      excitation = params.excitation_coef[..., inf, rec] # do I really care for the case with two Tensor dimensions (no batching)?
      suppression = params.suppression_coef[..., inf, rec]
      excitation_quantile = mquantiles(excitation, credible_quantile, 1/3, 1/3)
      suppression_quantile = mquantiles(suppression, credible_quantile, 1/3, 1/3)
      included = False
      if excitation_quantile >= excitation_threshold:
        graph.add_edge(event_type_names[inf], event_type_names[rec], color="blue")
        included = True
      if suppression_quantile >= suppression_threshold:
        graph.add_edge(event_type_names[inf], event_type_names[rec], color="black")
        included = True
      if included and event_type_color_categories is not None:
        set_color(event_type_names[inf], event_type_color_categories[inf])
        set_color(event_type_names[rec], event_type_color_categories[rec])
  graph.draw(filename, prog="dot", args="-Gdpi=700")

# The above tries to make sense of marginal posteriors. Also do entropies of discrete QoIs (quantities of interest)?
