import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
import argh
import pickle as pkl
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import json
import datetime as dt
from os import path
from glob import glob
if path.isfile("../style.mplstyle"):
  from matplotlib import pyplot as plt
  from matplotlib import colors as clrs
  plt.style.use("../style.mplstyle")
  plt.ion()
  import seaborn as sb
  from cmcrameri import cm

from multivariate import MultivariateProcess
import analysis as an
import experiment as ex

@tf.function
def chop_record_by_time_interval(event_types, event_times, interval): # can we do this dynamically instead of relying on a static list? RaggedTensor *combined with* a variable axis size
  ref_times = event_times - event_times[0]
  interval_multiples = tf.math.floor(ref_times / interval)
  partitions = tf.cast(interval_multiples, tf.int32) # does not have to correspond to any other part of the multi-tentacled data structure
  n_partitions = partitions[-1] + 1
  interval_event_types = tf.ragged.stack_dynamic_partitions(event_types, partitions, n_partitions) # n_intervals x events
  interval_event_times = tf.ragged.stack_dynamic_partitions(event_times, partitions, n_partitions)
  return interval_event_types[:-1, :], interval_event_times[:-1, :], n_partitions-1 # throw away the last interval, which may have a smaller size (might rely on fixed windows in our metrics down the line)

@tf.function #                         <history record---------------------><forecast----------------------------------->
def forecast_window_aggregates(params, event_types, event_times, batch_size, window, n_events, n_simulations, sub_windows,
    warm_start_excitation=None, warm_start_suppression=None, time_of_warm_start=None, return_leftovers=False): # could be multiple processes, over which we will take an expectation. window is in real time units. hope that n_events generally is enough to exceed the window interval
  if warm_start_excitation is None:
    leftover_excitation = tf.zeros_like(params.excitation_coef)
    leftover_suppression = tf.zeros_like(params.suppression_coef)
    time_of_leftover = event_times[0]
  else:
    leftover_excitation = warm_start_excitation
    leftover_suppression = warm_start_suppression
    time_of_leftover = time_of_warm_start
  padded_event_types, padded_event_times = MultivariateProcess.append_terminal_event_for_batching(params, event_types, event_times)
  for t in tf.range(0, tf.shape(event_times)[-1], batch_size): # fast-forward
    batch_types = padded_event_types[t:(t+batch_size+1)]
    batch_times = padded_event_times[t:(t+batch_size+1)] # go up 'til the very end
    if tf.size(batch_types) <= 1:
      continue # if we happen to have nothing in the 'prelude,' OR we are smack-dab at the terminal fake padded event
    eval_types, eval_times = batch_types[:1], batch_times[:1] # tiny non-vacuous and non-nonsensical (not necessarily sensical, however) vectors
    response, leftover_excitation, leftover_suppression, time_of_leftover = MultivariateProcess\
      .infer_response(params, batch_types, batch_times, eval_types, eval_times, leftover_excitation, leftover_suppression, time_of_leftover)
  if tf.size(event_times) == 0:
    forecast_beginning = time_of_leftover # note that if there are no events on which to fast-forward, we start off forecasting on the basis of time_of_leftover, which is not entirely correct: it may be *slightly* after the actual occurrence of the last event
  else:
    forecast_beginning = event_times[-1]
    rewind = padded_event_times[-1] - event_times[-1] # a miniscule overstep, but one that shold be rectified for absolute correctness
    leftover_excitation *= tf.math.exp(params.excitation_rate * rewind)
    leftover_suppression *= tf.math.exp(params.suppression_rate * rewind)
  if len(params.excitation_coef.shape) == 2:
    processes = [an.trim_sample(params, True)]
    process_excitations = [leftover_excitation]
    process_suppressions = [leftover_suppression]
  else:
    n_processes = params.excitation_coef.shape[0]
    processes = (an.trim_and_select_sample(params, p, True) for p in range(n_processes))
    process_excitations = (leftover_excitation[p, ...] for p in range(n_processes))
    process_suppressions = (leftover_suppression[p, ...] for p in range(n_processes))
  n_splits, = sub_windows.shape # one dimension only
  aggregates = []
  max_n_events_counted = 0
  for process, excitation, suppression in zip(processes, process_excitations, process_suppressions):
    process_aggregates = tf.zeros([params.n_dims, n_splits], dtype=event_times.dtype)
    for sim in tf.range(n_simulations):
      sim_types, sim_times = MultivariateProcess\
        .simulate_events_memoryless(process, n_events, warm_excitations=excitation, warm_suppressions=suppression)
      window_filter = sim_times <= window # no need to add forecast_beginning because simulation always starts at 0
      window_types = tf.where(window_filter, sim_types, -tf.ones([], dtype=sim_types.dtype)) # -1 marks events that are thrown away
      window_assignments = tf.searchsorted(sub_windows, sim_times) # len(sim_times)-vector of integers from [0, n_splits], where the last value means out-of-bounds
      instances = tf.math.equal(window_types, tf.range(params.n_dims, dtype=window_types.dtype)[:, None]) # (n_types x window events)
      aggregate = tf.math.count_nonzero(instances, axis=1, dtype=event_times.dtype) # this one is not really needed anymore
      instance_indicators = tf.transpose(tf.cast(instances, tf.int32)) # (window events x n_types)
      split_aggregates = tf.math.segment_sum( # ASSIGNMENTS NEED TO BE SORTED
        tf.concat([instance_indicators, tf.broadcast_to([[0]], [1, params.n_dims])], axis=0),
        tf.concat([window_assignments, [n_splits]], axis=0) ) # (n_splits+1 x n_types). append a "guard" element to ensure consistent sizes
      # see tf.math.bincount as well (which only accepts non-negative integer arrays), or tf.math.segment_sum that operates on a tensor's first dimension..
      n_events_counted = tf.cast(tf.math.reduce_sum(aggregate, axis=0), tf.int32)
      if n_events_counted > max_n_events_counted: # does this serial dependence hinder parallelization?
        max_n_events_counted = n_events_counted
      process_aggregates += tf.cast(tf.transpose(split_aggregates[:-1, :]), event_times.dtype)
      process_aggregates = tf.ensure_shape(process_aggregates, [None, n_splits])
    aggregates.append(process_aggregates / tf.cast(n_simulations, event_times.dtype))
  aggregates = tf.convert_to_tensor(aggregates)
  if n_splits == 1:
    aggregates = aggregates[..., 0]
  if return_leftovers:
    return aggregates, max_n_events_counted, leftover_excitation, leftover_suppression, time_of_leftover
  return aggregates, max_n_events_counted

@tf.function
def backtest_rolling_forecast_aggregates(params, historic_types, historic_times, event_types, event_times,
    batch_size, window, n_events, n_simulations, score_against, score_shape=[], pred_window_multiple=1, n_splits=1): # our time step is the same as the window size. params may contain multiple processes once again
  interval_event_types, interval_event_times, n_intervals = chop_record_by_time_interval(event_types, event_times, window)
  leftover_excitation = tf.zeros_like(params.excitation_coef)
  leftover_suppression = tf.zeros_like(params.suppression_coef)
  time_of_leftover = historic_times[0]
  n_evaluations = (n_intervals - 1 if historic_types is None else n_intervals) - pred_window_multiple + 1 # if we are given (historic_types, historic_times), then we may evaluate every single interval by kickstarting on e.g. the training set. Helps especially when the test set is tiny.
  n_processes = params.excitation_coef.shape[0] if len(params.excitation_coef.shape) == 3 else 1 # static version of `tf.rank`
  scores = tf.zeros([n_evaluations, n_processes] + list(score_shape), dtype=event_times.dtype)
  anchor_time = tf.cast(0, dtype=event_times.dtype) # should really be set with the first interval, otherwise might produce garbage until changed
  prev_hard_window = tf.cast(window, event_times.dtype)
  for interval in tf.range(n_evaluations): # all completely dynamic. funny that this is a desideratum now that static unrolling has become a bottleneck, contraty to how it is in most programming environments!
    if historic_times is None:
      prelude_types, prelude_times = interval_event_types[interval, ...], interval_event_times[interval, ...]
    else:
      if interval == 0:
        prelude_types, prelude_times = historic_types, historic_times
      else:
        prelude_types, prelude_times = interval_event_types[interval-1, ...], interval_event_times[interval-1, ...]
    if tf.size(prelude_times) == 0:
      hard_window = prev_hard_window
    else:
      anchor_time = prelude_times[-1] # basically `prelude_times[-1]`, with the added functionality of tracking the last non-empty one
      # this shouldn't be needed now that the last interval is thrown away anyways. keep it as a failsafe
      hard_window = tf.math.minimum(event_times[-1] - anchor_time, window * pred_window_multiple) # if we are at the end of the record, our last interval must be subjected to a hard cutoff. pred_window_multiple allows us to achieve a crude version of overlapping sliding windows: make the intervals small, then forecast over a collection of them.
    sub_windows = tf.linspace(tf.cast(0, event_times.dtype), hard_window, n_splits+1)[1:]
    aggregates, max_n_events_counted, leftover_excitation, leftover_suppression, time_of_leftover = \
      forecast_window_aggregates(params, prelude_types, prelude_times, batch_size, hard_window, n_events, n_simulations, sub_windows,
        warm_start_excitation=leftover_excitation, warm_start_suppression=leftover_suppression, time_of_warm_start=time_of_leftover, return_leftovers=True)
    tf.print("Counted at most", max_n_events_counted, "simulated events out of", n_events, "in interval", interval+1, "out of", n_evaluations, "\b.")
    real_deal_intervals = interval_event_types[(interval+1):(interval+1+pred_window_multiple), ...] if historic_times is None \
      else interval_event_types[interval:(interval+pred_window_multiple), ...]
    real_deal_types = tf.concat([real_deal_intervals[i, ...] for i in range(pred_window_multiple)], axis=0) # how is it so hard to unstack and flatten a ragged tensor?
    real_deal_time_intervals = interval_event_times[(interval+1):(interval+1+pred_window_multiple), ...] if historic_times is None \
      else interval_event_times[interval:(interval+pred_window_multiple), ...]
    real_deal_times = tf.concat([real_deal_time_intervals[i, ...] for i in range(pred_window_multiple)], axis=0)
    real_instances = tf.math.equal(real_deal_types, tf.range(params.n_dims, dtype=real_deal_types.dtype)[:, None]) # (n_types x window events)
    if n_splits == 1:
      real_aggregates = tf.math.count_nonzero(real_instances, axis=1, dtype=event_times.dtype)
    else:
      real_instance_indicators = tf.transpose(tf.cast(real_instances, tf.int32)) # (window events x n_types)
      real_assignments = tf.searchsorted(sub_windows, real_deal_times - anchor_time) # shift the frame of reference
      split_real_aggregates = tf.math.segment_sum( # see `forecast_window_aggregates` for explanation
        tf.concat([real_instance_indicators, tf.broadcast_to([[0]], [1, params.n_dims])], axis=0),
        tf.concat([real_assignments, [n_splits]], axis=0) ) # (n_splits+1 x n_types)
      real_aggregates = tf.cast(tf.transpose(split_real_aggregates[:-1, :]), event_times.dtype) # touched-up/refined version of the above
    interval_scores = score_against(aggregates, real_aggregates)# real_aggregate if n_splits==1 else real_aggregates -- for some reason this is not treated purely like a static evaluation.. must be something with the decorator
    # ^ must return a tensor with one entry (row? hyper-row?) for each process
    selector = tf.reshape(tf.one_hot(interval, n_evaluations, dtype=scores.dtype), [-1, 1] + [1]*len(score_shape))
    scores += tf.ensure_shape(interval_scores, [n_processes] + score_shape) * selector
    prev_hard_window = hard_window
  return scores

@tf.function
def get_top_event_types(aggregates, cutoff_ranks): # batched along the first dimension
  assert aggregates.dtype not in (tf.int32, tf.int64) # can't be integers because it messes up our rounding
  n_quantiles = len(cutoff_ranks)
  n_processes, n_types = tf.shape(aggregates)[0], tf.shape(aggregates)[1] # can't unravel implicitly because it's a tensor
  sorted_aggregates_indices = tf.argsort(aggregates, axis=1, direction="DESCENDING") # I never end up using the indices themselves
  sorted_aggregates = tf.gather(aggregates, sorted_aggregates_indices, axis=1, batch_dims=1)
  cutoffs = tf.gather(sorted_aggregates, tf.tile(cutoff_ranks[None, :], [n_processes, 1]), axis=1, batch_dims=1) # processes x aggregates ==> processes x quantiles
  is_top = tf.where(aggregates[:, None, :] >= cutoffs[:, :, None], 1, 0) # processes x quantiles x aggregates
  is_top = tf.where(aggregates[:, None, :] > 0.0, is_top, 0) # if there are absolutely no occurrences (most likely happening in the singular ground truth,) truncate those as well
  select = lambda it: tf.dynamic_partition(tf.range(n_types), it, 2)[1] # the dtype of the range doesn't matter, since it's used internally for counting purposes
  selection = tf.map_fn(lambda x: tf.map_fn(select, x, fn_output_signature=tf.RaggedTensorSpec([None], tf.int32)),
    is_top, fn_output_signature=tf.RaggedTensorSpec([n_quantiles, None], tf.int32)) # Insanely complex. I wonder if it could be simplified vastly. Seems to work though...
  return selection # it seems there is a fundamental limitation to the way ragged tensors are represented that forces the quantiles dimension to be None too, as it's stacked along map_fn. relative ordering, i.e. importance of inner ragged dimensions versus outer? # above: why not have a batched dynamic_partition?


def score_precision_and_recall(cutoff_ranks, real_cutoff_ranks): # returns a closure and output shape to be fed into the above method and compiled into the computation graph. quantiles and real_quantiles must be vectors of the same size. for classical precision and recall analysis, set real_quantiles to a homogeneous value greater than quantiles (so the max precision is 1.0) and see how the predictor's ranked recommendation fares with that. also, small real_quantiles are just as interesting
  n_quantiles = len(cutoff_ranks)
  assert len(real_cutoff_ranks) == n_quantiles
  def eval(aggregates, real_aggregate):
    n_processes, n_types = tf.shape(aggregates)[0], tf.shape(aggregates)[1]
    #cutoff_ranks = tf.cast(tf.math.floor(tf.cast(quantiles, aggregates.dtype) * tf.cast(n_types-1, aggregates.dtype)), tf.int32)
    #real_cutoff_ranks = tf.cast(tf.math.round(tf.cast(real_quantiles, aggregates.dtype) * tf.cast(n_types-1, aggregates.dtype)), tf.int32)
    top_types = get_top_event_types(aggregates, tf.cast(cutoff_ranks, tf.int32)) # processes x quantiles x event_types
    top_real_types = get_top_event_types(real_aggregate[None, :], tf.cast(real_cutoff_ranks, tf.int32)) # both have the same number of types under each quantile, so 'precision' and 'recall' mean the same thing here. this is because our ground truth varies as well, unlike in diagnostic tests and whatnot.
    n_top_types_in_common = tf.map_fn(
      lambda a: tf.map_fn( # foreshadowing that the dimensions will vary from call to call
        lambda b: tf.math.count_nonzero(
            tf.math.equal(b[0][None, :], b[1][:, None]), axis=[0, 1], dtype=aggregates.dtype), # collapse into a normal tensor # tf.sets.intersection(top_types, top_real_types) can't do ragged tensors... # processes x quantiles x event_types_in_common
        (a, top_real_types[0, ...]), fn_output_signature=tf.TensorSpec([], aggregates.dtype)),
      top_types, fn_output_signature=tf.TensorSpec([n_quantiles], aggregates.dtype))
    n_top_types = tf.cast(top_types.row_lengths(axis=2).to_tensor(), aggregates.dtype) #(tf.cast(cutoff_ranks, aggregates.dtype) + 1.0)
    precision = tf.where(tf.broadcast_to(n_top_types, tf.shape(n_top_types_in_common)) == 0.0,
      tf.ones_like(n_top_types_in_common), n_top_types_in_common / n_top_types) # can it be that we predict *absolutely* nothing? maybe on tiny sequences...
    n_top_real_types = tf.cast(top_real_types.row_lengths(axis=2).to_tensor(), aggregates.dtype)
    recall = tf.where(tf.broadcast_to(n_top_real_types, tf.shape(n_top_types_in_common)) == 0.0,
      tf.ones_like(n_top_types_in_common), n_top_types_in_common / n_top_real_types) # fine to compare exact floats since they're cast directly from integers. NaN-dodging trickery that makes recall=1 when there are no true event types: fine to do a tf.where on potential NaNs because we are not differentiating this path.
    return tf.stack([precision, recall], axis=-1) # this is most closely related to mRHR: the modified Reciprocal Hit Rank metric when there are multiple "correct" things... kind of. ground truth still doesn't vary along with the ranked recommendations here.
  return eval, [n_quantiles, 2]

def score_reciprocal_rank(real_cutoff_ranks): # 1/rank, where rank starts at 1, of the first correct finding, i.e. true positive. average that over all ranked forecasts.
  n_scores = len(real_cutoff_ranks)
  def eval(aggregates, real_aggregate): # aggregates is (n_processes x n_types). real_aggregates is (n_types)
    n_processes, n_types = aggregates.shape[0], tf.shape(aggregates)[1] # can't elegantly unpack a Tensor. also inhomogeneous requirements
    sorted_indices = tf.argsort(aggregates, axis=1, direction="DESCENDING")
    update_indices = tf.concat([tf.tile(tf.range(n_processes)[:, None, None], [1, n_types, 1]), sorted_indices[..., None]], axis=2) # concatenate with the batch indexer, so that last dim refers to (n_processes x n_types)
    unfiltered_ranks = tf.scatter_nd(update_indices, tf.tile(tf.range(n_types)[None, :], [n_processes, 1]), [n_processes, n_types]) # scatter, or dynamic_partition (like opposite of gather) followed directly by a reshape (flatten) or concatenation. if only you could conveniently batch the scatter_nd operation, just as you can for tf.gather as of recently. at least the structure of updates can have any dimensionality (only last dim plays a special role)
    ranks = tf.where(aggregates > 0.0, unfiltered_ranks + 1, tf.int32.max)
    true_positives = get_top_event_types(real_aggregate[None, :], tf.cast(real_cutoff_ranks, tf.int32))[0, ...] # n_scores x types (variable size, ragged)
    lowest_correct_ranks = tf.map_fn(
      lambda positives: tf.math.reduce_min(tf.gather(ranks, positives, axis=1), axis=1), # tf.math.reduce_any(tf.equal(ranks[:, :, None], positives[None, :]), axis=2) what was I thinking?
      true_positives, fn_output_signature=tf.TensorSpec([n_processes], tf.int32)) # n_scores x n_processes
    reciprocal_rank = 1.0 / tf.cast(lowest_correct_ranks, aggregates.dtype)
    return tf.transpose(reciprocal_rank) # n_processes x n_scores
  return eval, [n_scores]

def score_rank_biased_overlap(discounts):
  n_scores = len(discounts)
  def eval(aggregates, real_aggregate):
    nonlocal discounts # `global` probably would not do precisely what's intended here
    discounts = tf.cast(discounts, aggregates.dtype)
    n_processes, n_types = aggregates.shape[0], tf.shape(aggregates)[1]
    prediction_indices = tf.argsort(aggregates, axis=1, direction="DESCENDING")
    actual_indices = tf.argsort(real_aggregate, direction="DESCENDING")
    total_scores = tf.zeros([n_processes, n_scores], dtype=aggregates.dtype)
    for size in tf.range(1, n_types+1): # dynamical slicing
      prediction_set = prediction_indices[:, :size]
      actual_set = tf.broadcast_to(actual_indices[:size], tf.shape(prediction_set))
      # did I really find a more chill way that gets away without invoking `get_top_event_types` ? I think the accumulated complexity is a holdover from attempting to use quantiles in place of ranks
      intersection = tf.sets.intersection(prediction_set, actual_set) # returns a sparse tensor with non-null (nonzero?) entries representing the set, and dimensions padded to the longest. was this developed before the invention of ragged tensors?
      commonality = tf.sets.size(intersection) # (n_processes)
      real_size = tf.cast(size, aggregates.dtype)
      agreement = tf.cast(commonality, aggregates.dtype) / real_size
      scores = tf.math.pow(discounts, real_size-1) * agreement[:, None] # (n_processes x n_scores)
      total_scores += tf.ensure_shape(scores, [n_processes, n_scores])
    return (1-discounts) * total_scores
  return eval, [n_scores]

def score_normed_error(power, transform = lambda x: x, n_splits=None):
  def eval(aggregates, real_aggregate):
    nonlocal power
    power = tf.cast(power, aggregates.dtype)
    error = transform(aggregates) - transform(real_aggregate)
    normed_error = tf.math.pow(tf.math.abs(error), power)
    if n_splits is None:
      rank = len(aggregates.shape)
      summation_axes = list(range(1, rank)) # either (1, 2) or (1,) depending on whether a temporal dimension is present
    else:
      summation_axes = 1 # if n_splits is supplied, then only sum over the event types
    mean_normed_error = tf.math.reduce_mean(normed_error, axis=summation_axes)
    return tf.math.pow(mean_normed_error, 1/power)
  shape = [] if n_splits is None else [n_splits]
  return eval, shape

def score_squared_error(transform = lambda x: x, n_splits=None):
  return score_normed_error(2.0, transform, n_splits)

def blacklist_score(types, scorer): # `scorer` is the return value of one of the above
  score, score_shape = scorer
  types = tf.convert_to_tensor(types)
  def eval(aggregates, real_aggregate):
    nonlocal types # not needed, but added just in case
    n_types = tf.shape(aggregates)[1]
    filtered_indices_sparse = tf.sets.difference(tf.range(n_types)[None, :], types[None, :])
    filtered_indices = tf.sort(tf.sparse.to_dense(filtered_indices_sparse)[0, :])
    filtered_aggregates = tf.gather(aggregates, filtered_indices, axis=1)
    filtered_real = tf.gather(real_aggregate, filtered_indices)
    return score(filtered_aggregates, filtered_real)
  return eval, score_shape

# todo various rank-based metrics like normalized discounted cumulative gain
# another todo: learn a MultivariateProcess (excitations only) by means of EM, and compare the results here.

# Kendall's tau per event type, across time (within a mesoscale window)...
