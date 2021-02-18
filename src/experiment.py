import numpy as np
import pandas as pd
import tensorflow as tf
import argh
import pickle as pkl
import msgpack # import from Julia simulations
import msgpack_numpy
msgpack_numpy.patch()
import datetime as dt
from termcolor import colored
from os import path
from glob import glob
if path.isfile("../style.mplstyle"):
  from matplotlib import pyplot as plt
  from matplotlib import colors as clrs
  plt.style.use("../style.mplstyle")
  plt.ion()
  import seaborn as sb
  from cmcrameri import cm

#tf.config.optimizer.set_jit(True) instead of this, set TF_XLA_FLAGS="--tf_xla_cpu_global_jit --tf_xla_auto_jit=2" on the command line. the first argument enables XLA for the CPU in addition to the default for GPU. With TF 2.4, it looks like we need to also set --tf_xla_enable_xla_devices to allow the possibility of XLA at all. Also, that version comes with MLIR that is somehow different (and works together with?) XLA?
# (A word of caution) BE WARY: I've noticed that in some instances, like when --do-newton is employed, the options TF_XLA_FLAGS="--tf_xla_cpu_global_jit --tf_xla_auto_jit=2 --tf_xla_enable_xla_devices" lead to the proliferation of NaNs. (TF 2.4 nightly now)
# Additionally, --do-newton is much more performant on CPU-only machines. I wonder how the bandwidth selector fares outside of the GPU execution model...

from multivariate import MultivariateProcess
import analysis

dtype = MultivariateProcess.dtype

def generate_synthetic_process(n_event_types, excitation_coef_mean, excitation_rate_alpha, excitation_rate_beta,
                               suppression_coef_mean, suppression_rate_alpha, suppression_rate_beta, background_mean, cross_suppressions, n_particles=None): # make it unlikely, but still allow, faster suppressions than excitations. study these behaviors ex post facto
  batch_dim = [] if n_particles is None else [n_particles]
  excitation_coef = tf.random.gamma(shape=(batch_dim + [n_event_types, n_event_types]), alpha=1.0, beta=1/excitation_coef_mean, dtype=dtype) # exponential variate
  excitation_rate = tf.random.gamma(shape=(batch_dim + [1, 1]), alpha=excitation_rate_alpha, beta=excitation_rate_beta, dtype=dtype)
  if not cross_suppressions:
    suppression_coef = tf.linalg.diag(
      tf.random.gamma(shape=(batch_dim + [n_event_types]), alpha=1.0, beta=1/suppression_coef_mean, dtype=dtype) if suppression_coef_mean > 1e-10 else \
      tf.zeros(batch_dim + [n_event_types], dtype=dtype)) # DON'T SCALE because these parameters, as I've chosen them, should remain independent
  else:
    suppression_coef = \
      tf.random.gamma(shape=(batch_dim + [n_event_types, n_event_types]), alpha=1.0, beta=1/suppression_coef_mean, dtype=dtype) if suppression_coef_mean > 1e-10 else \
      tf.zeros(batch_dim + [n_event_types, n_event_types], dtype=dtype)
  suppression_rate = tf.random.gamma(shape=(batch_dim + [1, 1]), alpha=suppression_rate_alpha, beta=suppression_rate_beta, dtype=dtype)
  if tf.rank(background_mean) == 0:
    background = tf.random.gamma(shape=(batch_dim + [n_event_types]), alpha=1.0, beta=1/background_mean, dtype=dtype)
  else:
    background = tf.random.gamma(shape=batch_dim, alpha=1.0, beta=1/background_mean, dtype=dtype)
  return MultivariateProcess.Parameters(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background)

def generate_hawkes_ground_truth(n_event_types, n_particles=None):
  return generate_synthetic_process(n_event_types, 0.5 / n_event_types, 3.0, 3.0, 0.0,  3.0, 3000.0, 0.001, n_particles=n_particles) # hope the spectral radius of the excitations is less than 1. could compute the probability of such an occurrence analytically or numerically
  # with four event types, mean excitation=0.11 led to fewer than 0.7% of the models to be unstable. the cool thing is that n_event_types increases, the distribution of spectral radii becomes narrower, firmly below 1.0

def generate_ground_truth(n_event_types, n_particles=None, cross_suppressions=False, mean_arrival_time=None, mean_rates=None, initial_excitation=None):
  if mean_arrival_time is not None:
    return generate_synthetic_process(n_event_types,
      initial_excitation, 3.0, 3.0*mean_arrival_time, initial_excitation*mean_arrival_time, 3.0, 3000.0*mean_arrival_time, mean_rates, cross_suppressions, n_particles)
  else:
    assert initial_excitation is None or initial_excitation == 1.5
    return generate_synthetic_process(n_event_types, 1.5 / n_event_types, 3.0, 3.0, 50.0 / n_event_types, 3.0, 3000.0, 0.001, cross_suppressions, n_particles)

def generate_initial_parameters(n_particles, n_event_types, cross_suppressions=False, mean_arrival_time=None, mean_rates=None, initial_excitation=None): # no hyperparameters employed here anymore
  params = generate_ground_truth(n_event_types, n_particles, cross_suppressions, mean_arrival_time, mean_rates, initial_excitation)
  collapsed_params = MultivariateProcess.Parameters(params.excitation_coef, params.excitation_rate[:, 0, 0],
    tf.linalg.diag_part(params.suppression_coef) if not cross_suppressions else params.suppression_coef, params.suppression_rate[:, 0, 0],
    params.background)
  return collapsed_params

def generate_initial_parameters_lognormal(n_particles, n_event_types, rate_sigma): # deprecated in favor of a reference distribution harmonized with the ground-truth parametrization
  supp = tf.exp(tf.random.normal([n_particles], dtype=dtype) * rate_sigma) / 10
  return MultivariateProcess.Parameters(tf.exp(tf.random.normal([n_particles, n_event_types, n_event_types], dtype=dtype)),
  tf.exp(tf.random.normal([n_particles], dtype=dtype) * rate_sigma), tf.fill([n_particles, n_event_types], tf.cast(0.001, dtype)) / supp[:, None],
  supp, tf.exp(tf.random.normal([n_particles, n_event_types], dtype=dtype)) / 1000)

def import_record(filename):
  with open(filename, "rb") as f:
    data = msgpack.unpack(f)
  return tf.constant(data[0]), tf.constant(data[1], dtype=dtype)

def compute_mean_rates(event_types, event_times, n_event_types): # in case event types are missing in the record, fill them in vis a vis n_event_types
  actual_types, n_occurrences = np.unique(event_types.numpy(), return_counts=True) # always sorted
  proper_n_occurrences = np.zeros(n_event_types)
  proper_n_occurrences[actual_types] = n_occurrences
  interval = event_times[-1] - event_times[0]
  return tf.constant(proper_n_occurrences / interval.numpy(), dtype=dtype)

def simulate_processes_from_result(result, n_events):
  all_event_types, all_event_times = [], []
  for p in range(result["n_particles"]):
    samples = result["sampled_processes"]
    params = analysis.trim_and_select_process(samples, p)
    event_types, event_times = MultivariateProcess.simulate_events_memoryless(params, n_events)
    all_event_types.append(event_types)
    all_event_times.append(event_times)
  return all_event_types, all_event_times

def evaluate_white_log_likelihood(event_types, event_times):
  if not isinstance(event_types, list):
    event_types, event_times = [event_types], [event_times] # easier to just make everything operate on a list
  log_likelihoods = []
  for evt_types, evt_times in zip(event_types, event_times):
    _, occurrences = np.unique(evt_types, return_counts=True) # if some type never occurs, its rate would be zero anyway
    interval = evt_times[-1] - evt_times[0] # formally and ideally, the interval may actually end a little after event_times[-1]. Say you record a day from 8am to 8pm, but the last event happened at 7:39pm. You would want your interval to still be from 8am to 8pm according to the Poisson theory.
    rates = occurrences / interval
    log_likelihood = occurrences * (np.log(rates) - 1) # this form may shed light on how to interpret the likelihood. integral: rates * interval simplifies to occurrences
    log_likelihoods.append(np.sum(log_likelihood) / len(evt_types))
  scale = tf.math.log(tf.cast(len(event_types), dtype))
  posterior = tf.math.reduce_logsumexp(tf.convert_to_tensor(log_likelihoods)) - scale
  return posterior.numpy()

def create_white_process(event_types, event_times, n_event_types):
  if not isinstance(event_types, list):
    event_types, event_times = [event_types], [event_times]
  all_rates = []
  for evt_types, evt_times in zip(event_types, event_times):
    actual_types, occurrences = np.unique(evt_types, return_counts=True) # "reproduced" from the above...
    proper_occurrences = np.zeros(n_event_types)
    proper_occurrences[actual_types] = occurrences
    interval = evt_times[-1] - evt_times[0]
    rates = proper_occurrences / interval
    all_rates.append(rates)
  rates = np.mean(all_rates, axis=0)
  return MultivariateProcess.Parameters(
    tf.zeros([n_event_types, n_event_types], dtype), tf.ones([n_event_types, n_event_types], dtype),
    tf.zeros([n_event_types, n_event_types], dtype), tf.ones([n_event_types, n_event_types], dtype),
    rates)


@argh.arg("-l", "--learning-rates", nargs="*", type=float)
def stein(n_events=20000, n_event_types=4, n_particles=20, batch_size=700, n_iterations=1000, n_test_events=10000, lag=0.0, initial_excitation=1.5,
    rate_kernel=1.0, supp_kernel=0.1, excite_kernel=0.7, excess_suppression_reg=None, learning_rates=None, do_newton=False,
    cross_suppressions=False, optimize_excitations=False, fisher_subsample=None, conservative_fisher_subsample=False,
    truncate_record=False, hawkes_stability_reg=1e3, nonstationary_decay_exponent=None, permanent_suppression_rate=None,
    n_bandwidth_samples=None, bandwidth_sample_radius=None, subsample_for_bandwidths=2, bandwidth_iteration_interval=1, horseshoe_precision=0.0,
    suppression_prior_strength=None, suppression_prior_sharpness=0.0, hawkes=False, learn_hawkes=False, import_filename=None, export_filename=None,
    append_result=False, infer_n_event_types=False, validate_other_result=None, other_result_index=-1, parallel_bundle_size=1,
    silent=False, use_dtype="float64", tag="", kernel_verbosity_freq=0.05, n_repetitions=1): # optimize batch size for GPU. default of do_newton is false so that supplying --do-newton makes it true
  """RUN STEIN.
  I have added tons of runtime options since last documenting this command. Here is a note on the more esoteric options available.
    -> optimize-excitations
      Should we employ the EM algorithm for the Hawkes part of our logistic point process, and perform the Stein iterative inference on the suppression parameters only?
    -> fisher-subsample=[int] ∈ (0, batch-size]
      How many randomly selected samples out of each batch should we take to estimate (without extra bias, but potentially high variance) the Fisher information matrix? Speeds up the computation dramatically if do-newton is selected.
    -> hawkes
      Make the ground truth a (most likely) stable Hawkes process.
    -> learn-hawkes (formerly disallow-suppressions)
      Learn a Hawkes process from the data, regardless of whether that is Hawkes.
    -> import-filename=[str]
      A MsgPack file that contains a two-item list.
        (1) the event types as an array of integers
        (2) the event times as an array of floats, automagically cast to the required dtype
  """

  global ground_truth, event_types, event_times, test_event_types, test_event_times, samples, sampled_processes, scores, insample_posterior, predictive_posterior, ground_truth_score, kernel_metrics
  global results # this is the one that truly matters

  if use_dtype != "float64":
    global dtype
    if use_dtype == "float32":
      dtype = tf.float32
    elif use_dtype == "float16":
      dtype = tf.float16
    elif use_dtype == "bfloat16":
      dtype = tf.bfloat16 # float32 range with truncated precision. GPUs prefer float16
    else:
      print("Unrecognized dtype.")
      return
    MultivariateProcess.amend_dtype(dtype)


  #if eagerly_debug: # don't have to pass this to subsequent stein calls
  #  tf.config.run_functions_eagerly(True) INTRODUCES OTHER PROBLEMS

  if n_repetitions > 1: # so that a value of 2 makes 2 results, even if semantically it should be 3 results
    stein(n_events, n_event_types, n_particles, batch_size, n_iterations, n_test_events, lag, initial_excitation, rate_kernel, supp_kernel,
    excite_kernel, excess_suppression_reg, learning_rates, do_newton, cross_suppressions, optimize_excitations, fisher_subsample, conservative_fisher_subsample,
    truncate_record, hawkes_stability_reg, nonstationary_decay_exponent, permanent_suppression_rate,
    n_bandwidth_samples, bandwidth_sample_radius, subsample_for_bandwidths, bandwidth_iteration_interval, horseshoe_precision,
    suppression_prior_strength, suppression_prior_sharpness, hawkes, learn_hawkes, import_filename, None, False, infer_n_event_types, validate_other_result,
    other_result_index, parallel_bundle_size, silent, use_dtype, tag, kernel_verbosity_freq, n_repetitions-1)
  else:
    results = [] # end of the call chain. initialize the list for everyone else. this call is placed here before the computing happens so that as soon as the leaf node finishes, it can add its result to the list

  if learning_rates is None:
    learning_rates = [1e0, 1e-1, 1e0, 1e-1, 1e-1] if not do_newton else [1e-2]*5 # (can do single-element list too) in the case of Newton-Raphson, it should scale by number of events on its own. slower rates may allow sparser fisher_subsamples

  ground_truth = None
  mean_arrival_time, mean_rates = None, None # only set this if we import an external dataset
  if import_filename is not None:
    all_event_types, all_event_times = import_record(import_filename)
    all_event_types, all_event_times = tf.constant(all_event_types), tf.constant(all_event_times)
    all_event_types -= tf.reduce_min(all_event_types) # might start at 1 if sourced from Julia
    if truncate_record and len(all_event_types) > (n_events + n_test_events): # for the impatient to speed up inference
      all_event_types = all_event_types[:(n_events + n_test_events)]
      all_event_times = all_event_times[:(n_events + n_test_events)]
    else:
      n_events = len(all_event_types) - n_test_events # stubborn test and flexible train sizes
    n_unique_event_types = len(np.unique(all_event_types))
    if infer_n_event_types:
      n_event_types = tf.math.reduce_max(all_event_types).numpy() + 1 # n_unique_event_types ?
    mean_rates = compute_mean_rates(all_event_types, all_event_times, n_event_types)
    mean_arrival_time = np.mean(np.diff(all_event_times.numpy())) * n_event_types # TF doesn't have a native diff function. COMMENTED BECAUSE added degree of freedom not necessary for now...
    print("I see {} real and {} nominal event types.".format(n_unique_event_types, tf.math.reduce_max(all_event_types)+1))
    print("I am also contending with {} events in the training set.{}".format(n_events, " Artificially truncated!" if truncate_record else ""))
    print("Also, the mean rates are as follows:\n{}.".format(mean_rates.numpy()))
    assert tf.math.reduce_max(all_event_types) == n_event_types-1
  elif validate_other_result is not None:
    with open(validate_other_result, "rb") as f:
      original_result = msgpack.load(f)[other_result_index]
    all_event_types, all_event_times = simulate_processes_from_result(original_result, n_events + n_test_events)
    if infer_n_event_types:
      n_event_types = original_result["n_event_types"]
    mean_arrival_time = np.mean([np.mean(np.diff(times.numpy())) for times in all_event_times]) * n_event_types
    mean_rates = np.mean([compute_mean_rates(etypes, etimes, n_event_types) for etypes, etimes in zip(all_event_types, all_event_times)], axis=0)
    ground_truth = MultivariateProcess.deserialize(original_result["sampled_processes"]) # there is already code in place to evaluate ground-truth scores on multiple records... it was purposeless earlier and never even executed
    print("I see {} event types in the original result. It had {} particles.".format(original_result["n_event_types"], original_result["n_particles"]))
    assert original_result["n_event_types"] == n_event_types
    print("Validating a previous result. The mean rates are as follows:\n{}.".format(mean_rates))
  else:
    ground_truth = generate_ground_truth(n_event_types, cross_suppressions=cross_suppressions) if not hawkes else generate_hawkes_ground_truth(n_event_types)
    all_event_types, all_event_times = MultivariateProcess.simulate_events_thinning(ground_truth, n_events+n_test_events, n_events+n_test_events)
    mean_rates = compute_mean_rates(all_event_types, all_event_times, n_event_types)  # Controversial (in my own mind):
    mean_arrival_time = np.mean(np.diff(all_event_times.numpy())) * n_event_types     #   Data-driven initialization even in synthetic experiments, instead of drawing from the same distribution as the ground truth's? Large *wrong* initial excitations can throw the particles way off...
  if isinstance(all_event_types, list):
    event_types, event_times = [t[:-n_test_events] for t in all_event_types], [t[:-n_test_events] for t in all_event_times]
    test_event_types, test_event_times = [t[-n_test_events:] for t in all_event_types], [t[-n_test_events:] for t in all_event_times]
  else:
    event_types, event_times = all_event_types[:-n_test_events], all_event_times[:-n_test_events]
    test_event_types, test_event_times = all_event_types[-n_test_events:], all_event_times[-n_test_events:]

  rate_kernel_bandwidth = rate_kernel # we're talking log scale here.
  suppression_kernel_bandwidth = supp_kernel * (np.sqrt(n_event_types) if not cross_suppressions else n_event_types) # do scale fully, and make bandwidths generally tiny instead. old note ==> DO NOT SCALE FULLY because suppressions have a limited outlier profile: they can only range from 0 to 1 in the first place. highly non-Gaussian
  excitation_kernel_bandwidth = excite_kernel * n_event_types # sqrt of square

  fisher_subsample = int(fisher_subsample) if fisher_subsample is not None else None
  n_bandwidth_samples = None if n_bandwidth_samples is None else int(n_bandwidth_samples)
  bandwidth_sample_radius = None if bandwidth_sample_radius is None else float(bandwidth_sample_radius)
  excess_suppression_reg = None if excess_suppression_reg is None else tf.cast(float(excess_suppression_reg), dtype) # API is not currently uniform in how it treats casting of these regularization hyperparameters...
  nonstationary_decay_exponent = None if nonstationary_decay_exponent is None else float(nonstationary_decay_exponent)
  suppression_prior_strength = None if suppression_prior_strength is None else float(suppression_prior_strength)
  permanent_suppression_rate = None if permanent_suppression_rate is None else tf.cast(float(permanent_suppression_rate), dtype)

  gated_mean_rates = np.maximum(mean_rates, 1e-20) # avoid division by zero
  initial_parameters = generate_initial_parameters(n_particles, n_event_types, cross_suppressions, mean_arrival_time, gated_mean_rates, initial_excitation)

  samples, kernel_metrics, bandwidth_curves, learning_curve, seconds_taken = \
    MultivariateProcess.stein_sample_homogeneous_rates_and_diagonal_suppressions(initial_parameters, event_types, event_times,
      tf.constant(learning_rates, dtype=dtype)[None, :] / (n_events if not do_newton else 1.0), n_iterations, # learning rates are with reference to average log likelihood in gradient descent. for newton, variance is corrected already
      tf.cast(rate_kernel_bandwidth, dtype), tf.cast(suppression_kernel_bandwidth, dtype), lag=lag,
      batch_size=batch_size, optimize_excitations=optimize_excitations, cross_suppressions=cross_suppressions, horseshoe_precision=horseshoe_precision,
      n_bandwidth_samples=n_bandwidth_samples, bandwidth_sample_radius=bandwidth_sample_radius, subsample_for_bandwidths=subsample_for_bandwidths,
      excitation_kernel_bandwidth=tf.cast(excitation_kernel_bandwidth, dtype), bandwidth_iteration_interval=bandwidth_iteration_interval,
      do_newton=do_newton, fisher_subsample=fisher_subsample, learn_hawkes=learn_hawkes, excess_suppression_reg=excess_suppression_reg,
      hawkes_stability_reg=hawkes_stability_reg, nonstationary_decay_exponent=nonstationary_decay_exponent, parallel_bundle_size=parallel_bundle_size,
      suppression_prior_strength=suppression_prior_strength, suppression_prior_sharpness=suppression_prior_sharpness,
      permanent_suppression_rate=permanent_suppression_rate,
      verbose=(not silent), reporting_tag=f"{tag} repeated trial #{n_repetitions}", kernel_verbosity_freq=kernel_verbosity_freq)

  sampled_processes = MultivariateProcess.Parameters(
    samples[0], tf.tile(samples[1][:, None, None], [1, n_event_types, n_event_types]), tf.linalg.diag(samples[2]) if not cross_suppressions else samples[2],
    tf.tile(samples[3][:, None, None], [1, n_event_types, n_event_types]), samples[4], tf.fill(tf.shape(samples[4]), tf.cast(lag, dtype)))

  white_process = create_white_process(event_types, event_times, n_event_types)

  with tf.device("/cpu:0"): # GPU RAM would flounder in the presence of many particles
    if isinstance(event_types, list):
      score_list, test_score_list, ground_truth_score_list, ground_truth_test_score_list = [], [], [], []
      for a, b, ta, tb in zip(event_types, event_times, test_event_types, test_event_times):
        score, excitation_leftover, suppression_leftover, time_of_leftover = MultivariateProcess.compute_log_likelihood(
          sampled_processes, a, b, batch_size=batch_size, excess_suppression_reg=excess_suppression_reg, return_leftovers=True)
        score_list.append(score)
        test_score_list.append(MultivariateProcess.compute_log_likelihood(
          sampled_processes, ta, tb, batch_size=batch_size, excess_suppression_reg=excess_suppression_reg,
          warm_start_excitation=excitation_leftover, warm_start_suppression=suppression_leftover, time_of_warm_start=time_of_leftover)) # warm start from the intermediate variables that the training set left off
        if ground_truth is not None:
          gt_score, gt_excitation_leftover, gt_suppression_leftover, gt_time_of_leftover = MultivariateProcess.compute_log_likelihood(
            ground_truth, a, b, batch_size=batch_size, excess_suppression_reg=excess_suppression_reg, return_leftovers=True)
          ground_truth_score_list.append(gt_score)
          ground_truth_test_score_list.append(MultivariateProcess.compute_log_likelihood(
            ground_truth, ta, tb, batch_size=batch_size, excess_suppression_reg=excess_suppression_reg,
            warm_start_excitation=gt_excitation_leftover, warm_start_suppression=gt_suppression_leftover, time_of_warm_start=gt_time_of_leftover))
      scale = tf.math.log(tf.cast(len(event_types), dtype))
      scores = tf.math.reduce_logsumexp(tf.convert_to_tensor(score_list), axis=0) - scale
      test_scores = tf.math.reduce_logsumexp(tf.convert_to_tensor(test_score_list), axis=0) - scale
      ground_truth_score = tf.math.reduce_logsumexp(tf.convert_to_tensor(ground_truth_score_list)) - scale if ground_truth is not None else None
      ground_truth_test_score = tf.math.reduce_logsumexp(tf.convert_to_tensor(ground_truth_test_score_list)) - scale if ground_truth is not None else None
    else:
      scores, excitation_leftover, suppression_leftover, time_of_leftover = MultivariateProcess.compute_log_likelihood(
        sampled_processes, event_types, event_times, batch_size=batch_size, excess_suppression_reg=excess_suppression_reg, return_leftovers=True)
      test_scores = MultivariateProcess.compute_log_likelihood(
        sampled_processes, test_event_types, test_event_times, batch_size=batch_size, excess_suppression_reg=excess_suppression_reg,
        warm_start_excitation=excitation_leftover, warm_start_suppression=suppression_leftover, time_of_warm_start=time_of_leftover)
      if ground_truth is not None:
        ground_truth_score, gt_excitation_leftover, gt_suppression_leftover, gt_time_of_leftover = MultivariateProcess.compute_log_likelihood(
          ground_truth, event_types, event_times, batch_size=batch_size, excess_suppression_reg=excess_suppression_reg, return_leftovers=True)
        ground_truth_test_score = MultivariateProcess.compute_log_likelihood(
          ground_truth, test_event_types, test_event_times, batch_size=batch_size, excess_suppression_reg=excess_suppression_reg,
          warm_start_excitation=gt_excitation_leftover, warm_start_suppression=gt_suppression_leftover, time_of_warm_start=gt_time_of_leftover)

    insample_posterior = (tf.math.reduce_logsumexp(scores) - tf.math.log(tf.cast(n_particles, dtype))) / tf.cast(n_events, dtype)
    predictive_posterior = (tf.math.reduce_logsumexp(test_scores) - tf.math.log(tf.cast(n_particles, dtype))) / tf.cast(n_test_events, dtype) # make them commensurable (to be technically correct, would probably have to divide by time intervals instead. but everyone does it this way)

    if validate_other_result is not None:
      validation_scores = MultivariateProcess.compute_log_likelihood(
        sampled_processes, original_result["event_types"], tf.cast(original_result["event_times"], dtype), # train events or test events?
        batch_size=batch_size, excess_suppression_reg=excess_suppression_reg)
      events_scale = tf.cast(original_result["n_events"], dtype)
      validation_posterior = (tf.math.reduce_logsumexp(validation_scores) - tf.math.log(tf.cast(n_particles, dtype))) / events_scale
      print(colored(
        "You went through countless hoops. Now you may finally gaze upon the coveted validation posterior: {}.".format(validation_posterior.numpy()),
        "green"))

    accuracies = analysis.compute_categorical_accuracy(sampled_processes, event_types, event_times, batch_size)
    test_accuracies = analysis.compute_categorical_accuracy(sampled_processes, test_event_types, test_event_times, batch_size)
    white_accuracy = analysis.compute_categorical_accuracy(white_process, event_types, event_times, batch_size)
    white_test_accuracy = analysis.compute_categorical_accuracy(white_process, test_event_types, test_event_times, batch_size)

    white_score = MultivariateProcess.compute_log_likelihood(white_process, event_types, event_times, batch_size=batch_size)
    white_test_score = MultivariateProcess.compute_log_likelihood(white_process, test_event_types, test_event_times, batch_size=batch_size)


  if learn_hawkes: # n_effective_parameters may be a more reasonable quantity, taking sparsity into account. Although a sparse model could more flexibly choose its significant parameters, allowing for overfitting (a la spurious correlations) even with few effective parameters.
    n_parameters = n_event_types**2 + 1 + n_event_types # coef, rate, background
  elif cross_suppressions:
    if permanent_suppression_rate is None:
      n_parameters = 2*(n_event_types**2) + 2 + n_event_types # (coef and coef), rates, background
    else:
      n_parameters = 2*(n_event_types**2) + 1 + n_event_types
  else:
    if permanent_suppression_rate is None:
      n_parameters = n_event_types**2 + 2 + 2*n_event_types # coef, rates, (coef and background)
    else:
      n_parameters = n_event_types**2 + 1 + 2*n_event_types

  get_bic = lambda scores, n_events: \
    (0.5*tf.cast(n_parameters, dtype)*tf.math.log(tf.cast(n_events, dtype)) - tf.math.reduce_max(scores)) / tf.cast(n_events, dtype)  # scores are assumed to be aggregated (sum, not expectation of log-likelihoods) over the process, not averaged
  get_aic = lambda scores, n_events: \
    (tf.cast(n_parameters, dtype) - tf.math.reduce_max(scores)) / tf.cast(n_events, dtype)

  insample_bic = get_bic(scores, n_events) # BIC and AIC operate on the maximized likelihood, even when an entire posterior is sampled
  insample_aic = get_aic(scores, n_events) # do people ever train on AIC or BIC to implicitly induce sparsity, if done on effective present parameters? Aram pointed out that one is used more for training whereas the other is for judging/critiquing.
  # ... Wikipedia states the arguments of various researchers, broadly summed up as BIC having the asymptotic guarantee to find the best model while AIC will generally find the best "approximating" model regardless of whether the true one is sampled
  # Interesting quantities like the KL-divergence and entropies, which invoke an integral over the data distribution, are complicated when observing a single realization of each time instance in a stochastic process as we are here. If anything, averaging should be over the time interval and not over the number of events, though in practice both may work out fine. Most people average over events in machine learning.
  # see for the widely applicable Bayesian (or Akaike http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/waicwbic_e.html) information criterion --> Watanabe 2013, Gelman (this man is everywhere!) 2014
  # WAIC and WBIC are tickier for the reasons stated above. Explore further, later on. Also possibly DIC, which implements a statistical measure---think bits of information---of "effective parameters".
  predictive_bic = get_bic(test_scores, n_test_events)
  predictive_aic = get_aic(test_scores, n_test_events)

  #white_score = evaluate_white_log_likelihood(all_event_types, all_event_times)
  #white_bic = get_bic([white_score], n_events + n_test_events)
  #white_aic = get_aic([white_score], n_events + n_test_events)
  predictive_white_bic = get_bic([white_test_score], n_test_events)
  predictive_white_aic = get_aic([white_test_score], n_test_events)
  insample_white_bic = get_bic([white_score], n_events)
  insample_white_aic = get_aic([white_score], n_events)

  print(colored(f"Final report for {tag}.\n", "blue"), f"""\
    Predictive posterior: {predictive_posterior.numpy()},\tinsample posterior: {insample_posterior.numpy()};
    out-of-sample baseline: {white_test_score.numpy() / n_test_events},\tin-sample baseline: {white_score.numpy() / n_events}.""")

  result = {
    "utctime_finished": dt.datetime.utcnow().isoformat(),
    "seconds_taken": seconds_taken,
    "import_filename": import_filename,
    "validate_other_result": validate_other_result,
    "infer_n_event_types": infer_n_event_types,
    "use_dtype": use_dtype,
    "ground_truth": MultivariateProcess.serialize(ground_truth) if ground_truth is not None else None,
    "initial_parameters": MultivariateProcess.serialize(initial_parameters),
    "initial_excitation": initial_excitation, # should have been called initial excitement, haha
    "event_types": event_types.numpy() if not isinstance(event_types, list) else [e.numpy() for e in event_types],
    "event_times": event_times.numpy() if not isinstance(event_types, list) else [e.numpy() for e in event_times],
    "test_event_types": test_event_types.numpy() if not isinstance(event_types, list) else [e.numpy() for e in test_event_types],
    "test_event_times": test_event_times.numpy() if not isinstance(event_types, list) else [e.numpy() for e in test_event_times],
    "samples": samples,
    "sampled_processes": MultivariateProcess.serialize(sampled_processes),
    "scores": scores.numpy(),
    "insample_posterior": insample_posterior.numpy(),
    "insample_aic": insample_aic.numpy(),
    "insample_bic": insample_bic.numpy(),
    "test_scores": test_scores.numpy(),
    "predictive_posterior": predictive_posterior.numpy(),
    "predictive_aic": predictive_aic.numpy(),
    "predictive_bic": predictive_bic.numpy(),
    "validation_scores": validation_scores.numpy() if validate_other_result is not None else None,
    "validation_posterior": validation_posterior.numpy() if validate_other_result is not None else None,
    "ground_truth_score": ground_truth_score.numpy() if ground_truth is not None else None,
    "ground_truth_test_score": ground_truth_test_score.numpy() if ground_truth is not None else None,
    "white_log_likelihood": white_score.numpy() / n_events,
    "white_test_log_likelihood": white_test_score.numpy() / n_test_events,
    "predictive_white_bic": predictive_white_bic.numpy(),
    "predictive_white_aic": predictive_white_aic.numpy(),
    "insample_white_bic": insample_white_bic.numpy(),
    "insample_white_aic": insample_white_aic.numpy(),
    "accuracies": accuracies.numpy(),
    "test_accuracies": test_accuracies.numpy(),
    "white_accuracy": white_accuracy.numpy(),
    "white_test_accuracy": white_test_accuracy.numpy(),
    "white_process_background_rates": white_process.background.numpy(),
    "kernel_metrics": kernel_metrics,
    "n_particles": n_particles,
    "n_events": n_events,
    "n_test_events": n_test_events,
    "n_event_types": n_event_types,
    "n_parameters": n_parameters,
    "truncate_record": truncate_record,
    "batch_size": batch_size,
    "n_iterations": n_iterations,
    "lag": lag,
    "cross_suppressions": cross_suppressions,
    "init_rate_kernel_bandwidth": rate_kernel_bandwidth,
    "init_suppression_kernel_bandwidth": suppression_kernel_bandwidth, # these are transformed versions of e.g. supp_kernel
    "init_excitation_kernel_bandwidth": excitation_kernel_bandwidth,
    "excess_suppression_reg": excess_suppression_reg.numpy() if excess_suppression_reg is not None else None,
    "hawkes_stability_reg": hawkes_stability_reg if learn_hawkes else None,
    "nonstationary_decay_exponent": nonstationary_decay_exponent,
    "learning_rates": learning_rates,
    "learning_curve": learning_curve,
    "do_newton": do_newton,
    "optimize_excitations": optimize_excitations,
    "fisher_subsample": fisher_subsample,
    "conservative_fisher_subsample": conservative_fisher_subsample,
    "parallel_bundle_size": parallel_bundle_size,
    "n_bandwidth_samples": n_bandwidth_samples,
    "bandwidth_sample_radius": bandwidth_sample_radius,
    "particle_subsample_for_bandwidth_selection": subsample_for_bandwidths,
    "bandwidth_iteration_interval": bandwidth_iteration_interval,
    "excitation_kernel_bandwidth": bandwidth_curves[0],
    "suppression_kernel_bandwidth": bandwidth_curves[1],
    "rate_kernel_bandwidth": bandwidth_curves[2],
    "horseshoe_precision": horseshoe_precision,
    "suppression_prior_strength": suppression_prior_strength,
    "suppression_prior_sharpness": suppression_prior_sharpness,
    "hawkes": hawkes,
    "learn_hawkes": learn_hawkes,
    "permanent_suppression_rate": permanent_suppression_rate.numpy() if permanent_suppression_rate is not None else None,
    "tag": tag
  }

  results += [result]

  if export_filename is not None:
    if append_result:
      with open(export_filename, "rb") as f:
        existing_results = msgpack.load(f)
      results = existing_results + results
    with open(export_filename, "wb") as f:
      msgpack.dump(results, f) # encoding="utf-8" or not?



def mcmc(n_chains, n_samples, n_burnin, n_events=20000, n_event_types=4, batch_size=700, n_moments=2, tiny_lognormal_sigma=1e-4, beta_sum=1e5, lognormal_sigma=0.3, rate_gamma_prior=[10.0, 10.0], hawkes=False, import_filename=None, n_repetitions=1):
  "RUN MCMC."

  global ground_truth, event_types, event_times, sampled_variables, sampled_log_likelihoods, acceptance_rate, moments, results

  if n_repetitions > 1: # see comments in stein(...)
    stein(n_chains, n_samples, n_burnin, n_events, n_event_types, batch_size, n_moments, tiny_lognormal_sigma, beta_sum, lognormal_sigma, rate_gamma_prior, hawkes, import_filename, n_repetitions-1)
  else:
    results = []

  if import_filename is not None:
    ground_truth = None
    event_types, event_times = import_record(import_filename)
    event_types -= tf.reduce_min(event_types)
    n_events = len(event_types)
    assert tf.reduce_max(event_types) == n_event_types-1
  else:
    ground_truth = generate_ground_truth(n_event_types) if not hawkes else generate_hawkes_ground_truth(n_event_types)
    event_types, event_times = MultivariateProcess.simulate_events_thinning(ground_truth, n_events, n_events)

  params = generate_initial_parameters(int(n_chains), n_event_types)
  sampled_variables, sampled_log_likelihoods, acceptance_rate, moments = MultivariateProcess.sample_homogeneous_rates_and_diagonal_suppressions(
    params, tf.constant(event_types), tf.constant(event_times), int(n_samples), int(n_burnin), n_moments,
    tf.cast(lognormal_sigma, dtype), tf.cast(beta_sum, dtype), tf.cast(tiny_lognormal_sigma, dtype),
    rate_gamma_prior=rate_gamma_prior, do_maximizations=True, batch_size=batch_size, verbose=True)

  ground_truth_score = MultivariateProcess.compute_log_likelihood(ground_truth, event_types, event_times, batch_size=batch_size) if ground_truth is not None else None

  predictive_posterior = tf.math.reduce_logsumexp(sampled_log_likelihoods) - tf.math.log(tf.cast(int(n_samples), dtype)) # NOT really "predictive" because it's evaluated on the training data

  result = {
    "utctime_finished": dt.datetime.utcnow(),
    "ground_truth": MultivariateProcess.serialize(ground_truth) if ground_truth is not None else None,
    "event_types": event_types.numpy(),
    "event_times": event_times.numpy(),
    "samples": sampled_variables,
    "log_likelihoods": sampled_log_likelihoods,
    "predictive_posterior": predictive_posterior,
    "acceptance_rate": acceptance_rate,
    "moments": moments,
    "ground_truth_score": ground_truth_score.numpy() if ground_truth is not None else None,
    #"kernel_metrics": see which dimensions of the proposal are likely to be at fault for rejection?
    "n_chains": n_chains,
    "n_events": n_events,
    "n_event_types": n_event_types,
    "batch_size": batch_size,
    "n_samples": n_samples,
    "n_burnin": n_burnin,
    "n_moments": n_moments,
    "tiny_lognormal_sigma": tiny_lognormal_sigma,
    "beta_sum": beta_sum,
    "lognormal_sigma": lognormal_sigma,
    "rate_gamma_prior": rate_gamma_prior,
    "hawkes": hawkes
  }

  results += [result]

if __name__ == "__main__":
  argh.dispatch_commands([stein, mcmc])
