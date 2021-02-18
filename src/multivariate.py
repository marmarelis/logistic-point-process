import numpy as np
import pandas as pd
import tensorflow as tf # REQUIRES TF 2.2
from types import SimpleNamespace
from tqdm import tqdm
import pprint # pretty printing
from termcolor import colored # colorful printing
from collections import namedtuple
from functools import reduce
#import tensorflow_probability as tfp
import sys

import stein

# In the likelihood calculation, I want to be able to supply all the parameters as arguments. In fact, the function itself may be static.
# Then, for my Markov-chain Monte Carlo experiments, I can batch the parameters. In order to keep that versatile, I will only add rightward
# dimensions for events and let the broadcasting be arbitrary.

class MultivariateProcess(tf.Module):
  dtype = tf.float64
  zero = tf.zeros([], dtype=dtype)
  unit = tf.ones([], dtype=dtype)
  intensity_floor = tf.cast(1e-35, dtype) # we want this to be OBSESSIVELY TINY. A zero gradient is most problematic in scenarios where the log tends to -infinity. For instance, inf*0 becomes nan, when we truly want the inf to persist.
  numerical_limit = tf.cast(1e38, dtype) # getting closer to more unified "epsilons"

  @classmethod
  def amend_dtype(cls, dtype):
    cls.dtype = dtype
    cls.zero = tf.zeros([], dtype=dtype)
    cls.unit = tf.ones([], dtype=dtype)
    cls.intensity_floor = tf.cast(cls.intensity_floor, dtype)
    cls.numerical_limit = tf.cast(cls.numerical_limit, dtype)

  #class Parameters:
  #  def __init__(self, excitation_coef, excitation_rate, suppression_coef, suppression_rate, background, cast=True): # a restricted SimpleNamespace
  #    dtype = MultivariateProcess.dtype
  #    self.n_dims = tf.shape(excitation_coef)[-1]
  #    self.excitation_coef = tf.cast(excitation_coef, dtype=dtype) if cast else excitation_coef
  #    self.suppression_coef = tf.cast(suppression_coef, dtype=dtype) if cast else suppression_coef
  #    self.excitation_rate = tf.cast(excitation_rate, dtype=dtype) if cast else excitation_rate
  #    self.suppression_rate = tf.cast(suppression_rate, dtype=dtype) if cast else suppression_rate
  #    self.background = tf.cast(background, dtype=dtype) if cast else background

  #  def __repr__(self):
  #    return pprint.pformat({k: v for k, v in self.__dict__.items()}) # .numpy() doesn't exist in precompiled mode
    #def __eq__(self, other):
    #  return self.__dict__ == other.__dict__ # to alleviate the needless recompiling of @tf.functions
    #def __hash__(self):
    #  return hash(repr(self.__dict__)) # kind of tricky. if bugs surface, I should remove this.

  ParameterTuple = namedtuple("Parameters", ["n_dims", "excitation_coef", "excitation_rate", "suppression_coef", "suppression_rate", "background", "lag"])
  @classmethod # fake class that returns a namedtuple because Tensorflow knows how to handle these
  def Parameters(cls, excitation_coef, excitation_rate, suppression_coef, suppression_rate, background, lag=None, cast=True):
      dtype = cls.dtype
      n_dims = tf.shape(excitation_coef)[-1]
      excitation_coef = tf.cast(excitation_coef, dtype=dtype) if cast else excitation_coef
      suppression_coef = tf.cast(suppression_coef, dtype=dtype) if cast else suppression_coef
      excitation_rate = tf.cast(excitation_rate, dtype=dtype) if cast else excitation_rate
      suppression_rate = tf.cast(suppression_rate, dtype=dtype) if cast else suppression_rate
      background = tf.cast(background, dtype=dtype) if cast else background
      if lag is None: lag = tf.zeros_like(background)
      lag = tf.cast(lag, dtype=dtype) if cast else lag # Introducing a new parameter to this whole mess won't be easy... positive lag means there's a delay before an effect kicks in. Also: different lags for different event types? Might as well if not difficult...
      self = cls.ParameterTuple(n_dims, excitation_coef, excitation_rate, suppression_coef, suppression_rate, background, lag)
      return self

  def __init__(self, excitation_coef, excitation_rate, suppression_coef, suppression_rate, background, lag=None):
    self.n_dims = tf.shape(excitation_coef)[-1] # must make sure the last dim is never a batch dimension
    self.excitation_coef = tf.Variable(tf.cast(excitation_coef, dtype=self.dtype)) # cast in case it's a tensor of the wrong type
    self.suppression_coef = tf.Variable(tf.cast(suppression_coef, dtype=self.dtype))
    self.excitation_rate = tf.Variable(tf.cast(excitation_rate, dtype=self.dtype))
    self.suppression_rate = tf.Variable(tf.cast(suppression_rate, dtype=self.dtype))
    self.background = tf.Variable(tf.cast(background, dtype=self.dtype))
    self.lag = tf.Variable(tf.cast(lag, dtype=self.dtype) if lag is not None else tf.zeros_like(background))

  def package_parameters(self):
    return self.Parameters(
        self.excitation_coef, self.excitation_rate, self.suppression_coef, self.suppression_rate, self.background, self.lag
      )

  def unpackage_parameters(self, params):
    self.excitation_coef.assign(params.excitation_coef); self.excitation_rate.assign(params.excitation_rate)
    self.suppression_coef.assign(params.suppression_coef); self.suppression_rate.assign(params.suppression_rate)
    self.background.assign(params.background); self.lag.assign(params.lag)

  @classmethod
  @tf.function
  def gather_background(cls, background, event_types):
    n_batch_dims = len(tf.shape(background)) - 1
    expanded_event_types = tf.tile(
      tf.reshape(event_types, tf.concat([tf.cast([1 for _ in range(n_batch_dims)], tf.int32), [-1]], axis=0)), # sometimes that array is converted to floats?
      tf.concat([tf.shape(background)[:n_batch_dims], [1]], axis=0))
    return tf.gather(background, expanded_event_types, batch_dims=n_batch_dims) # would setting axis=-1 simplify this immensely, if broadcasting were supported?

  @classmethod
  @tf.function
  def append_terminal_event_for_batching(cls, params, event_types, event_times, rough_extension=1e-7): # 1e-7 is the smallest we can go such that 1+1e-7 > 1 in float32
    if tf.size(event_types) == 0: # all the corner cases...
      return event_types, event_times
    max_lag = tf.math.reduce_max(cls.gather_background(params.lag, event_types[-1][None])[..., 0]) # we do not care for inhomogeneous lags
    ultimate_time = (event_times[-1] + max_lag) * (1.0 + rough_extension)
    return tf.concat([event_types, tf.cast([-1], event_types.dtype)], axis=-1),\
           tf.concat([event_times, tf.cast([ultimate_time], event_times.dtype)], axis=-1) # we still want to carry over the leftovers to this location, which is after the real ultimate event so that it gets counted too in case we want to use those leftovers for a forecast.

  @classmethod # only this ordering appears to work
  @tf.function
  def compute_log_likelihood(cls, params, event_types, event_times, # idea. if this is the test set, warm up with preceding training-set batch leftovers?
      batch_size=None, verbose=False, do_batched_gradient=False, excess_suppression_reg=None, lower_bound=False, weigh=None,
      warm_start_excitation=None, warm_start_suppression=None, time_of_warm_start=None, return_leftovers=False): # complete pipeline to take the immediately preceding training set's leftovers and feed them into the test-set likelihood evaluation
    all_types = tf.range(tf.cast(params.n_dims, dtype=event_types.dtype))
    integral_upper_bound = lower_bound # lower bound of the integral translates to an upper bound on the log-likelihood, and vice versa
    if batch_size is None:
      intensity = cls.infer_intensity(params, event_types, event_times, event_types, event_times)
      score = tf.math.reduce_sum(tf.math.log(intensity), axis=-1)
      penalty = tf.math.reduce_sum(cls.integrate_intensity(params, event_types, event_times, all_types, upper_bound=integral_upper_bound), axis=-1) # should be 1D, except for parameter batching
      if excess_suppression_reg is not None:
        regularization = cls.compute_excess_suppression_penalty(params, event_types, event_times)
        return score - penalty - excess_suppression_reg*regularization
      else:
        return score - penalty
    else:
      if warm_start_excitation is None:
        leftover_excitation = tf.zeros_like(params.excitation_coef)
        leftover_suppression = tf.zeros_like(params.suppression_coef)
        time_of_leftover = event_times[0]
      else:
        leftover_excitation = warm_start_excitation
        leftover_suppression = warm_start_suppression
        time_of_leftover = time_of_warm_start
      parallel_shape = tf.shape(params.excitation_coef)[:-2] # sets of parameters batched in parallel. excitation_coef is the most reliable
      overall_score, overall_penalty, overall_reg = tf.zeros(parallel_shape, dtype=cls.dtype), tf.zeros(parallel_shape, dtype=cls.dtype), tf.zeros(parallel_shape, dtype=cls.dtype) # so that I can debug them separately
      variables = [params.excitation_coef, params.excitation_rate, params.suppression_coef, params.suppression_rate, params.background] # for gradient
      n_batches = tf.cast(tf.math.ceil(tf.cast(tf.size(event_times), cls.dtype) / tf.cast(batch_size, cls.dtype)), tf.int32)
      gradient = [tf.zeros(tf.concat([[n_batches], tf.shape(var)], axis=0), dtype=cls.dtype) for var in variables]
      padded_event_types, padded_event_times = cls.append_terminal_event_for_batching(params, event_types, event_times) # last event, that we labored to include here, does not contribute to the log-likelihood besides in evaluating intensity at that point, which we already did beforehand
      def while_body(t, overall_score, overall_penalty, overall_reg, gradient, leftover_excitation, leftover_suppression, time_of_leftover):
        batch_types = padded_event_types[t:(t+batch_size+1)]
        batch_times = padded_event_times[t:(t+batch_size+1)] # batches henceforth include the first element of the next batch, just to complete the intervals
        eval_types = event_types[t:(t+batch_size)] # final-batch behavior is different between this and batch_types[:-1], depending on whether it partially fills the batch
        eval_times = event_times[t:(t+batch_size)] # further, batch_types will always be trimmed at the end, which is why we had to pad it. We want to count every last event.
        if weigh is not None:
          weight = weigh(event_times[-1] - batch_times[0]) # weigh by batch because otherwise it is difficult to wrangle the integrals by each individual occurrence
        else:
          weight = cls.unit
        if do_batched_gradient:
          tape_monitor = tf.GradientTape(watch_accessed_variables=False) # do tape so that we control where the gradients propagate; make sure they don't flow through leftover_* variables
          tape = tape_monitor.__enter__() # flexibly emulate the with statement
          tape.watch(variables)
        intensity, next_leftover_excitation, next_leftover_suppression, time_of_next_leftover = \
          cls.infer_intensity(params, batch_types, batch_times, eval_types, eval_times, leftover_excitation, leftover_suppression, time_of_leftover)
        score = tf.math.reduce_sum(tf.math.log(tf.math.maximum(intensity, cls.intensity_floor)), axis=-1)
        penalty = tf.math.reduce_sum(cls.integrate_intensity(params, batch_types, batch_times, all_types, event_times[-1],
          leftover_excitation, leftover_suppression, time_of_leftover, upper_bound=integral_upper_bound), axis=-1)
        if excess_suppression_reg is not None:
          regularization = excess_suppression_reg * cls.compute_excess_suppression_penalty(params, eval_types, eval_times, leftover_suppression, time_of_leftover)
        else:
          regularization = cls.zero
        if do_batched_gradient: # do separate gradient calculations for each batch, in an attempt to make it efficient for large records
          batch_log_likelihood = score - penalty - regularization
          tape_monitor.__exit__(None, None, None) # these things are populated when cleaning up from an exception. that may only occur when constructing an incorrect graph
          batch_gradient = tape.gradient(batch_log_likelihood, variables) # in compiled mode, we don't need GradientTape. We can even employ tf.hessians.
          for i in range(len(variables)):
            selector = tf.reshape(tf.one_hot(t // batch_size, n_batches, dtype=cls.dtype), [-1] + [1]*len(batch_gradient[i].shape))
            gradient[i] = tf.ensure_shape(gradient[i] + batch_gradient[i] * selector * weight, gradient[i].shape) # safeguard needed for when n_batches=1, which makes the shape broadcastable and confuses TF
        overall_score += score * weight
        overall_penalty += penalty * weight
        overall_reg += regularization * weight
        leftover_excitation, leftover_suppression, time_of_leftover = next_leftover_excitation, next_leftover_suppression, time_of_next_leftover
        return t + batch_size, overall_score, overall_penalty, overall_reg, gradient, leftover_excitation, leftover_suppression, time_of_leftover
      starting_variables = (0, overall_score, overall_penalty, overall_reg, gradient, leftover_excitation, leftover_suppression, time_of_leftover) # making it all explicit because I'm tired of obscure "this tensor was defined inside another block" errors from a frail static analyzer
      _, overall_score, overall_penalty, overall_reg, gradient, next_leftover_excitation, next_leftover_suppression, time_of_next_leftover = \
        tf.while_loop(lambda t, os, op, oreg, g, le, ls, tol: t < tf.size(event_times), while_body, starting_variables) # last batch will most likely be of different size
      if verbose: tf.print(overall_score, overall_penalty, overall_reg, output_stream=sys.stdout)
      log_likelihood = overall_score - overall_penalty - overall_reg
      if do_batched_gradient:
        if return_leftovers:
          return log_likelihood, gradient, next_leftover_excitation, next_leftover_suppression, time_of_next_leftover
        return log_likelihood, gradient
      else:
        if return_leftovers:
          return log_likelihood, next_leftover_excitation, next_leftover_suppression, time_of_next_leftover
        return log_likelihood

  @classmethod
  @tf.function
  def compute_excess_suppression_penalty(cls, params, event_types, event_times, leftover_suppression=None, time_of_leftover=None): # it appears futile to eliminate the intensity integral underestimate when suppressions surpass the limit, so heavily regularize against these conditions instead.
    event_lags = cls.gather_background(params.lag, event_types)
    max_event_lag = tf.math.reduce_max(event_lags) # to avoid having to batch eval_times by the available parameters, select simply the largest event lag found in the time series.
    time_epsilon = 1e-7 # so that 1+1e-7 > 1 in float32
    eval_times = (event_times + max_event_lag) * (1.0 + time_epsilon) # so that we get the whole peak, counting the contribution of the current event
    if leftover_suppression is not None:
      suppressions, _, _ = cls.infer_suppression(params, event_types, event_times, event_types[:-1], eval_times[:-1], leftover_suppression, time_of_leftover) # we look at the peaks. integrating all the excess (above unit) is possible, but more complicated and probably unnecessary.
    else:
      suppressions = cls.infer_suppression(params, event_types, event_times, event_types, eval_times)
    suppression = tf.math.reduce_sum(suppressions, axis=-2)
    penalties = tf.math.maximum(suppression - cls.unit, cls.zero)
    penalty = tf.math.reduce_sum(tf.square(penalties), axis=-1)
    return penalty

  @classmethod
  @tf.function # You know, it turns out that a lot of the Hessian is blank here; within a column (receiving event type), event types do have cross-influences. But not across columns, by nature of the point process. Worth optimizing further?
  def estimate_fisher_information(cls, params, event_types, event_times, batch_size,
      subsample_size=None, excess_suppression_reg=None, robust_subsample=False, weigh=None): # tape.jacobian does not support batching. batch_jacobian is limited, so I'm not bothering with any parameter batching support for now. UPDATE: will use batch_jacobian to avoid finding derivatives between independent parameter sets only when parameters ARE batched
    #tf.assert_equal(tf.rank(params.excitation_coef), 2) # try to catch misguided batching (hitherto unimplemented) --- experimental support has been added
    parameters_are_bundled = len(params.excitation_coef.shape) == 3 # very specific type of batching supported
    parameter_bundle_size = tf.shape(params.excitation_coef)[0] if parameters_are_bundled else None
    leftover_excitation = tf.zeros_like(params.excitation_coef)
    leftover_suppression = tf.zeros_like(params.suppression_coef)
    time_of_leftover = event_times[0]
    variables = [params.excitation_coef, params.excitation_rate, params.suppression_coef, params.suppression_rate, params.background]
    reg_variables = [False, False, True, True, False] # keep index positions
    n_batches = tf.cast(tf.math.ceil(tf.cast(tf.size(event_times), cls.dtype) / tf.cast(batch_size, cls.dtype)), tf.int32)
    if parameters_are_bundled:
      covariance = [tf.zeros([parameter_bundle_size, tf.size(var[0, ...]), tf.size(var[0, ...])], dtype=cls.dtype) for var in variables]
    else:
      covariance = [tf.zeros([tf.size(var), tf.size(var)], dtype=cls.dtype) for var in variables] # covariance of INTENSITIES at each event
    def while_body(t, covariance, leftover_excitation, leftover_suppression, time_of_leftover):
      batch_types = event_types[t:(t+batch_size+1)]
      batch_times = event_times[t:(t+batch_size+1)] # batches henceforth include the first element of the next batch, just to complete the intervals
      if weigh is not None:
        weight = weigh(event_times[-1] - batch_times[0])
      else:
        weight = cls.unit
      if subsample_size is not None:
        random_indices = tf.random.shuffle(tf.range(tf.size(batch_types) - 1)) # in case the final batch is smaller, which it often is
        sorted_indices = random_indices[:subsample_size] # sample_size random indices without replacement; sorting is not actually necessary to maintain order
        eval_types = tf.gather(batch_types, sorted_indices)
        eval_times = tf.gather(batch_times, sorted_indices)
        scale = tf.cast(batch_size, cls.dtype) / tf.cast(subsample_size, cls.dtype)
      else:
        eval_types, eval_times = batch_types[:-1], batch_times[:-1]
        scale = cls.unit
      with tf.GradientTape(persistent=parameters_are_bundled, watch_accessed_variables=False) as tape: # we want fine-grained control, also for efficiency's sake
        tape.watch(variables)
        intensity, next_leftover_excitation, next_leftover_suppression, time_of_next_leftover = \
          cls.infer_intensity(params, batch_types, batch_times, eval_types, eval_times, leftover_excitation, leftover_suppression, time_of_leftover)
        log_intensity = tf.math.log(tf.math.maximum(intensity, cls.intensity_floor)) # silly me! I forgot this crucial step earlier. I establish a lower bound for stability, since this may otherwise drive chaotic behavior.
      if parameters_are_bundled:
        batch_jacobian = [tape.batch_jacobian(log_intensity, variable, experimental_use_pfor=True) for variable in variables] # no list of sources supported
      else:
        batch_jacobian = tape.jacobian(log_intensity, variables, experimental_use_pfor=True) # We need GradientTape for tape.jacobian. There is no tf.jacobians. Output should be a list of n_events x (variable shape...) for each variable. experimental_use_pfor causes some crash currently that nightly has fixed, but not any stable release --> UPDATE: turning it on boosts parallel performance massively now (TF 2.3).
      new_covariance = [] # create new list as an attempted fix (didn't work in the end): some obscure nonsense (control flows; discovered it when a print statement would rectify this) produces a bug in the absence of this, with the following necessary condition: I don't even know anymore; conda and pip and GPU and CPU produce this eventually, but it's not deterministic.
      for i in range(len(variables)):
        #with tf.control_dependencies(batch_jacobian if i == 0 else [covariance[i-1]]): # serialize. attempted fix.
        if parameters_are_bundled:
          flat_jacobian = tf.reshape(batch_jacobian[i], [parameter_bundle_size, tf.shape(intensity)[1], tf.size(variables[i][0, ...])])
        else:
          flat_jacobian = tf.reshape(batch_jacobian[i], [tf.shape(intensity)[0], tf.size(variables[i])]) # FORCE intensity to be 1D (no parameter batching support)
        products = flat_jacobian[..., :, :, None] * flat_jacobian[..., :, None, :] #tf.linalg.matmul(flat_jacobian[:, :, None], flat_jacobian[:, None, :]) # effectively a batched outer product
        corrected_products = tf.where(tf.math.is_nan(products), tf.cast(np.inf, products.dtype), products) # the only way nans are produced here is from 0 (due to floored log) times inf. We can't tell the desired sign of the zero, so do we trust that both infs would have had the same sign?
        update = tf.math.reduce_sum(corrected_products, axis=-3)*(tf.square(scale) if robust_subsample else scale)*tf.square(weight) # sum across events and correct for subsampling. square the scale simply for a dramatized overestimate, to make moves more conservative. remember, the gradient is also a biased estimate resulting from the truncated batch. determinant takes this factor to the power of the matrix rank
        updated_cov = covariance[i] + update
        if excess_suppression_reg is not None and reg_variables[i]: # Fisher information is the negated Hessian in expectation (as well as the gradient's outer product)
          prior_variables = [variables[i] for i in range(len(variables)) if reg_variables[i]]
          with tf.GradientTape(watch_accessed_variables=False) as tape_two: # tightly control the scope of differentiation. MUCH FASTER when we substituted tf.hessians for this
            tape_two.watch(prior_variables)
            regularization = -excess_suppression_reg * cls.compute_excess_suppression_penalty(params, eval_types, eval_times, # a measly under-approximation, but better than nothing nonetheless
              leftover_suppression, time_of_leftover)
          prior_gradient = tape_two.gradient(regularization, variables[i]) # no batching version of this supported
          #prior_hessian = tape_one.jacobian(prior_gradient, variables[i], experimental_use_pfor=False) # does the second derivative of this become some sort of pieecwise constant? also: fell once again to the segfault trap of experimental_use_pfor; unfortunately, fixing this did kill the performance quite a bit. No wonder it's enabled by default.
          if parameters_are_bundled:
            flat_prior_gradient = tf.reshape(prior_gradient, [parameter_bundle_size, tf.size(variables[i][0, ...])])
          else:
            flat_prior_gradient = tf.reshape(prior_gradient, [tf.size(variables[i])]) # no '-1' ambiguous dimension sizes in order to explicitly catch mistakes
          flat_prior_hessian = -flat_prior_gradient[..., :, None] * flat_prior_gradient[..., None, :] # we cannot afford the costly direct Hessian call, so approximate it like we usually do. Such an approximation only holds in theory when treating a logL, and when it's maximized. Let's see how it does here.
          #flat_prior_hessian = tf.reshape(prior_hessian, [tf.size(variables[i]), tf.size(variables[i])])
          updated_cov -= flat_prior_hessian * scale # leaner syntax than above, which suffered from debugging obscure Tensorflow bugs. If it ain't broke, don't fix it!
        new_covariance.append(updated_cov)
      leftover_excitation, leftover_suppression, time_of_leftover = next_leftover_excitation, next_leftover_suppression, time_of_next_leftover
      return t + batch_size, new_covariance, leftover_excitation, leftover_suppression, time_of_leftover
    starting_variables = (0, covariance, leftover_excitation, leftover_suppression, time_of_leftover)
    _, covariance, _, _, _ = tf.while_loop(lambda t, c, le, ls, tol: t < tf.size(event_times), while_body, starting_variables, parallel_iterations=1) # parallel_iteratinos=1 ended the saga documented above! Did TF sometimes get ahead of itself, and cause a race condition of some sort? It tormented me for so long!
    return covariance # covariance is kind of a misnomer, but it permeates the codebase now...

  @classmethod
  @tf.function # todo some fancy decorator that creates a function of a similar name, using implicit parameters?
  def infer_causalities(cls, params, event_types, event_times, leftover_excitation=None, leftover_suppression=None, time_of_leftover=None):
    if leftover_excitation is None:
      intensity = cls.infer_intensity(params, event_types, event_times, event_types, event_times)
      excitation = cls.infer_excitation(params, event_types, event_times, event_types, event_times)
      suppression = cls.infer_suppression(params, event_types, event_times, event_types, event_times)
    else:
      intensity, next_leftover_excitation, next_leftover_suppression, next_time_of_leftover = cls.infer_intensity(params, event_types, event_times, event_types[:-1], event_times[:-1], leftover_excitation, leftover_suppression, time_of_leftover) # ignoring arbitrary number of extra return values with *_ doesn't work on tensors
      excitation, _, _ = cls.infer_excitation(params, event_types, event_times, event_types[:-1], event_times[:-1], leftover_excitation, time_of_leftover)
      suppression, _, _ = cls.infer_suppression(params, event_types, event_times, event_types[:-1], event_times[:-1], leftover_suppression, time_of_leftover)
    total_excitation = tf.math.reduce_sum(excitation, axis=-2)
    total_suppression = tf.math.reduce_sum(suppression, axis=-2)
    total_unsuppressed = tf.math.minimum(cls.unit, tf.math.maximum(cls.zero, (1.0 - total_suppression)))
    #p = excitation * total_unsuppressed[None, :] / intensity[None, :]
    #q = suppression * total_excitation[None, :] / intensity[None, :]
    #b = self.background / intensity
    #q_sum = b #tf.math.reduce_sum(p, axis=0) + b - 1.0
    #corrected_q = q * tf.math.divide_no_nan(q_sum, tf.math.reduce_sum(q, axis=0))[None, :]
    p = excitation * tf.math.divide_no_nan(total_unsuppressed, intensity)[..., None, :] # propagate batches correctly
    #q = suppression * total_excitation / intensity # let's get this straight. High Q -> suppression at actual event occurrence -> PUNISH, as a counterweight to the savings in the intensity integral. If estimated suppression coefficient is too high, then not enough punishing happened. Tiny Q's: goes negative. Giant Q's: grows larger.
    #corrected_q = tf.math.divide_no_nan(q, tf.maximum(unit, tf.math.reduce_sum(q, axis=0)))
    if leftover_excitation is None:
      b = tf.math.divide_no_nan(cls.gather_background(params.background, event_types), intensity)
      return p, b
    else:
      b = tf.math.divide_no_nan(cls.gather_background(params.background, event_types[:-1]), intensity)
      return p, b, next_leftover_excitation, next_leftover_suppression, next_time_of_leftover

  @classmethod
  @tf.function
  def filter_deltas(cls, event_times, event_lags, times, filter=None, epsilon=1e-30): # np.inf causes problems with the gradient code, which has this known propensity to generate NaNs if such numbers are anywhere near it
    if filter is None:
      filter = cls.numerical_limit # if dtype has changed at runtime, we need to extract this at the moment the function is first called
    deltas = times - (event_times + event_lags)[..., None] # max filter is 1e38. How come I kept it finite, when Inf seems to work fine?
    filtered_deltas = tf.where(deltas > epsilon, deltas, filter) # epsilon should be 0.0, but playing it extra safe. edit: can't believe this actually had an effect? how? GPU inaccuracies? Hopefully my ground-truth results still hold water. (Is this a case where signed zeros play a role?) This actually could all have been due to different random initializations.
    return filtered_deltas

  @staticmethod
  @tf.function
  def gather_from_matrix(matrix, row_indices, col_indices): # collect_* is also a nice name
    batch_dims = tf.shape(row_indices)[:-1]
    matrix_batch_dims = tf.shape(matrix)[:-2]
    tf.assert_equal(batch_dims, tf.shape(col_indices)[:-1])
    if matrix.shape[-1] == 1 and matrix.shape[-2] == 1: # support full broadcasting. looks like gather_nd does this automagically on GPUs only? ALSO: must make this static because otherwise this branch is supposed to be able to execute even when the condition is not met
      values = tf.reshape(matrix, tf.concat([matrix_batch_dims, batch_dims, [1, 1]], axis=0))
      return tf.tile(values, tf.concat([tf.ones([len(matrix_batch_dims) + len(batch_dims)], dtype=tf.int32), [len(row_indices), len(col_indices)]], axis=0)) # match the prescribed shape stated at the end of this function
    broadcast_shape = tf.concat([ # len(matrix_batch_dims) may turn out to be a Tensor in some rarified cases, if the RANK is unknown at compile time. Turns out we cannot overcome this at the gather_nd call, but the tf.ones notation looks better anyway.
        tf.ones([len(matrix_batch_dims)], dtype=tf.int32), batch_dims, [tf.shape(row_indices)[-1], tf.shape(col_indices)[-1], 1]
      ], axis=0)
    indices = tf.concat([
      tf.broadcast_to(row_indices[..., :, None, None], broadcast_shape),
      tf.broadcast_to(col_indices[..., None, :, None], broadcast_shape)], axis=-1)
    tiled_indices = tf.tile(indices,
        tf.concat([matrix_batch_dims, tf.ones([len(broadcast_shape) - len(matrix_batch_dims)], dtype=tf.int32)], axis=0) # new law: the least important bits of a program take up the most lines of code
      ) # repeat along matrix batches
    return tf.gather_nd(matrix, tiled_indices, batch_dims=len(matrix_batch_dims)) # should be matrix_batch x index_batch x len(row_indices) x len(col_indices). batching should work as-is

  @classmethod
  @tf.function
  def collect_along_dim(cls, contributions, event_types, n_types, axis): # partition and sum.
    dims = tf.shape(contributions)
    if axis < 0:
      axis = len(dims) + axis
    input = tf.transpose(contributions, tf.concat([[axis], tf.range(axis), tf.range(axis+1, len(dims))], axis=0)) # all because dynamic_partition doesn't have an axis parameter
    #type_contributions = [tf.math.reduce_sum(partition, axis=0)
    #  for partition in tf.ragged.stack_dynamic_partitions(input, event_types, n_types)]
    type_contributions = tf.math.reduce_sum(
      tf.ragged.stack_dynamic_partitions(input, event_types, n_types), axis=1) # the sum should create a normal Tensor. sometimes n_types is a Tensor that is fed in dynamically, not unrolled statically
    return tf.transpose(type_contributions, tf.concat([tf.range(axis)+1, [0], tf.range(axis+1, len(dims))], axis=0))

  @classmethod
  @tf.function
  def extend_leftover_contributions(cls, event_types, event_times, decay_rate, leftover_excitation, time_of_leftover): # leftover_excitations are batched as parameters, and also (n_types x n_types) because we allow decay rates to vary dyadically... so much batching!!! leftover_excitations and decay_rate better be batched the same way.
    event_leftovers = tf.gather(leftover_excitation, event_types, axis=-1) # batch_dims applies to batched event_types, which we don't have. axis argument is really handy!
    event_rates = tf.gather(tf.broadcast_to(decay_rate, tf.shape(leftover_excitation)), event_types, axis=-1)
    leftover_deltas = event_times - time_of_leftover
    contributions = event_leftovers * tf.exp(-event_rates * leftover_deltas) # batch by n_types by n_events
    #type_contributions = cls.collect_along_dim(contributions, event_types, n_dims, axis=-1) # partition by source type. currently I'm gathering and then parititioning, whereas I could simply partition event_times from the get go. not sure which would be faster at runtime, though...
    return contributions # contributions to each individual event

  @classmethod
  @tf.function
  def infer_excitation(cls, params, event_types, event_times, types, times, leftover_excitation=None, time_of_leftover=None): # invoke batching with the optional arguments. formally batch along (event_types, event_times), the "real data."
    if leftover_excitation is None:
      event_lags = cls.gather_background(params.lag, event_types) # borrowing this method for the lags, which are shaped in the same way
      filtered_deltas = cls.filter_deltas(event_times, event_lags, times)
      excitation_coef = cls.gather_from_matrix(params.excitation_coef, event_types, types)
      excitation_rate = cls.gather_from_matrix(params.excitation_rate, event_types, types)
    else:
      event_lags = cls.gather_background(params.lag, event_types[:-1])
      filtered_deltas = cls.filter_deltas(event_times[:-1], event_lags, times)
      excitation_coef = cls.gather_from_matrix(params.excitation_coef, event_types[:-1], types)
      excitation_rate = cls.gather_from_matrix(params.excitation_rate, event_types[:-1], types)
    excitations = excitation_coef * excitation_rate * tf.exp(-excitation_rate * filtered_deltas)
    if leftover_excitation is not None: # I'm convinced that 10% of the important logic takes 90% of the implementation...
      contributions = cls.extend_leftover_contributions(types, times, params.excitation_rate, leftover_excitation, time_of_leftover)
      contribution = tf.math.reduce_sum(contributions, axis=-2)
      final_time = event_times[-1][None] # future note: if next time_of_leftover overshoots the first event_time, that's okay: extrapolation still works
      final_deltas = cls.filter_deltas(event_times[:-1], event_lags, final_time)
      all_types = tf.range(tf.cast(params.n_dims, dtype=event_types.dtype))
      final_excitation_coef = cls.gather_from_matrix(params.excitation_coef, event_types[:-1], all_types) # STILL worth it to batch
      final_excitation_rate = cls.gather_from_matrix(params.excitation_rate, event_types[:-1], all_types)
      final_excitations = final_excitation_coef * final_excitation_rate * tf.exp(-final_excitation_rate * final_deltas)
      carryover = cls.extend_leftover_contributions(all_types, final_time, params.excitation_rate, leftover_excitation, time_of_leftover)
      final_leftovers = cls.collect_along_dim(final_excitations, event_types[:-1], params.n_dims, axis=-2) + carryover # NO LONGER for params.n_dims: weird that we're calling .numpy() inside here?? only possible because it's not a Tensor argument to the function, so this part is evaluated statically
      return tf.concat([contribution[..., None, :], excitations], axis=-2), tf.ensure_shape(final_leftovers, leftover_excitation.shape), final_time[0] # add one more excitation source
    else:
      return excitations

  @classmethod
  @tf.function
  def infer_suppression(cls, params, event_types, event_times, types, times, leftover_suppression=None, time_of_leftover=None):
    if leftover_suppression is None:
      event_lags = cls.gather_background(params.lag, event_types)
      filtered_deltas = cls.filter_deltas(event_times, event_lags, times)
      suppression_coef = cls.gather_from_matrix(params.suppression_coef, event_types, types)
      suppression_rate = cls.gather_from_matrix(params.suppression_rate, event_types, types)
    else:
      event_lags = cls.gather_background(params.lag, event_types[:-1])
      filtered_deltas = cls.filter_deltas(event_times[:-1], event_lags, times)
      suppression_coef = cls.gather_from_matrix(params.suppression_coef, event_types[:-1], types)
      suppression_rate = cls.gather_from_matrix(params.suppression_rate, event_types[:-1], types)
    suppressions = suppression_coef * suppression_rate * tf.exp(-suppression_rate * filtered_deltas)
    if leftover_suppression is not None:
      contributions = cls.extend_leftover_contributions(types, times, params.suppression_rate, leftover_suppression, time_of_leftover)
      contribution = tf.math.reduce_sum(contributions, axis=-2)
      final_time = event_times[-1][None]
      final_deltas = cls.filter_deltas(event_times[:-1], event_lags, final_time)
      all_types = tf.range(tf.cast(params.n_dims, dtype=event_types.dtype))
      final_suppression_coef = cls.gather_from_matrix(params.suppression_coef, event_types[:-1], all_types)
      final_suppression_rate = cls.gather_from_matrix(params.suppression_rate, event_types[:-1], all_types)
      final_suppressions = final_suppression_coef * final_suppression_rate * tf.exp(-final_suppression_rate * final_deltas)
      carryover = cls.extend_leftover_contributions(all_types, final_time, params.suppression_rate, leftover_suppression, time_of_leftover)
      final_leftovers = cls.collect_along_dim(final_suppressions, event_types[:-1], params.n_dims, axis=-2) + carryover
      return tf.concat([contribution[..., None, :], suppressions], axis=-2), tf.ensure_shape(final_leftovers, leftover_suppression.shape), final_time[0]
    else:
      return suppressions

  @classmethod
  @tf.function
  def infer_response(cls, params, event_types, event_times, types, times, leftover_excitation=None, leftover_suppression=None, time_of_leftover=None):
    if leftover_excitation is None:
      excitation = tf.math.reduce_sum(cls.infer_excitation(params, event_types, event_times, types, times), axis=-2)
      suppression = tf.math.reduce_sum(cls.infer_suppression(params, event_types, event_times, types, times), axis=-2)
    else:
      excitations, leftover_excitation, time_of_next_leftover = cls.infer_excitation(params, event_types, event_times, types, times, leftover_excitation, time_of_leftover)
      suppressions, leftover_suppression, _ = cls.infer_suppression(params, event_types, event_times, types, times, leftover_suppression, time_of_leftover)
      excitation = tf.math.reduce_sum(excitations, axis=-2)
      suppression = tf.math.reduce_sum(suppressions, axis=-2)
    response = excitation * tf.math.maximum(cls.zero, (1.0 - suppression))
    if leftover_excitation is None:
      return response
    else:
      return response, leftover_excitation, leftover_suppression, time_of_next_leftover

  @classmethod
  @tf.function
  def infer_intensity(cls, params, event_types, event_times, types, times, leftover_excitation=None, leftover_suppression=None, time_of_leftover=None):
    if leftover_excitation is None:
      return cls.infer_response(params, event_types, event_times, types, times) + cls.gather_background(params.background, types)
    else:
      response, leftover_excitation, leftover_suppression, time_of_leftover = \
        cls.infer_response(params, event_types, event_times, types, times, leftover_excitation, leftover_suppression, time_of_leftover)
      return response + cls.gather_background(params.background, types), leftover_excitation, leftover_suppression, time_of_leftover

  @classmethod
  @tf.function
  def infer_excitational_intensity(cls, params, event_types, event_times, types, times, leftover_excitation=None, time_of_leftover=None):
    if leftover_excitation is None:
      return tf.math.reduce_sum(
        cls.infer_excitation(params, event_types, event_times, types, times), axis=-2) + cls.gather_background(params.background, types)
    else:
      excitation, leftover_excitation, time_of_leftover = cls.infer_excitation(params, event_types, event_times, types, times, leftover_excitation, time_of_leftover)
      return tf.math.reduce_sum(excitation, axis=-2) + cls.gather_background(params.background, types), leftover_excitation, time_of_leftover

  @classmethod
  @tf.function
  def integrate_intensity(cls, params, event_types, event_times, types, ultimate_time=None, leftover_excitation=None, leftover_suppression=None, time_of_leftover=None, upper_bound=False): # last argument: which types do I integrate?
    time_interval = event_times[-1] - event_times[0]
    if ultimate_time is None:
      event_lags = cls.gather_background(params.lag, event_types)
      deltas = cls.filter_deltas(event_times, event_lags, event_times) # positive side is upper-triangular
      residual_times = event_times[-1] - (event_times + event_lags)
    else:
      event_types = event_types[:-1]
      event_times = event_times[:-1] # batches include one extra element at the end, of which we may safely dispose after finding the time interval.
      event_lags = cls.gather_background(params.lag, event_types)
      deltas = cls.filter_deltas(event_times, event_lags, event_times)
      residual_times = ultimate_time - (event_times + event_lags)
    excitation_coef_rank = len(tf.shape(params.excitation_coef))
    excitation_coef = tf.transpose(cls.gather_from_matrix(params.excitation_coef, event_types, types), [i for i in range(excitation_coef_rank-2)] + [excitation_coef_rank-1, excitation_coef_rank-2]) # rank is a tensor, so can't iterate? so much work just to go from (event_types x types) to (types x event_types), and flexibly so
    excitation_rate_rank = len(tf.shape(params.excitation_rate))
    excitation_rate = tf.transpose(cls.gather_from_matrix(params.excitation_rate, event_types, types), [i for i in range(excitation_rate_rank-2)] + [excitation_rate_rank-1, excitation_rate_rank-2]) # things are a little more nuanced now that rates are full-rank
    suppression_coef_rank = len(tf.shape(params.suppression_coef))
    suppression_coef = tf.transpose(cls.gather_from_matrix(params.suppression_coef, event_types, types), [i for i in range(suppression_coef_rank-2)] + [suppression_coef_rank-1, suppression_coef_rank-2])
    suppression_rate_rank = len(tf.shape(params.suppression_rate))
    suppression_rate = tf.transpose(cls.gather_from_matrix(params.suppression_rate, event_types, types), [i for i in range(suppression_rate_rank-2)] + [suppression_rate_rank-1, suppression_rate_rank-2])
    background = cls.gather_background(params.background, types)
    combined_rate =    excitation_rate[..., None, :] + suppression_rate[..., :, None] # batch by k by k_j (suppression) by k_i (excitation)
    corrective_rate = tf.math.divide_no_nan(excitation_rate[..., None, :] * suppression_rate[..., :, None], combined_rate)
    corrective_coef =  excitation_coef[..., None, :] * suppression_coef[..., :, None]
    corrective_scale = corrective_coef * corrective_rate
    delta_dims = tf.range(tf.rank(deltas))
    transposed_deltas = tf.transpose(deltas, tf.concat([delta_dims[:-2], [delta_dims[-1], delta_dims[-2]]], axis=0)) # with the introduction of the lag, deltas are batched now
    correction_excitation_rate =  tf.math.exp(-excitation_rate[..., None, :]  * transposed_deltas[..., None, :, :]) * (1.0 - tf.exp(-combined_rate * residual_times[..., None, :, None])) # this involves the (t_j - t_i) part, where suppression is k_j and excitation is k_i. deltas are (u,v) -> t_v - t_u, which need to transpose in order to match our (k_j,k_i), and excitations that are (... x k_i)
    #correction_suppression_rate = tf.math.exp(-suppression_rate[..., :, None] * transposed_deltas[..., None, :, :]) * (1.0 - tf.exp(-combined_rate * residual_times[..., None, None, :])) # all this mess means (batch) x k x [the rest], i.e. the None after the dots makes room for the ks
    correction_suppression_rate = tf.math.exp(-suppression_rate[..., :, None] * deltas[..., None, :, :]) * (1.0 - tf.exp(-combined_rate * residual_times[..., None, None, :])) # all this mess means (batch) x k x [the rest], i.e. the None after the dots makes room for the ks
    correction_summand = corrective_scale * (correction_excitation_rate + correction_suppression_rate)
    #summed_correction = tf.math.reduce_sum(correction_summand, axis=-1) # keep a spare k_j axis # HOW THINGS USED TO BE DONE, retained for documentary purposes
    correction_plain = excitation_coef * suppression_coef \
      * tf.math.divide_no_nan(excitation_rate * suppression_rate, (excitation_rate + suppression_rate)) \
      * (1.0 - tf.exp(-(excitation_rate + suppression_rate) * residual_times[..., None, :])) # this is the part where j and i happen to align
    corrections = tf.linalg.diag(correction_plain) + correction_summand # line everything up exactly as they should be, this time
    correction = tf.math.reduce_sum(corrections, axis=-2) # we will later want to filter batch x k x k_i, the overall contribution of each excitation, to be non-negative. events may have second-order negative contributions via their suppressive tendencies instead
    if ultimate_time is not None:
      #leftover_excitation_rank = len(tf.shape(leftover_excitation))
      #leftover_excitation_events = tf.transpose(cls.gather_from_matrix(leftover_excitation, event_types, types), [i for i in range(leftover_excitation_rank-2)] + [leftover_excitation_rank-1, leftover_excitation_rank-2])
      #leftover_suppression_rank = len(tf.shape(leftover_suppression))
      #leftover_suppression_events = tf.transpose(cls.gather_from_matrix(leftover_suppression, event_types, types), [i for i in range(leftover_suppression_rank-2)] + [leftover_suppression_rank-1, leftover_suppression_rank-2])
      all_types = tf.range(params.n_dims, dtype=types.dtype)
      leftover_excitation_events = cls.gather_from_matrix(leftover_excitation, all_types, types) # rearrange the matrices iff types is in a bizarre order
      leftover_suppression_events = cls.gather_from_matrix(leftover_suppression, all_types, types)
      leftover_excitation_rate = cls.gather_from_matrix(params.excitation_rate, all_types, types)
      leftover_suppression_rate = cls.gather_from_matrix(params.suppression_rate, all_types, types)
      #correction_leftover_excitation = tf.math.exp(-excitation_rate * (event_times+event_lags - time_of_leftover)[..., None, :]) * (1.0 - tf.exp(-(excitation_rate+suppression_rate) * residual_times[..., None, :]))
      #correction_leftover_suppression = tf.math.exp(-suppression_rate * (event_times+event_lags - time_of_leftover)[..., None, :]) * (1.0 - tf.exp(-(excitation_rate+suppression_rate) * residual_times[..., None, :])) # will need to think about this a little harder.
      correction_leftover_excitation = tf.math.exp(-leftover_excitation_rate[..., None] * (event_times+event_lags - time_of_leftover)[..., None, None, :]) \
        * (1.0 - tf.exp(-(leftover_excitation_rate[..., None] + suppression_rate[..., None, :, :]) * residual_times[..., None, None, :])) # batch by l by k by k_j. sum over l (the 'sources') immediately after
      correction_leftover_suppression = tf.math.exp(-leftover_suppression_rate[..., None] * (event_times+event_lags - time_of_leftover)[..., None, None, :]) \
        * (1.0 - tf.exp(-(excitation_rate[..., None, :, :] + leftover_suppression_rate[..., None]) * residual_times[..., None, None, :])) # batch by l by k by k_i
      #correction_leftover = (correction_leftover_excitation * leftover_excitation_events * suppression_rate * suppression_coef
      #                     + correction_leftover_suppression * leftover_suppression_events * excitation_rate * excitation_coef)
      correction_from_the_past = tf.math.reduce_sum(correction_leftover_suppression * leftover_suppression_events[..., :, :, None]
        * tf.math.divide_no_nan((excitation_rate * excitation_coef)[..., None, :, :],
          (excitation_rate[..., None, :, :] + leftover_suppression_rate[..., None])), axis=-3) # batch by k by k_i
      correction_on_the_past = tf.math.reduce_sum(correction_leftover_excitation * leftover_excitation_events[..., :, :, None]
        * tf.math.divide_no_nan((suppression_rate * suppression_coef)[..., None, :, :],
          (leftover_excitation_rate[..., None] + suppression_rate[..., None, :, :])), axis=-3) # our only source of unbounded negative contributions to an integral: current batch's suppression of a previous one's excitations. excitation_rate part is used because we need to see how the past's excitation decayed before reaching this point
      #correction += correction_leftover / (excitation_rate + suppression_rate)
      correction += correction_from_the_past
    core = excitation_coef * (1.0 - tf.exp(-excitation_rate * residual_times[..., None, :])) # batch by k by k_j (comment: shouldn't it be k_i? I think that's what it really is.)
    contribution = tf.math.maximum(core - correction, cls.zero)
    main_integral = tf.math.reduce_sum(contribution, axis=-1) # funny how the main contribution is so much more straightforward to calculate
    if ultimate_time is not None and not upper_bound: # a (possibly quite accurate) lower bound instead by taking into account the deferred corrections onto a past batch. the upper bound is guaranteed to be such albeit a lot further away from the truth, typically
      main_integral = tf.math.maximum(main_integral
        - tf.math.reduce_sum(correction_on_the_past, axis=-1), cls.zero) # finally ascribed some intuition to these obscure quantities!
    integral = main_integral + time_interval * background #tf.math.maximum(cls.zero, main_integral) + time_interval * background # hard to account for an overwhelming suppression in the integral. Make sure, with the proper bias, that this doesn't hit the bottom too often.
    return integral

  @classmethod
  @tf.function
  def numerically_integrate_intensity(cls, params, event_types, event_times, types, resolution): # for sanity checks
    time_interval = event_times[-1] - event_times[0]
    sample_times = tf.linspace(event_times[0], event_times[-1], resolution)
    integrals = tf.zeros([len(types)], dtype=cls.dtype)
    for i in tf.range(len(types)):
      sample_types = tf.tile(types[i][None], [resolution])
      intensity = cls.infer_intensity(params, event_types, event_times, sample_types, sample_times)
      integral = tf.math.reduce_sum(intensity, axis=-1) * time_interval / resolution
      integrals += integral * tf.one_hot(i, len(types), dtype=cls.dtype) # loop is unravelled, and list is constructed *statically* <-- it turns out, actually, that 'types' is implicitly converted from a numpy array to a tensor, so this is all dynamic
    return integrals

  @classmethod
  @tf.function
  def estimate_excitation_coefficient(cls, params, p, event_types, event_times, ultimate_time=None, leftover_excitation=None, leftover_suppression=None, time_of_leftover=None): # can we realistically batch this? the answer is yes. --- it looks like we can partition along params.n_dims even though it's a Tensor because it's passed inside a Python object, so it evaluates statically? make sure n_dims doesn't change dynamically!
    time_residue = event_times[-1] - event_times
    tf.debugging.assert_greater_equal(time_residue, cls.zero)
    all_types = tf.range(tf.cast(params.n_dims, dtype=event_types.dtype))
    with tf.GradientTape() as tape:
      tape.watch(params.excitation_coef)
      integral = tf.math.reduce_sum(cls.integrate_intensity(params, event_types, event_times, all_types, ultimate_time, leftover_excitation, leftover_suppression, time_of_leftover), axis=-1) # LINEAR DERIVATIVE (in each of the coefficients..). not sure if batching will interoperate well here.
    denominator = tape.gradient(integral, params.excitation_coef) # below we will partition along the last dimension of a batched p
    if ultimate_time is not None: # be careful with the trimming here...
      p = p[..., 1:, :] # leftover events are encoded
      event_types = event_types[:-1]
    sources = tf.ragged.stack_dynamic_partitions(tf.transpose(p), event_types, params.n_dims) # tf.ragged.stack(...) and then .to_tensor inside the loop are buggy as heck! are they super recent features? somehow, it works only when the loop body calls another function
    def collect_row(source): # make this a dynamic loop to avoid ballooning the computation graph
      if isinstance(source, tf.RaggedTensor):
        source = source.to_tensor() # usually ragged, but if scalar it is automatically made a tensor. inconsistent!
      if tf.size(source) == 0: # dynamic_partition can't handle empty 'data' argument
        return tf.zeros(tf.shape(params.excitation_coef)[:-1], dtype=cls.dtype) # batch x n_dims
      targets = tf.ragged.stack_dynamic_partitions(tf.transpose(source, tf.concat([[1, 0], tf.range(tf.rank(source))[2:]], axis=0)), event_types, params.n_dims) # source can be turned into a tensor now, since its dimensions are X by n_events by batch... and flipped to n_events by X by batch
      influence_row = tf.math.reduce_sum(targets, axis=(1, 2)) # tf.convert_to_tensor([tf.math.reduce_sum(target, axis=(0, 1)) for target in targets])
      return tf.transpose(influence_row) # make it batch x n_dims
    influences = tf.zeros_like(params.excitation_coef)
    for i in tf.range(params.n_dims): # really stumbled into, and struggled with an esoteric bug in iterating directly on 'sources'
      influence_row = collect_row(sources[i, ...])
      update = influence_row[..., None, :] * tf.one_hot(i, params.n_dims, dtype=cls.dtype)[:, None]
      influences = tf.ensure_shape(influences + update, influences.shape) # safeguard guarantee for when n_dims=1
    return tf.math.divide_no_nan(influences, denominator) # figure out a way to include credit for carried-over excitations in leftover_excitation? at least the integral and hopefully the causalities are calculated accurately

  @classmethod
  @tf.function
  def estimate_background(cls, params, b, event_types, event_times): # wrote this one entirely from memory
    all_types = tf.range(tf.cast(params.n_dims, dtype=event_types.dtype))
    time_interval = event_times[-1] - event_times[0]
    background_probabilities = tf.ragged.stack_dynamic_partitions(tf.transpose(b), event_types, params.n_dims)
    background_probability = tf.math.reduce_sum(background_probabilities, axis=1) # each one should be 1D, besides batching
    background = tf.math.divide_no_nan(tf.transpose(background_probability), time_interval) # maybe some batches are crazy
    return tf.ensure_shape(background, params.background.shape) # for when n_dims=1...

  @classmethod
  def create_mcmc_state(cls, initial_variables, n_samples): # creating variables has to occur outside the @tf.function
    n_chains = tf.shape(initial_variables[0])[0] # should be 1D. take first axis so that broadcasting fails in case it's not.
    sampled_params = [tf.Variable(tf.zeros(tf.concat([[n_samples], tf.shape(variable)], axis=0), dtype=cls.dtype)) for variable in initial_variables]
    sampled_log_likelihoods = tf.Variable(tf.zeros([n_samples, n_chains], dtype=cls.dtype))
    n_successes = tf.Variable(tf.zeros([n_chains], dtype=tf.int32))
    return initial_variables, n_samples, n_chains, sampled_params, sampled_log_likelihoods, n_successes

  @classmethod
  @tf.function                       # propose(...) must return three things: batched parameter proposals, as well as the (log) ratio (p(x|x_new)/p(x_new|x)). If symmetric, this is (log) unit. I thought I would fetch many proposals in parallel, and then choose one and move on. What I should do instead is simply run many chains in parallel.
  def sample_mcmc(cls, state, propose, evaluate, n_burnin, n_moments, verbose=False): # Metropolis-Hastings Markov-chain Monte Carlo (did Metropolis coin the term Monte Carlo?), with the Metropolis acceptance criterion and multiple proposals evaluated in parallel. For once, the Wikipedia article is very intuitive on the formal definition of this algorithm. Proposal can be sampled in a Gibbs style.
    #sampled_params = [initial_variables] # these are lists of the specific parameters batched along the first dimension
    #sampled_log_likelihoods = [evaluate(*initial_variables)]
    initial_variables, n_samples, n_chains, sampled_params, sampled_log_likelihoods, n_successes = state
    n_variables = len(initial_variables)
    current_params = initial_variables
    current_log_likelihood = evaluate(*initial_variables)
    start_time = tf.timestamp()
    for t in tf.range(n_burnin + n_samples): # can't make t a Tensor, so hope instead that we don't bloat the computation graph. JUST MADE IT ONE
      proposal, log_ratio = propose(*current_params) # i.e. CANDIDATE
      proposal_log_likelihood = evaluate(*proposal)
      log_probability = proposal_log_likelihood - current_log_likelihood + log_ratio
      probability = tf.math.exp(tf.math.minimum(cls.zero, log_probability))
      gamble = tf.random.uniform([n_chains], dtype=cls.dtype)
      decision = tf.logical_and(gamble <= probability, tf.logical_not(tf.math.is_nan(log_probability))) # NaNs can occur if intensity at event is 0.0
      new_log_likelihood = tf.where(decision, proposal_log_likelihood, current_log_likelihood)
      n_successes.assign_add(tf.cast(decision, dtype=tf.int32))
      #new_params = [cls.zero for _ in range(n_variables)] # list multiplication notation is weird under tf.function
      new_params = [
        tf.transpose(tf.where(decision, tf.transpose(proposal[var]), tf.transpose(current_params[var]))) # select along the first axis, which is made last in order to broadcast nicely
        for var in range(n_variables)]
      if verbose:
        tf.print(tf.timestamp() - start_time, "seconds.\t", t, "-->", n_samples, " (+", n_burnin, ")\t", "Acceptances", decision, "Cumulative", n_successes, "\n",
          "Log-probabilities", log_probability, "Proposed - current", proposal_log_likelihood - current_log_likelihood, "Proposal log-ratio", log_ratio,
          output_stream=sys.stdout)
      if t >= n_burnin:
        i = t - n_burnin
        for var in range(n_variables):
          sampled_params[var].assign_add(tf.transpose(tf.transpose(new_params[var])[..., None] * tf.one_hot(i, n_samples, dtype=cls.dtype))) # tricks to broadcast along first dimension
        sampled_log_likelihoods.assign_add(new_log_likelihood * tf.one_hot(i, n_samples, dtype=cls.dtype)[:, None])
      current_params = new_params
      current_log_likelihood = new_log_likelihood
    moments = tf.range(1, n_moments+1, dtype=cls.dtype)
    #collected_samples = [tf.stack([sample[var] for sample in sampled_params], axis=0) for var in range(n_variables)]
    sampled_moments = [tf.math.reduce_mean(tf.pow(variable[..., None], moments), axis=0) for variable in sampled_params] # sum over batch; maintain (var_dims...) x n_moments
    success_rate = tf.cast(n_successes, cls.dtype) / tf.cast(n_burnin + n_samples, cls.dtype)
    return sampled_params, sampled_log_likelihoods, success_rate, sampled_moments

  @classmethod # we create variables, so this is not a tf.function
  def sample_homogeneous_rates_and_diagonal_suppressions(cls, params, event_types, event_times, n_samples, n_burnin, n_moments, lognormal_sigma, beta_sum, tiny_lognormal_sigma, do_maximizations, rate_gamma_prior=None, batch_size=None, verbose=False): # excitations are maximized outside
    n_chains = tf.shape(params.suppression_coef)[0]
    balloon = [1, params.n_dims, params.n_dims]
    new_excitation_coef_var = tf.Variable(tf.zeros([n_chains, params.n_dims, params.n_dims], dtype=cls.dtype)) # ended up listing all the variables anyways
    new_excitation_rate_var = tf.Variable(tf.zeros([n_chains, params.n_dims, params.n_dims], dtype=cls.dtype))
    new_suppression_coef_var = tf.Variable(tf.zeros([n_chains, params.n_dims, params.n_dims], dtype=cls.dtype))
    new_suppression_rate_var = tf.Variable(tf.zeros([n_chains, params.n_dims, params.n_dims], dtype=cls.dtype))
    new_background_var = tf.Variable(tf.zeros([n_chains, params.n_dims], dtype=cls.dtype))
    new_params = cls.Parameters(
      new_excitation_coef_var,
      new_excitation_rate_var, # prevent needless recompiling: maintain the same Parameters, and update the insides as variables
      new_suppression_coef_var, # diagonalizes the last axis and turns it into two
      new_suppression_rate_var,
      new_background_var, cast=False) # in order to keep variables bound instead of just their initial values, we must not perform tensor operations before assignment to Parameter members
    @tf.function
    def load_params(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background):
      new_excitation_coef_var.assign(excitation_coef)
      new_excitation_rate_var.assign(tf.tile(excitation_rate[:, None, None], balloon))
      new_suppression_coef_var.assign(tf.linalg.diag(suppression_coef))
      new_suppression_rate_var.assign(tf.tile(suppression_rate[:, None, None], balloon))
      new_background_var.assign(background)

    @tf.function
    def propose(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background): # gamma with fixed alpha=2? set the mode to the previous value
      #new_excitation_rate = tf.random.gamma([], gamma_alpha, (gamma_alpha - 1.0)/excitation_rate, dtype=cls.dtype) # match MODES instead of EXPECTATIONS/MEANS because otherwise variables drift to 0
      new_excitation_rate = tf.math.exp(tf.math.log(excitation_rate) + lognormal_sigma*tf.random.normal([n_chains], dtype=cls.dtype))
      log_excitation_rate_ratio = tf.math.log(new_excitation_rate) - tf.math.log(excitation_rate)
      new_suppression_rate = tf.math.exp(tf.math.log(suppression_rate) + lognormal_sigma*tf.random.normal([n_chains], dtype=cls.dtype))
      log_suppression_rate_ratio = tf.math.log(new_suppression_rate) - tf.math.log(suppression_rate)
      beta_alpha = suppression_coef*suppression_rate[..., None] * beta_sum + 1.0 # mode matching. beta_sum is actually (beta_alpha+beta_beta-2). 1/suppression_rate is the upper limit
      beta_beta  = (1.0 - suppression_coef*suppression_rate[..., None]) * beta_sum + 1.0 # multiply suppression_coef by new_suppression_rate[..., None] (Gibbs) here and above to set mode to absolute suppression_coef. to keep the same proportion of suppression_rate, do old suppression_rate
      #new_suppression_coef = tf.math.exp(tf.math.log(suppression_coef) + lognormal_sigma*tf.random.normal([n_chains, params.n_dims], dtype=cls.dtype))
      #log_suppression_coef_ratio = tf.math.log(new_suppression_coef) - tf.math.log(suppression_coef)
      new_suppression_coef = sample_beta_variate(beta_alpha, beta_beta, shape=[], dtype=cls.dtype) / new_suppression_rate[..., None] # note that samples take on the shape of the parameters
      log_beta = lambda x, y: beta_sum*y*tf.math.log(x) + beta_sum*(1.0-y)*tf.math.log(1.0 - x) - tf.transpose(tf.math.lbeta(tf.transpose([beta_sum*y + 1.0, beta_sum*(1.0-y) + 1.0]))) # the last dimension in Beta([...]) is reduced upon
      log_suppression_coef_ratio = (log_beta(suppression_coef * new_suppression_rate[..., None], new_suppression_coef * new_suppression_rate[..., None])
                                  - log_beta(new_suppression_coef * suppression_rate[..., None], suppression_coef * suppression_rate[..., None]) # hierarchical. in the above line, it's all reversed, i.e. as if suppression_rate were the new sample
                                  + (tf.math.log(suppression_rate) - tf.math.log(new_suppression_rate))[..., None]) # cancels with the above? not entirely, due to broadcasting...
      #supp_beta_alpha = suppression_rate / excitation_rate * beta_sum + 1.0
      #supp_beta_beta = (1.0 - suppression_rate / excitation_rate) * beta_sum + 1.0
      #new_suppression_rate = excitation_rate * sample_beta_variate(supp_beta_alpha, supp_beta_beta, shape=[], dtype=cls.dtype) # hierarchical means we generate p(x), then p(y|x). It is NOT Gibbs, and X is not independent of Y. THIS HAS BEEN THE PROBLEMATIC PART
      #log_suppression_rate_ratio = log_beta(suppression_rate / new_excitation_rate, new_suppression_rate / new_excitation_rate) - log_beta(new_suppression_rate / excitation_rate, suppression_rate / excitation_rate) + tf.math.log(excitation_rate) - tf.math.log(new_excitation_rate)
      log_ratio = log_excitation_rate_ratio + tf.math.reduce_sum(log_suppression_coef_ratio, axis=-1) + log_suppression_rate_ratio
      if do_maximizations:
        load_params(excitation_coef, new_excitation_rate, new_suppression_coef, new_suppression_rate, background)
        if batch_size is None:
          p, b = cls.infer_causalities(new_params, event_types, event_times)
          opt_excitation_coef = cls.estimate_excitation_coefficient(new_params, p, event_types, event_times)
          opt_background = cls.estimate_background(new_params, b, event_types, event_times)
        else:
          opt_excitation_coef = tf.zeros_like(excitation_coef)
          opt_background = tf.zeros_like(background)
          leftover_excitation = tf.zeros_like(new_excitation_coef_var)
          leftover_suppression = tf.zeros_like(new_suppression_coef_var) # this is of the correctly expanded size
          time_of_leftover = event_times[0]
          for t in tf.range(0, tf.size(event_times), batch_size): # use convenience method for all this?
            batch_types = event_types[t:(t+batch_size+1)]
            batch_times = event_times[t:(t+batch_size+1)]
            n_events_in_batch = tf.cast(len(batch_times) - 1, cls.dtype)
            p, b, next_leftover_excitation, next_leftover_suppression, next_time_of_leftover = cls.infer_causalities(new_params, batch_types, batch_times, leftover_excitation, leftover_suppression, time_of_leftover)
            opt_excitation_coef += cls.estimate_excitation_coefficient(new_params, p, batch_types, batch_times, event_times[-1], leftover_excitation, leftover_suppression, time_of_leftover) * n_events_in_batch
            opt_background += cls.estimate_background(new_params, b, batch_types[:-1], batch_times) * n_events_in_batch # kind of a hack to trim types but not times
            leftover_excitation, leftover_suppression, time_of_leftover = next_leftover_excitation, next_leftover_suppression, next_time_of_leftover
          opt_excitation_coef /= tf.cast(tf.size(event_times), cls.dtype)
          opt_background /= tf.cast(tf.size(event_times), cls.dtype)
        new_excitation_coef = tf.math.exp(tf.math.log(opt_excitation_coef)
          + tiny_lognormal_sigma*tf.random.normal([n_chains, params.n_dims, params.n_dims], dtype=cls.dtype))
        new_background = tf.math.exp(tf.math.log(opt_background)
          + tiny_lognormal_sigma*tf.random.normal([n_chains, params.n_dims], dtype=cls.dtype))
        log_excitation_coef_ratio = (tf.math.log(new_excitation_coef) - tf.math.log(excitation_coef)
          - (tf.square(tf.math.log(tf.math.divide_no_nan(excitation_coef, opt_excitation_coef))) - tf.square(tf.math.log(tf.math.divide_no_nan(new_excitation_coef, opt_excitation_coef)))) / 2.0*tf.square(tiny_lognormal_sigma))
        log_background_ratio = tf.math.log(new_background) - tf.math.log(background)
        log_ratio += tf.math.reduce_sum(log_excitation_coef_ratio, axis=(-1, -2)) + tf.math.reduce_sum(log_background_ratio, axis=-1)
      else:
        new_excitation_coef = excitation_coef # simply carry over
        new_background = background
      return [new_excitation_coef, new_excitation_rate, new_suppression_coef, new_suppression_rate, new_background], log_ratio

    @tf.function
    def evaluate(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background): # if tf.function can't convert, pass state explicitly and make these functions class-level
      load_params(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background)
      if verbose: tf.print(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background, sep="\n\n", output_stream=sys.stdout)
      log_likelihood = cls.compute_log_likelihood(new_params, event_types, event_times, batch_size=batch_size)
      if rate_gamma_prior is not None: # excitation_rate and suppression_rate have the same gamma prior to avoid degenerate (wrt suppression) HUGE rates and TINY coefficients, or TINY rates and HUGE coefficients. tuple of (alpha, beta)
        log_prior = (rate_gamma_prior[0]-1.0) * (tf.math.log(excitation_rate) + tf.math.log(suppression_rate)) - rate_gamma_prior[1] * (excitation_rate + suppression_rate) # sans additive constants. essentially penalize tiny and huge
        return log_likelihood + log_prior
      else: # stylistic choice to include the else
        return log_likelihood
    initial_variables = [params.excitation_coef, params.excitation_rate, params.suppression_coef, params.suppression_rate, params.background]
    sampled_variables, sampled_log_likelihoods, n_successes, moments = cls.sample_mcmc(cls.create_mcmc_state(initial_variables, n_samples), propose, evaluate, n_burnin, n_moments, verbose=verbose)
    return sampled_variables, sampled_log_likelihoods, n_successes, moments

  @classmethod
  def stein_sample_homogeneous_rates_and_diagonal_suppressions(cls, params, event_types, event_times, learning_rate, n_iterations,
      rate_kernel_bandwidth, suppression_kernel_bandwidth, batch_size, optimize_excitations, excitation_kernel_bandwidth=None, lag=0.0,
      cross_suppressions=False, excess_suppression_reg=None, do_newton=False, fisher_subsample=None, conservative_fisher_subsample=False,
      eigval_floor=1e-5, learn_hawkes=False, hawkes_stability_reg=0.0, nonstationary_decay_exponent=None, permanent_suppression_rate=None,
      n_bandwidth_samples=None, bandwidth_sample_radius=None, subsample_for_bandwidths=None, bandwidth_iteration_interval=1, rate_lower_bound=1e-20,
      suppression_prior_strength=None, suppression_prior_sharpness=0.0, horseshoe_precision=0.0, parallel_bundle_size=1, reporting_tag="", verbose=False,
      kernel_verbosity_freq=0.0):
    n_particles = tf.shape(params.suppression_coef)[0]
    balloon = [params.n_dims, params.n_dims]

    #assert tf.rank(event_types) == 1 or tf.rank(event_types) == 2 # we support first-order batching of event streams for an expected log-likelihood
    do_likelihood_expectation = isinstance(event_types, list)
    if do_likelihood_expectation:
      n_records = len(event_types)
      event_types = tf.ragged.stack(event_types, axis=0)
      event_times = tf.ragged.stack(event_times, axis=0)

    if nonstationary_decay_exponent is not None:
      assert nonstationary_decay_exponent > 0.0
      weigh = lambda tau: tf.math.pow(cls.unit + tau, -nonstationary_decay_exponent) # power-law decay for real time units, as how far in the past we have the batch's start. hence always positive. I add one to tau to ensure that the weight is always at most unit.
    else:
      weigh = None

    rolling_excit_coef_kernel = tf.Variable(cls.zero) # update for exponential-decay moving average: x_t = (1-a)*y_t + a*x_{t-1}. unbiased x_t = x_t / (1 - a^t)
    rolling_excit_rate_kernel = tf.Variable(cls.zero)
    rolling_suppr_coef_kernel = tf.Variable(cls.zero)
    rolling_suppr_rate_kernel = tf.Variable(cls.zero)
    rolling_background_kernel = tf.Variable(cls.zero)
    rolling_memory_rate = tf.cast(1.0 - 1e-4, cls.dtype) # Ever since I incorporated this logging, the output on all the kernels gets mixed in with some particle updates. I guess because strict ordering is enforced between kernel calls now, the scheduler finds it opportune to run other stuff in parallel instead of all the kernels at once.
    def update_rolling_average(var, value):
      var.assign((1.0-rolling_memory_rate)*tf.math.reduce_mean(value) + rolling_memory_rate*var)
    n_rolling_average_updates = tf.Variable(0)

    excitation_kernel_bandwidth, suppression_kernel_bandwidth, rate_kernel_bandwidth = (tf.Variable(excitation_kernel_bandwidth) if not optimize_excitations else None), tf.Variable(suppression_kernel_bandwidth), tf.Variable(rate_kernel_bandwidth)
    def set_bandwidths(bandwidths):
      if not optimize_excitations: excitation_kernel_bandwidth.assign(bandwidths[0])
      suppression_kernel_bandwidth.assign(bandwidths[1])
      rate_kernel_bandwidth.assign(bandwidths[2])
      if verbose:
        tf.print("Set the new bandwidths: excitation ->", excitation_kernel_bandwidth,
          "\b, suppression ->", suppression_kernel_bandwidth,
          "\b, rate ->", rate_kernel_bandwidth, "\b.",
          output_stream=sys.stdout)

    def explore_bandwidths(): # 'suggest' or 'propose'
      sample_lognormal = lambda param: tf.math.exp(tf.random.normal([n_bandwidth_samples], dtype=cls.dtype)*tf.cast(bandwidth_sample_radius, cls.dtype) + tf.math.log(param)) # median is the original value here
      sample_lognormal_whilst_preserving = lambda param: tf.concat([[param], sample_lognormal(param)], axis=-1) # one dimension usually...
      excitation_bandwidth = sample_lognormal_whilst_preserving(excitation_kernel_bandwidth) if not optimize_excitations else None
      suppression_bandwidth = sample_lognormal_whilst_preserving(suppression_kernel_bandwidth)
      rate_bandwidth = sample_lognormal_whilst_preserving(rate_kernel_bandwidth)
      return [excitation_bandwidth, suppression_bandwidth, rate_bandwidth]

    def evaluate_rate_softplus(x):
      return tf.math.softplus(x) + tf.cast(rate_lower_bound, cls.dtype) # cap it while never disconnecting the gradients

    @tf.function
    def evaluate_kernel(a, b, bandwidths=[excitation_kernel_bandwidth, suppression_kernel_bandwidth, rate_kernel_bandwidth]): # lists of parameters. bandwidths can be overridden for adaptive heuristics; supports 1D-batched bandwidths. Currently, either the bandwidths can be batched or the parameters, but not both at the same call. That is fine for now.
      excitation_kernel_bandwidth, suppression_kernel_bandwidth, rate_kernel_bandwidth = bandwidths
      safe_recip = lambda x: tf.math.divide_no_nan(tf.ones_like(x), x) # say a rate is 1e-20. log is much more stable than reciprocal here.
      excitation_rate_prox  = tf.math.exp(-tf.square(tf.math.log(evaluate_rate_softplus(a[1])) - tf.math.log(evaluate_rate_softplus(b[1]))) / (2.0*tf.square(rate_kernel_bandwidth)))
      suppression_rate_prox = tf.math.exp(-tf.square(tf.math.log(evaluate_rate_softplus(a[3])) - tf.math.log(evaluate_rate_softplus(b[3]))) / (2.0*tf.square(rate_kernel_bandwidth)))
      if not cross_suppressions:
        suppression_coef_prox = tf.math.exp(-tf.square(tf.math.sigmoid(a[2]) - tf.math.sigmoid(b[2])) / (2.0*tf.square(suppression_kernel_bandwidth[..., None]))) # [None, ...] on a scalar creates a singular dimension. on a vector it creates a second dimension.
        agg_suppression_coef_prox = tf.math.reduce_prod(suppression_coef_prox, axis=-1)
      else:
        suppression_coef_prox = tf.math.exp(-tf.square(tf.math.sigmoid(a[2]) - tf.math.sigmoid(b[2])) / (2.0*tf.square(suppression_kernel_bandwidth[..., None, None]))) # [None, ...] on a scalar creates a singular dimension. on a vector it creates a second dimension.
        agg_suppression_coef_prox = tf.math.reduce_prod(suppression_coef_prox, axis=[-1, -2])
      if tf.rank(suppression_kernel_bandwidth) == 0:
        #nonlocal n_rolling_average_updates # this here seems to impair our ability to make evaluate_kernel a @tf.function
        n_rolling_average_updates.assign_add(1)
        update_rolling_average(rolling_suppr_coef_kernel, tf.math.log(agg_suppression_coef_prox)) # presently, this is a scalar so the averaging (reduce_mean) is unnecessary. but that may change in the future.
        update_rolling_average(rolling_suppr_rate_kernel, tf.math.log(suppression_rate_prox))
        update_rolling_average(rolling_excit_rate_kernel, tf.math.log(excitation_rate_prox))
      if not learn_hawkes:
        if permanent_suppression_rate is not None:
          accum_prod = (agg_suppression_coef_prox * excitation_rate_prox + cls.zero*suppression_rate_prox)
        else:
          accum_prod = (agg_suppression_coef_prox * suppression_rate_prox * excitation_rate_prox)
      else:
        accum_prod = (excitation_rate_prox + cls.zero*agg_suppression_coef_prox + cls.zero*suppression_rate_prox)
      accum_prod += cls.zero * tf.math.reduce_sum(a[4], axis=-1) # connect the background gradients vacuously
      if optimize_excitations:
        return accum_prod + cls.zero*tf.math.reduce_sum(a[0], axis=[-2, -1])
      else:
        excitation_coef_prox = tf.math.exp(-tf.square(tf.math.softplus(a[0]) - tf.math.softplus(b[0])) / (2.0*tf.square(excitation_kernel_bandwidth[..., None, None])))
        agg_excitation_coef_prox = tf.math.reduce_prod(excitation_coef_prox, axis=[-2, -1])
        if tf.rank(excitation_kernel_bandwidth) == 0:
          update_rolling_average(rolling_excit_coef_kernel, tf.math.log(agg_excitation_coef_prox)) # geometric rolling average
        return agg_excitation_coef_prox * accum_prod

    def pack_parameters(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background):
      scaled_suppression_coef = tf.math.sigmoid(suppression_coef) # IF CONSTRAINED LINEAR SCALE: make the sigmoid not steep. a bug that took forever to trace (because I didn't shine light into this matrix) occurred from doing the linalg.diag part BEFORE transforming, giving nonzero values to the nondiagonal entries
      rate_batch_size = tf.rank(excitation_rate)
      lag_param = tf.fill(tf.shape(background), tf.cast(lag, cls.dtype)) # no batching when it comes to our Stein evaluator, but there is batching in other places
      our_balloon = tf.concat([tf.ones([rate_batch_size], tf.int32), balloon], axis=0)
      if not cross_suppressions:
        real_suppression_coef = tf.linalg.diag(tf.math.divide_no_nan(scaled_suppression_coef, tf.math.softplus(suppression_rate)[..., None]))
      else:
        real_suppression_coef = tf.math.divide_no_nan(scaled_suppression_coef, tf.math.softplus(suppression_rate)[..., None, None])
      params = cls.Parameters( # DO GRADIENT DESCENT ON LOG-PARAMETERS, WHICH MUST ALL BE POSITIVE
        tf.math.softplus(excitation_coef),
        tf.squeeze(evaluate_rate_softplus(tf.tile(excitation_rate[..., None, None], our_balloon))), # dots to enable optional batching. squeeze is necessary if e.g. excitation_rate has initial shape [1]
        tf.squeeze(real_suppression_coef),
        tf.squeeze(evaluate_rate_softplus(tf.tile(suppression_rate[..., None, None], our_balloon))), # be careful with the careless squeeze operations. we don't want to induce anything unexpected.
        tf.math.softplus(background), lag_param)
      if learn_hawkes:
        return cls.Parameters(params.excitation_coef, params.excitation_rate,
          tf.zeros_like(params.suppression_coef), tf.ones_like(params.suppression_rate), params.background, lag_param) # all else falls into place
      if permanent_suppression_rate is not None: # accommodate the more rudimentary HawkesN
        return cls.Parameters(params.excitation_coef, params.excitation_rate, # we don't really mind the extra degree of freedom in the otherwise unused "suppression_rate"
          params.suppression_coef / permanent_suppression_rate,
          permanent_suppression_rate * tf.ones_like(params.suppression_rate),
          params.background, lag_param)
      return params

    def print_params(params):
      if verbose:
        tf.print(
          tf.cast(tf.debugging.check_numerics(params.excitation_coef, "excitation coefficients"), tf.float32), "excitation coefficients\n",
          tf.cast(tf.debugging.check_numerics(params.excitation_rate[0, 0], "excitation rate"), tf.float32), "excitation rate\n",
          tf.cast(tf.debugging.check_numerics(
            tf.linalg.diag_part(params.suppression_coef) if not cross_suppressions else params.suppression_coef,
            "suppression coefficients"), tf.float32), "suppression coefficients\n",
          tf.cast(tf.debugging.check_numerics(params.suppression_rate[0, 0], "suppression rate"), tf.float32), "suppression rate\n",
          tf.cast(tf.debugging.check_numerics(params.background, "background intensities"), tf.float32), "background intensities",
          output_stream=sys.stdout)

    @tf.function
    def compute_log_prior(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background): # does not support batched parameters. operates on REAL and CONCRETE, not projected/nominal, parameters
      log_prior = cls.zero
      if not learn_hawkes and suppression_prior_strength is not None and permanent_suppression_rate is None: # suppression_prior_strength directly scales the *log* prior
        goalpost = tf.math.log(excitation_rate)
        value = tf.math.log(suppression_rate)
        sharpness = tf.cast(suppression_prior_sharpness, cls.dtype)
        logistic = tf.cast(suppression_prior_strength, cls.dtype) - tf.math.softplus(sharpness*(value - goalpost)) # positive sharpness encourages (value - goalpost) << 0 <==> suppression_rate << excitation_rate
        soft_log_uniform = tf.math.softplus(logistic) - value # suppression_rate is hereby log-uniformly distributed heterogeneously, depending on which side of the excitation_rate it falls. (- value) preserves volume in the log domain. softplus is a stable and battle-tested way to compute log(1 + <possibly huge number that can't be represented on its own>)
        log_prior += tf.math.reduce_sum(soft_log_uniform, axis=[-1, -2]) # currently, this is summed across the full (n_types x n_types) matrix
      if learn_hawkes and hawkes_stability_reg > 0.0: # multiply by number of events to match the scale of the likelihood? not always directly translatable...
        eigvals = tf.linalg.eigvals(excitation_coef) # complex, but imaginary part is zero when excitations are all non-negative (is this true? verify...)
        spectral_radius = tf.math.reduce_max(tf.math.abs(eigvals), axis=-1)
        log_prior -= tf.cast(hawkes_stability_reg, cls.dtype) * tf.math.maximum(cls.zero, spectral_radius - cls.unit) # Laplace-like prior on radius excess of unit to induce stability. don't have to subtract by one as long as derivative is constant when radius is too small
      if horseshoe_precision > 0.0:
        def horseshoe_log_prob(scale, x): # I get various import errors with tfp-nightly in connection to tf-nightly-gpu, so I extracted this code from its source and plastered it here wholesale.
          xx = (x / scale)**2 / 2
          g = 0.5614594835668851  # tf.exp(-0.5772156649015328606)
          b = 1.0420764938351215   # tf.sqrt(2 * (1-g) / (g * (2-g)))
          h_inf = 1.0801359952503342  #  (1-g)*(g*g-6*g+12) / (3*g * (2-g)**2 * b)
          q = 20. / 47. * xx**1.0919284281983377
          h = 1. / (1 + xx**(1.5)) + h_inf * q / (1 + q)
          c = -.5 * np.log(2 * np.pi**3) - tf.math.log(g * scale)
          z = np.log1p(-g) - np.log(g)
          return -tf.math.softplus(z - xx / (1 - g)) + tf.math.log(
            tf.math.log1p(g / xx - (1 - g) / (h + b * xx)**2)) + c
        def threshold(x, sparsity_threshold=1e-5): # these values are always positive, but whatever. treat values below 1e-5 as zero for the prior's intents and purposes: avoid squeezing the softplus too hard. we need cut it slack so that the HESSIAN stays finite.
          return tf.math.sign(x) * tf.math.maximum(tf.math.abs(x), tf.cast(sparsity_threshold, x.dtype)) # setting anything below a threshold to zero, as I did before, disconnects the gradients but makes all horseshoe values nan.
        horseshoe_scale = 1.0 / tf.cast(horseshoe_precision, cls.dtype) # unbatched parameters in this distribution
        log_prior += tf.math.reduce_sum(
          horseshoe_log_prob(horseshoe_scale, threshold(excitation_coef)),
          axis=[-1, -2])
        log_prior += tf.math.reduce_sum(
          horseshoe_log_prob(horseshoe_scale,
            # since the shrinkage prior disregards values less than about 1e-5, I should
            # refrain from multiplying by the potentially tiny suppression rate
            threshold(suppression_coef)),# * tf.stop_gradient(suppression_rate))),
          axis=[-1, -2]) # should be agnostic to specific scales, roughly
      return log_prior

    @tf.function
    def compute_log_likelihood(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background):
      if optimize_excitations:
        new_params = pack_parameters(
          tf.stop_gradient(excitation_coef), excitation_rate, suppression_coef, suppression_rate, tf.stop_gradient(background)) # I HOPE that the values of these aren't stuck statically due to Tensorflow's obstinance with Python objects (which somehow can be ameliorated by inheriting from a namedtuple? see the Limitations doc on Github)
      else:
        new_params = pack_parameters(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background)
      print_params(new_params)
      log_prior = compute_log_prior(new_params.excitation_coef, new_params.excitation_rate, new_params.suppression_coef, new_params.suppression_rate, new_params.background)
      if not do_likelihood_expectation:
        log_likelihood = cls.compute_log_likelihood(new_params, event_types, event_times,
          batch_size=batch_size, excess_suppression_reg=excess_suppression_reg, weigh=weigh)
        return log_likelihood + log_prior
      else:
        compute_logL = lambda p: \
          cls.compute_log_likelihood(new_params, p[0], p[1], batch_size=batch_size, excess_suppression_reg=excess_suppression_reg, weigh=weigh)
        log_likelihoods = tf.map_fn(compute_logL, (event_types, event_times), fn_output_signature=cls.dtype) # map_fn supports RaggedTensors and unstacks to Tensors implicitly, when possible
        log_likelihood = tf.math.reduce_logsumexp(log_likelihoods, axis=0) - tf.math.log(tf.cast(n_records, cls.dtype))
        return log_likelihood + log_prior

    @tf.function # RECALL that for massive efficiency gains, intensities remain exact but gradients are only computed on each batch, forgetting distant influencers
    def compute_log_likelihood_gradient(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background):
      new_params = pack_parameters(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background) # I HOPE that the values of these aren't stuck statically due to Tensorflow's obstinance with Python objects (which somehow can be ameliorated by inheriting from a namedtuple? see the Limitations doc on Github)
      print_params(new_params)
      if do_likelihood_expectation:
        def compute_logL_and_gradient(p):
          return cls.compute_log_likelihood(new_params, p[0], p[1],
            batch_size=batch_size, do_batched_gradient=True, excess_suppression_reg=excess_suppression_reg, weigh=weigh)
        log_likelihoods, gradients = tf.map_fn(compute_logL_and_gradient, (event_types, event_times),
          fn_output_signature=(cls.dtype, [tf.TensorSpec(tf.TensorShape([None]).concatenate(p.shape), cls.dtype) for p in (
            new_params.excitation_coef, new_params.excitation_rate, new_params.suppression_coef, new_params.suppression_rate, new_params.background)]))
        weights = tf.math.softmax(log_likelihoods) # according to WolframAlpha experiments, the derivative of the logsumexp operator is what we call the softmax
        weighted_gradients = [tf.transpose(tf.transpose(weights)[..., None, :] * tf.transpose(g)) for g in gradients] # cheap (for the programmer) way to multiply along axis 0. transpose the weights because in case they are records x batch, they must be flipped to match. the None in between is for the per-batch gradients
        gradient = [tf.math.reduce_sum(wg, axis=0) for wg in weighted_gradients] # even IF parameters are batched by the new particle bundle technique, map_fn stacks gradients on axis 0
      else:
        _, gradient = cls.compute_log_likelihood(new_params, event_types, event_times,
          batch_size=batch_size, do_batched_gradient=True, excess_suppression_reg=excess_suppression_reg, weigh=weigh) # gradient wrt unpacked variables
      suppression_coef_deriv_raw = tf.linalg.diag_part(gradient[2], k=0) if not cross_suppressions else gradient[2]
      suppression_coef_inner = suppression_coef #* tf.math.softplus(suppression_rate) # IF CONSTRAINED LINEAR SCALE
      def logistic_deriv(x, limit=30.0): # I keep relearning this paradigm: to safely differentiate, must filter the INPUT before NaNs even have a chance of sprouting up. Also, it's fine that the derivative caps out at zero beyond a certain point: that's how it would naturally be.
        y = tf.math.minimum(x, tf.cast(limit, x.dtype)) # tf.where(x < limit, x, tf.cast(limit, x.dtype)) # numerical stability! make sure tf.square(1.0 + tf.exp(30.0)) is finite in the chosen floating-point representation. I have tested this on float32 and float64 only.
        return tf.exp(y) / tf.square(1.0 + tf.exp(y)) # softplus(suppression_rate) is the TRUE suppression_rate, unlike tf.exp(suppression_coef). IF CONSTRAINED LINEAR SCALE (how it was previously): DON'T DIVIDE, BECAUSE THE PRODUCT INSIDE CANCELS THE OUTER DIVISION (sigmoid(x * rate) / rate derivative is sigmoid'(x * rate)). Currently I'm dividing on the line below.
      suppression_coef_transform = tf.math.divide_no_nan(
        logistic_deriv(suppression_coef_inner), tf.math.softplus(
          suppression_rate[..., None, None] if cross_suppressions else suppression_rate[..., None]))
      def get_covariance(deriv): # this has NOT been modified to support batched parameters
        if deriv is None:
          return None
        scale = tf.cast(tf.shape(deriv)[0], dtype=cls.dtype)
        flat_deriv = tf.reshape(deriv, [tf.shape(deriv)[0], -1])
        product = tf.linalg.matmul(flat_deriv[:, :, None], flat_deriv[:, None, :]) # effectively a batched outer product
        sum = tf.math.reduce_sum(flat_deriv, axis=0)
        return tf.math.reduce_sum(product, axis=0)*scale - tf.tensordot(sum, sum, axes=0) # multiply by scale if each piece is an estimate of gradient/scale, then sum_outerproduct(gradient/scale) is off by 1/(scale**2) * scale
      if optimize_excitations:
        excitation_coef_deriv = None
        background_deriv = None
      else:
        excitation_coef_deriv = gradient[0] * tf.math.sigmoid(excitation_coef) # derivative of softplus is sigmoid
        background_deriv = gradient[4] * tf.math.sigmoid(background)
      excitation_rate_deriv = tf.math.reduce_sum(gradient[1], axis=[-2, -1]) * tf.math.sigmoid(excitation_rate) # I had erroneously put axis=0 for the longest time
      suppression_coef_deriv = suppression_coef_deriv_raw * suppression_coef_transform
      suppression_rate_deriv = tf.math.reduce_sum(gradient[3], axis=[-2, -1]) * tf.math.sigmoid(suppression_rate)

      with tf.GradientTape(watch_accessed_variables=False, persistent=True) as prior_tape: # we will only call this into action in the do_newton branch. no harm is done otherwise
        prior_tape.watch([excitation_coef, excitation_rate, suppression_coef, suppression_rate, background])
        new_params_for_prior = pack_parameters(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background) # same thing with the original packing, but traced this time
        log_prior = compute_log_prior(new_params_for_prior.excitation_coef, new_params_for_prior.excitation_rate,
          new_params_for_prior.suppression_coef, new_params_for_prior.suppression_rate, new_params_for_prior.background)
        excitation_coef_prior_grad, = tf.gradients(log_prior, excitation_coef, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        excitation_rate_prior_grad, = tf.gradients(log_prior, excitation_rate, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        suppression_coef_prior_grad, = tf.gradients(log_prior, suppression_coef, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        suppression_rate_prior_grad, = tf.gradients(log_prior, suppression_rate, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        background_prior_grad, = tf.gradients(log_prior, background, unconnected_gradients=tf.UnconnectedGradients.ZERO)

      derivatives = [
        (tf.math.reduce_sum(excitation_coef_deriv, axis=0) + excitation_coef_prior_grad) if excitation_coef_deriv is not None else None,
        tf.math.reduce_sum(excitation_rate_deriv, axis=0) + excitation_rate_prior_grad, # again, bundled parameter batches do not affect the fact that gradients here must be summed along axis 0, the per-event-batch gradients
        tf.math.reduce_sum(suppression_coef_deriv, axis=0) + suppression_coef_prior_grad,
        tf.math.reduce_sum(suppression_rate_deriv, axis=0) + suppression_rate_prior_grad,
        (tf.math.reduce_sum(background_deriv, axis=0) + background_prior_grad) if background_deriv is not None else None]
      if do_newton: # according to the Stein Newton Online Bayesian Hawkes paper, who cites Alex Reinhardt's review, the expected Fisher information is a p.s.d approximation for the true Hessian. Note that these conditions might not hold after we apply our chain rule to the projected space. Neal Lawton recommends that I either add an identity*lambda where lambda is greater in magnitude than the most negative eigenvalue, or I transform the eigenvalues to their absolute values via SVD
        # we currently don't have priors like the excess suppression penalty factored in, although the gradients do.
        #covariances = [get_covariance(excitation_coef_deriv), get_covariance(excitation_rate_deriv), get_covariance(suppression_coef_deriv),
        #  get_covariance(suppression_rate_deriv), get_covariance(background_deriv)]

        parameters_are_bundled = len(excitation_coef.shape) == 3 # very specific type of batching supported
        parameter_bundle_size = excitation_coef.shape[0] if parameters_are_bundled else None
        reshape_spec = [-1] if not parameters_are_bundled else [parameter_bundle_size, -1]

        if False: # these are excessively slow, sadly.
          excitation_coef_prior_hess = tf.reshape(prior_tape.jacobian(excitation_coef_prior_grad, excitation_coef,
              unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False),
            [tf.size(excitation_coef), tf.size(excitation_coef)])
          excitation_rate_prior_hess = tf.reshape(prior_tape.jacobian(excitation_rate_prior_grad, excitation_rate,
              unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False),
            [tf.size(excitation_rate), tf.size(excitation_rate)])
          suppression_coef_prior_hess = tf.reshape(prior_tape.jacobian(suppression_coef_prior_grad, suppression_coef,
              unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False),
            [tf.size(suppression_coef), tf.size(suppression_coef)])
          suppression_rate_prior_hess = tf.reshape(prior_tape.jacobian(suppression_rate_prior_grad, suppression_rate,
              unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False),
            [tf.size(suppression_rate), tf.size(suppression_rate)])
          background_prior_hess = tf.reshape(prior_tape.jacobian(background_prior_grad, background,
              unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False),
            [tf.size(background), tf.size(background)])
        else: # diagonal approximation (not even block-diagonal)
          excitation_coef_prior_hess = tf.linalg.diag(tf.reshape(
              prior_tape.gradient(excitation_coef_prior_grad, excitation_coef, unconnected_gradients=tf.UnconnectedGradients.ZERO),
              [parameter_bundle_size, tf.size(excitation_coef[0, ...])] if parameters_are_bundled else [tf.size(excitation_coef)]))
          excitation_rate_prior_hess = tf.linalg.diag(tf.reshape(
              prior_tape.gradient(excitation_rate_prior_grad, excitation_rate, unconnected_gradients=tf.UnconnectedGradients.ZERO),
              [parameter_bundle_size, tf.size(excitation_rate[0, ...])] if parameters_are_bundled else [tf.size(excitation_rate)]))
          suppression_coef_prior_hess = tf.linalg.diag(tf.reshape(
              prior_tape.gradient(suppression_coef_prior_grad, suppression_coef, unconnected_gradients=tf.UnconnectedGradients.ZERO),
              [parameter_bundle_size, tf.size(suppression_coef[0, ...])] if parameters_are_bundled else [tf.size(suppression_coef)]))
          suppression_rate_prior_hess = tf.linalg.diag(tf.reshape(
              prior_tape.gradient(suppression_rate_prior_grad, suppression_rate, unconnected_gradients=tf.UnconnectedGradients.ZERO),
              [parameter_bundle_size, tf.size(suppression_rate[0, ...])] if parameters_are_bundled else [tf.size(suppression_rate)]))
          background_prior_hess = tf.linalg.diag(tf.reshape(
              prior_tape.gradient(background_prior_grad, background, unconnected_gradients=tf.UnconnectedGradients.ZERO),
              [parameter_bundle_size, tf.size(background[0, ...])] if parameters_are_bundled else [tf.size(background)]))

        if do_likelihood_expectation:
          def compute_fisher(p):
            return cls.estimate_fisher_information(new_params, p[0], p[1], batch_size,
              subsample_size=fisher_subsample, robust_subsample=conservative_fisher_subsample, excess_suppression_reg=excess_suppression_reg) # other priors are not yet incorporated
          prod = lambda v: reduce(lambda a, x: a*x, v)
          unweighted_raw_covariances = tf.map_fn(compute_fisher, (event_types, event_times),
            fn_output_signature=[tf.TensorSpec(
                [parameter_bundle_size, prod(p.shape[1:]), prod(p.shape[1:])] if parameters_are_bundled else [prod(p.shape), prod(p.shape)], cls.dtype)
            for p in (
              new_params.excitation_coef, new_params.excitation_rate, new_params.suppression_coef, new_params.suppression_rate, new_params.background)])
          weighted_raw_covariances = [tf.transpose(tf.transpose(weights) * tf.transpose(rc)) for rc in unweighted_raw_covariances]
          raw_covariances = [tf.math.reduce_sum(wrc, axis=0) for wrc in weighted_raw_covariances]
        else:
          raw_covariances = cls.estimate_fisher_information(new_params, event_types, event_times, batch_size,
            subsample_size=fisher_subsample, robust_subsample=conservative_fisher_subsample, excess_suppression_reg=excess_suppression_reg) # multiply this by outer product of chain-rule transformations. validity follows from gradient-covariance approximation of hessian
        if not cross_suppressions:
          if parameters_are_bundled: # quite the elusive bug was located here, wherein small particle bundles would innocuously weave into the wrong parts of this batched matrix
            unscaled_suppression_coef_cov = tf.transpose(tf.linalg.diag_part(
              tf.transpose(tf.linalg.diag_part(
                tf.reshape(-raw_covariances[2],
                  [parameter_bundle_size, new_params.n_dims, new_params.n_dims, new_params.n_dims, new_params.n_dims])),
                [0, 3, 2, 1])),
              [0, 2, 1]) # in innumerable occasions I've wanted a convenient function to do batched transposes. do they force you to write out the explicit dimensions on purpose? variable sizes are a major pain
          else:
            unscaled_suppression_coef_cov = tf.transpose(tf.linalg.diag_part(
              tf.transpose(tf.linalg.diag_part(
                tf.reshape(-raw_covariances[2],
                  [new_params.n_dims, new_params.n_dims, new_params.n_dims, new_params.n_dims]))))) # gather diagonals. diag_part turns the last two square dims into the diagonal
        else:
          unscaled_suppression_coef_cov = -raw_covariances[2] # NOTE: THESE SHALL ALL BE NEGATED COVARIANCES, I.E. HESSIAN APPROXIMATIONS (VIA FISHER INFORMATION)
        # See https://en.wikipedia.org/wiki/Chain_rule#Higher_derivatives_of_multivariable_functions.
        # I will try to recreate the second term as closely as possible. As explained earlier, "WE DO IT THIS WAY BECAUSE THE HIGHER-ORDER DERIVATIVES OF THE PROJECTION FUNCTIONS SHOULD BE TINY."
        # \partial u_k / \partial x_i = 0 when i != k, thus simplifying the summation. So it looks like the outer product of the projection's gradients is sufficient.
        # A similar logic simplifies the other (what I call the first-order) portion of the chain rule formula of second order: \partial^2 u_k / \partial x_i \partial x_j = 0 if i != k or j != k.
        # Hence this shall augment the diagonal of the Hessian with the gradient times the second derivative of the projection...
        logistic_second_deriv = lambda x: tf.gradients(logistic_deriv(x), x)[0] # an element in x only affects its corresponding entry, so gradient on the sum is fine.
        suppression_coef_gradient = tf.math.reduce_sum(suppression_coef_deriv_raw, axis=0) # gradients must be aggregated as per above
        suppression_coef_cov_logistic_second_deriv = tf.reshape(suppression_coef_gradient * logistic_second_deriv(suppression_coef), reshape_spec)
        suppression_coef_cov_logistic_deriv = tf.reshape(logistic_deriv(suppression_coef), reshape_spec) # multivariate version of Faa di Bruno's formula: second derivative times two first-order derivatives of the "inputs".
        suppression_coef_cov = tf.math.divide_no_nan(tf.linalg.diag(suppression_coef_cov_logistic_second_deriv) \
          + unscaled_suppression_coef_cov * suppression_coef_cov_logistic_deriv[..., :, None] * suppression_coef_cov_logistic_deriv[..., None, :],
          tf.square(tf.math.softplus(suppression_rate))[..., None, None]) + suppression_coef_prior_hess # This guy's eigenvalues can be really negative! No rough simplifying in this line---don't neglect the sigmoid constraint part. Near zero the distortion is not negligible.
        suppression_rate_gradient = tf.reshape(tf.math.reduce_sum(gradient[3], axis=0), reshape_spec) # ^ what is d(sigmoid(coef) / rate)/d(coef) ?
        suppression_rate_cov = -raw_covariances[3] * tf.math.maximum(1.0/cls.numerical_limit, # prevent potential infs in the raw_covariances from being multiplied by zero. the other possible issue is -np.inf + np.inf -> nan, rendering it absolutely necessary to avoid nans in the prior and gradient
            tf.square(tf.math.sigmoid(suppression_rate))[..., None, None]) \
          + logistic_deriv(suppression_rate)[..., None, None] * tf.linalg.diag(suppression_rate_gradient) + suppression_rate_prior_hess
        # option: make the softpluses sharper to diminish their warping? for now, per Faa di Bruno's formula, we ignore the first-order component of the second-order chain rule in none of these parameters, the treatment for which is (experimentally) complete
        excitation_rate_gradient = tf.reshape(tf.math.reduce_sum(gradient[3], axis=0), reshape_spec) # reshape and linalg.diag are necessary here because rate matrices haven't been collapsed yet
        excitation_rate_cov = -raw_covariances[1] * tf.math.maximum(1.0/cls.numerical_limit,
            tf.square(tf.math.sigmoid(excitation_rate))[..., None, None]) \
          + logistic_deriv(excitation_rate)[..., None, None] * tf.linalg.diag(excitation_rate_gradient) + excitation_rate_prior_hess # doesn't matter if it's before summing, since the excitation_rate is a scalar.
        if optimize_excitations:
          assert not do_likelihood_expectation
          excitation_coef_cov = None
          background_cov = None
        else:
          excitation_coef_gradient = tf.math.reduce_sum(gradient[0], axis=0) # I know I'm repeating what I've written above but whatever
          excitation_coef_cov_logistic_second_deriv = tf.reshape(excitation_coef_gradient * logistic_second_deriv(excitation_coef), reshape_spec)
          excitation_coef_cov_deriv = tf.reshape(tf.math.sigmoid(excitation_coef), reshape_spec)
          excitation_coef_cov = -raw_covariances[0] * excitation_coef_cov_deriv[..., :, None] * excitation_coef_cov_deriv[..., None, :] \
            + tf.linalg.diag(excitation_coef_cov_logistic_second_deriv) + excitation_coef_prior_hess # these final terms are the first-order part of the chain rule formula
          background_gradient = tf.math.reduce_sum(gradient[4], axis=0)
          background_cov_logistic_second_deriv = background_gradient * logistic_second_deriv(background)
          background_cov_deriv = tf.reshape(tf.math.sigmoid(background), reshape_spec)
          background_cov = -raw_covariances[4] * background_cov_deriv[..., :, None] * background_cov_deriv[..., None, :] \
            + tf.linalg.diag(background_cov_logistic_second_deriv) + background_prior_hess
        def rectify(cov, tag): # cov may not be p.s.d due to the nonlinear projections, but it will always be Hermitian!
          # a,b=tf.linalg.eig(cov)
          # tf.math.real(tf.linalg.matmul(tf.linalg.matmul(b, tf.linalg.diag(tf.dtypes.complex(tf.math.abs(a), 0.0))), tf.linalg.inv(b)))
          #s, u, v = tf.linalg.svd(cov)
          #u tf.linalg.diag(s) tf.transpose(v)
          # above notes hint at possible ways to go about this.
          #with tf.device("/cpu:0"): # cuSolverDN has issues sometimes when performing self_adjoint_eig
          capped_hessian = tf.where( # NaN comparisons evaluate to false, so they always pass right through. Infs are caught and replaced with the highest limits we are willing to consider.
            cov < -cls.numerical_limit, -cls.numerical_limit,
            tf.where(cov > cls.numerical_limit, cls.numerical_limit, cov))
          safe_hessian = tf.debugging.check_numerics(capped_hessian, f"unrectified covariance---{tag}")
          eigvals, eigvecs = tf.linalg.eigh(safe_hessian) # at least these are real numbers. --> barring numerical inaccuracies, these are almost entirely positive and I might not have to do this after all. (not always the case actually) Only the lower-triangular part of this matrix is used...
          if verbose:
            tf.print(f"Hessian ({tag}) eigenvalues sans filtering:", tf.cast(eigvals, tf.float32), "\b.", output_stream=sys.stdout) # float64's are printed in a manner that showcases their precision. Not necessary.
          filtered_eigvals = -tf.math.maximum(tf.math.abs(eigvals), eigval_floor) # if the putative proper convariance must have non-negative eigenvalues, then the corresponding Hessian must have non-positive ones! Aha! In literature, this is all ameliorated by specifying that we look for the Hessian of the NEGATIVE log-likelihood. tf.where(eigvals > 0.0, eigvals, cls.zero) this could lead to null matrices, especially when they are of low rank
          # There should be a dedicated method to specifically transpose batched matrices..
          #if eigval_trunc > 0: # this doesn't seem to help at all...
          #  pad_shape = tf.concat([tf.shape(filtered_eigvals)[:-1], (tf.shape(filtered_eigvals)[-1] - eigval_trunc)[None]], axis=0)
          #  significant_eigvals = filtered_eigvals[..., :eigval_trunc] # Move along the most cogent subspace, like in a manifold. Without this, the resultant matrix would have full rank. The pseudoinverse dismisses zero eigenvalues, but the floor above establishes a reasonable floor for those that are counted.
          #  truncated_eigvals = tf.concat([significant_eigvals, tf.zeros(pad_shape, dtype=cls.dtype)], axis=-1)
          #else:
          #  truncated_eigvals = filtered_eigvals
          if parameters_are_bundled:
            transposed_eigvecs = tf.transpose(eigvecs, [0, 2, 1])
          else:
            transposed_eigvecs = tf.transpose(eigvecs)
          return tf.linalg.matmul(tf.linalg.matmul(eigvecs, tf.linalg.diag(filtered_eigvals)), transposed_eigvecs)
        hessians = [
          rectify(excitation_coef_cov, "ec") if not optimize_excitations else None,
          rectify(tf.math.reduce_sum(excitation_rate_cov, axis=[-2, -1], keepdims=True), "er") if not optimize_excitations else None, # try summing post-rectification
          rectify(suppression_coef_cov, "sc") if not learn_hawkes else
            tf.eye(tf.shape(suppression_coef_cov)[-1], batch_shape=([parameter_bundle_size] if parameters_are_bundled else []), dtype=cls.dtype), # oh! did I catch a source of learn_hawkes instability, and nans in the unrectified covariances?
          rectify(tf.math.reduce_sum(suppression_rate_cov, axis=[-2, -1], keepdims=True), "sr") if not learn_hawkes else  # negative axis indices in case parameters are batched
            tf.eye(1, batch_shape=([parameter_bundle_size] if parameters_are_bundled else []), dtype=cls.dtype),
          rectify(background_cov, "bc")] # careful with the signs.
        return derivatives, hessians
      else:
        return derivatives # get derivative wrt unpacked values

    softplus_inverse = lambda x: tf.where(x < 10.0, tf.math.log(tf.exp(tf.math.maximum(x, 1e-5)) - 1.0), x) # I've seen excitations go negative sometimes.. (not sure how that could be if they start positive, actually). Also if rate goes to zero, coefficients' gradient suffers and leads to NaNs.

    @tf.function
    def optimize(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background):
      new_params = pack_parameters(excitation_coef, excitation_rate, suppression_coef, suppression_rate, background)
      opt_excitation_coef = tf.zeros_like(excitation_coef)
      opt_background = tf.zeros_like(background)
      leftover_excitation = tf.zeros_like(excitation_coef)
      leftover_suppression = tf.zeros_like(new_params.suppression_coef)
      time_of_leftover = event_times[0]
      for t in tf.range(0, tf.size(event_times), batch_size): # use convenience method for all this?
        batch_types = event_types[t:(t+batch_size+1)]
        batch_times = event_times[t:(t+batch_size+1)]
        n_events_in_batch = tf.cast(len(batch_times) - 1, cls.dtype)
        p, b, next_leftover_excitation, next_leftover_suppression, next_time_of_leftover = cls.infer_causalities(new_params, batch_types, batch_times, leftover_excitation, leftover_suppression, time_of_leftover)
        opt_excitation_coef += cls.estimate_excitation_coefficient(new_params, p, batch_types, batch_times, event_times[-1], leftover_excitation, leftover_suppression, time_of_leftover) * n_events_in_batch
        opt_background += cls.estimate_background(new_params, b, batch_types[:-1], batch_times) * n_events_in_batch # kind of a hack to trim types but not times
        leftover_excitation, leftover_suppression, time_of_leftover = next_leftover_excitation, next_leftover_suppression, next_time_of_leftover
      opt_excitation_coef /= tf.cast(tf.size(event_times), cls.dtype)
      opt_background /= tf.cast(tf.size(event_times), cls.dtype)
      return [softplus_inverse(opt_excitation_coef), excitation_rate, suppression_coef, suppression_rate, softplus_inverse(opt_background)]

    learning_curve = tf.Variable(tf.zeros([n_iterations], cls.dtype))
    excitation_bandwidth_curve = tf.Variable(tf.zeros([n_iterations], cls.dtype)) if not optimize_excitations else None
    suppression_bandwidth_curve = tf.Variable(tf.zeros([n_iterations], cls.dtype))
    rate_bandwidth_curve = tf.Variable(tf.zeros([n_iterations], cls.dtype))

    @tf.function
    def report_progress(t, excitation_coef, excitation_rate, suppression_coef, suppression_rate, background, n_batch_particles=2): # components here are batched by the number of particles
      scores = tf.zeros([n_particles], dtype=cls.dtype)
      tf.assert_equal(n_particles % n_batch_particles, 0,
        message=f"n_particles must be divisible by the particle-batch size we're using, which is {n_batch_particles}")
      for i in tf.range(0, n_particles, n_batch_particles): # so that we don't overwhelm GPU memory
        j = i + n_batch_particles
        #batch_particles = tf.math.minimum(n_particles, j) - i # dynamic, but at least the rank will be resolved...
        params = pack_parameters(
          tf.ensure_shape(excitation_coef[i:j, ...], [n_batch_particles] + excitation_coef.shape[1:]),
          tf.ensure_shape(excitation_rate[i:j, ...], [n_batch_particles] + excitation_rate.shape[1:]),
          tf.ensure_shape(suppression_coef[i:j, ...], [n_batch_particles] + suppression_coef.shape[1:]),
          tf.ensure_shape(suppression_rate[i:j, ...], [n_batch_particles] + suppression_rate.shape[1:]),
          tf.ensure_shape(background[i:j, ...], [n_batch_particles] + background.shape[1:]))
        if do_likelihood_expectation:
          compute_logL = lambda p: \
            cls.compute_log_likelihood(params, p[0], p[1], batch_size=batch_size, excess_suppression_reg=excess_suppression_reg, weigh=weigh)
          log_likelihoods = tf.map_fn(compute_logL, (event_types, event_times), fn_output_signature=tf.TensorSpec([n_batch_particles], cls.dtype))
          score = tf.math.reduce_logsumexp(log_likelihoods, axis=0) - tf.math.log(tf.cast(n_records, cls.dtype))
        else:
          score = cls.compute_log_likelihood(params, event_types, event_times,
            batch_size=batch_size, excess_suppression_reg=excess_suppression_reg, weigh=weigh) # OR: intentionally without regularization
        scores = tf.tensor_scatter_nd_update(scores, tf.range(i, j)[:, None], score)
      if do_likelihood_expectation:
        record_lengths = tf.map_fn(lambda v: tf.cast(tf.size(v), cls.dtype), event_times, fn_output_signature=cls.dtype) # NOTE!!! fn_output_signature is the new name for dtype when that gets deprecated. the ISI servers still use TF 2.2 which aren't aware of the name change yet, so I have to stick to 'dtype' for now
        n_events = tf.math.reduce_mean(record_lengths)
      else:
        n_events = len(event_times)
      insample_posterior = (tf.math.reduce_logsumexp(scores) - tf.math.log(tf.cast(n_particles, cls.dtype))) / tf.cast(n_events, cls.dtype)
      with tf.control_dependencies([insample_posterior]):
        tf.print(colored(f"{reporting_tag} progress report incoming." if reporting_tag != "" else "Progress report incoming.",
          "red", attrs=["bold"]), output_stream=sys.stdout)
        tf.print(colored("Insample score:", "red"), insample_posterior, "\b.", output_stream=sys.stdout)
        interpret_avg = lambda x: tf.math.exp(x / (1.0 - tf.math.pow(rolling_memory_rate, tf.cast(n_rolling_average_updates, cls.dtype))))
        tf.print(colored("Kernel averages.", "yellow"),
          "excitation coef", interpret_avg(rolling_excit_coef_kernel), "excitation rate", interpret_avg(rolling_excit_rate_kernel),
          "suppression coef", interpret_avg(rolling_suppr_coef_kernel), "suppression rate", interpret_avg(rolling_suppr_rate_kernel),
          output_stream=sys.stdout) # by default, tensorflow prints everything to stderr!
      selection = tf.one_hot(t, n_iterations, dtype=cls.dtype)
      #nonlocal learning_curve # force capture. needed when this was not a tf.Variable. In TF 2.2 (which the GPU machines with 'conda' necessitate), this causes the static_analysis/reaching_definitions.py logic to bug out---but only when I added the above loop, somehow...
      learning_curve.assign_add(selection * insample_posterior)
      if not optimize_excitations:
        excitation_bandwidth_curve.assign_add(selection * excitation_kernel_bandwidth)
      suppression_bandwidth_curve.assign_add(selection * suppression_kernel_bandwidth)
      rate_bandwidth_curve.assign_add(selection * rate_kernel_bandwidth)

    logit_epsilon = tf.cast(1e-20, cls.dtype) # tested on tf.float32 and above
    logit = lambda x: tf.where(x > logit_epsilon,
      tf.where((1.0-x) > logit_epsilon,
        tf.math.log(tf.math.divide_no_nan(x, 1.0 - x)), tf.math.log(1.0/logit_epsilon - 1.0)),
      -tf.math.log(1.0/logit_epsilon - 1.0)) # in practice, the -1.0 probably doesn't make any difference
    assert not (cross_suppressions and permanent_suppression_rate is not None)
    if cross_suppressions:
      suppression_scale = params.suppression_rate[..., None, None]
    elif permanent_suppression_rate is None:
      suppression_scale = params.suppression_rate[..., None]
    else:
      suppression_scale = params.suppression_rate[..., None] * permanent_suppression_rate
    particles = [softplus_inverse(params.excitation_coef), softplus_inverse(params.excitation_rate),
      logit(params.suppression_coef * suppression_scale), # scaled logistic sigmoid for contrained gradient descent. the inverse is the logit function. divide by params.suppression_rate[..., None] to scale back to (constrained) suppression_coef space
      softplus_inverse(params.suppression_rate), softplus_inverse(params.background)]
    proj_samples, seconds_taken = stein.flow_particles(particles, learning_rate, evaluate_kernel, n_iterations,
      compute_log_likelihood_gradient=compute_log_likelihood_gradient, optimize=(optimize if optimize_excitations else None),
      do_newton=do_newton, set_bandwidths=(set_bandwidths if n_bandwidth_samples is not None else None),
      explore_bandwidths=(explore_bandwidths if n_bandwidth_samples is not None else None),
      particle_subsample_for_bandwidth_selection=subsample_for_bandwidths, bandwidth_iteration_interval=bandwidth_iteration_interval,
      parallel_bundle_size=parallel_bundle_size, report_progress=report_progress, kernel_verbosity_freq=(1.0 if verbose else kernel_verbosity_freq),
      tag=reporting_tag)
    samples = [tf.math.softplus(proj_samples[0]), tf.math.softplus(proj_samples[1]),
               tf.math.sigmoid(proj_samples[2]) / (
                 tf.math.softplus(proj_samples[3][:, None] if not cross_suppressions else proj_samples[3][:, None, None])),
               tf.math.softplus(proj_samples[3]), tf.math.softplus(proj_samples[4])]
    if learn_hawkes:
      samples[2] = tf.zeros_like(samples[2])
      samples[3] = tf.zeros_like(samples[3])
    if permanent_suppression_rate is not None:
      samples[2] = samples[2] / permanent_suppression_rate
      samples[3] = permanent_suppression_rate * tf.ones_like(samples[3])
    rolling_averages = [rolling_excit_coef_kernel, rolling_excit_rate_kernel, rolling_suppr_coef_kernel, rolling_suppr_rate_kernel, rolling_background_kernel]
    exp_rolling_averages = [tf.math.exp(avg).numpy() for avg in rolling_averages]
    bandwidth_curves = [(excitation_bandwidth_curve.numpy() if not optimize_excitations else None), suppression_bandwidth_curve.numpy(), rate_bandwidth_curve.numpy()]
    samples = [sample.numpy() for sample in samples]
    return samples, exp_rolling_averages, bandwidth_curves, learning_curve.numpy(), seconds_taken.numpy()


  @classmethod
  @tf.function
  def simulate_events_thinning(cls, params, n_events, n_last_events_to_use, time_epsilon=1e-8, time_bound=1e30, type_dtype=tf.int32): # taken from project "polynomial-hawkes" HOW THE SCOPE HAS CHANGED --- At some point, I should enable parameter batches herein. ALSO NO SUPPORT FOR LAGS. SEE THE METHOD BELOW.
    time = cls.zero
    event_types = tf.zeros([n_events], dtype=type_dtype)
    event_times = tf.fill([n_events], tf.cast(time_bound, cls.dtype)) # can't be inf because we have to be able to multiply by 0
    event_index = tf.zeros([], dtype=tf.int32) # signal that this is a tensor
    all_types = tf.range(tf.cast(params.n_dims, dtype=type_dtype))
    while event_index < n_events:
      first_event_to_use = tf.maximum(event_index-n_last_events_to_use, 0)
      last_event_to_use  = tf.maximum(event_index, n_last_events_to_use) # so that the window is always of the same size
      last_event_times = event_times[first_event_to_use:last_event_to_use] - time # NOTICE I FIX THE REFERENCE LOCALLY, to preserve stability
      last_event_types = event_types[first_event_to_use:last_event_to_use]
      next_time = tf.fill([params.n_dims], tf.cast(time_epsilon, dtype=cls.dtype)) # since 0 refers to our current point in time
      intensity = cls.infer_excitational_intensity(params, last_event_types, last_event_times, all_types, next_time) # we need this to always be an UPPER BOUND
      total_intensity = tf.math.reduce_sum(intensity, axis=0) # should be 1D. adding the 'axis=0' makes dimension mismatch more apparent
      quantile = tf.random.uniform([], dtype=cls.dtype)
      possible_jump = -tf.math.log(quantile) / total_intensity
      possible_spike = next_time + possible_jump
      later_intensity = cls.infer_intensity(params, last_event_types, last_event_times, all_types, possible_spike)
      cumulative_later_intensity = tf.math.cumsum(later_intensity, axis=0)
      probability = cumulative_later_intensity / total_intensity
      occurrence_time = time + possible_jump
      gamble = tf.random.uniform([], dtype=cls.dtype)
      if gamble <= probability[-1]:
        occurring_type = tf.where(gamble <= probability)[0] # note that argmax does not guarantee the index of the FIRST maximal value
        selection = tf.one_hot(event_index, n_events, dtype=cls.dtype)
        event_types += tf.cast(occurring_type, dtype=type_dtype) * tf.cast(selection, dtype=type_dtype)
        event_times *= 1.0 - selection # remove the present large number
        event_times += occurrence_time * selection
        event_index += 1
      time = occurrence_time # always advance. in next_time, small delta is to include current spike in next intensity. delta can't be too small that addition does nothing!
    return event_types, event_times

  @classmethod
  @tf.function
  def simulate_events_memoryless(cls, params, n_events, type_dtype=tf.int32, n_throwaway=0, warm_excitations=None, warm_suppressions=None): # note, this all operates from the ground up so if I ever add another fundamental feature I would have to amend this method
    tf.assert_equal(tf.rank(params.excitation_coef), 2, message="excitation coef has the wrong shape")
    tf.assert_equal(tf.rank(params.excitation_rate), 0, message="excitation rate has the wrong shape")
    tf.assert_equal(tf.rank(params.suppression_coef), 2, message="suppression coef has the wrong shape")
    tf.assert_equal(tf.rank(params.suppression_rate), 0, message="suppression rate has the wrong shape")
    tf.assert_equal(tf.rank(params.background), 1, message="background has the wrong shape")
    tf.assert_equal(tf.rank(params.lag), 0, message="lag has the wrong shape") # for simplicity, require lag and rates to be homegeneous. place these assertions to make sure the caller is aware and intentional with all this.
    if warm_excitations is not None:
      tf.assert_equal(tf.rank(warm_excitations), 2)
      tf.assert_equal(tf.rank(warm_suppressions), 2)
                                                                               # OVERENGINEERING abstractions by making room for features we don't need can be costly in many ways.
    n_total_events = n_events + n_throwaway
    event_types = tf.zeros([n_total_events], dtype=type_dtype)
    event_times = tf.zeros([n_total_events], cls.dtype)
    event_index = tf.zeros([], dtype=tf.int32) # signal that this is a tensor
    all_types = tf.range(tf.cast(params.n_dims, dtype=type_dtype))
    current_time = tf.zeros([], dtype=cls.dtype)             # if we wish to extend and realize a forecast after a batch, we have this functionality to recycle the batch leftovers instead of being subjected to a cold start
    excitations = tf.zeros([params.n_dims], dtype=cls.dtype) if warm_excitations is None else tf.math.reduce_sum(warm_excitations, axis=-2) # since rates are the same, we could consolidate these excitations into one accumulator that is later scaled by the relevant coefficients. Actually, no, since they change based on the target type.
    suppressions = tf.zeros([params.n_dims], dtype=cls.dtype) if warm_suppressions is None else tf.math.reduce_sum(warm_suppressions, axis=-2) # efficiently accumulate to the latest timestamp via the memoryless property
    get_intensities = lambda exc, sup: exc * tf.math.maximum(cls.unit - sup, cls.zero) + params.background
    while event_index < n_total_events: # assume, in the presence of a lag, that there is no chance an event will occur by sheer luck in between an occurrence and a lag. NO LONGER: to accommodate for the unnecessary possibility that lags may vary, we have multiple time pointers now advancing in parallel.
      current_intensities = get_intensities(excitations, suppressions)
      total_intensity = tf.math.maximum(tf.math.reduce_sum(current_intensities, axis=-1), cls.intensity_floor) # scalar result
      jump_quantile = tf.random.uniform([], dtype=cls.dtype)
      jump = -tf.math.log(jump_quantile) / total_intensity
      jump_excitations = excitations * tf.math.exp(-params.excitation_rate * jump)
      jump_suppressions = suppressions * tf.math.exp(-params.suppression_rate * jump)
      jump_intensities = get_intensities(jump_excitations, jump_suppressions)
      cumul_jump_probabilities = tf.math.cumsum(jump_intensities, axis=-1) / total_intensity
      gamble = tf.random.uniform([], dtype=cls.dtype)
      if gamble <= tf.debugging.check_numerics(cumul_jump_probabilities[-1], "thinning intensity ratio"):
        occurring_type = tf.where(gamble <= cumul_jump_probabilities)[0][0] # taken from above. first index to be reached. second [0] is to ensure it's a scalar
        selection = tf.one_hot(event_index, n_total_events, dtype=cls.dtype)
        event_types += tf.cast(occurring_type, dtype=type_dtype) * tf.cast(selection, dtype=type_dtype)
        event_times += (current_time + jump) * selection
        event_index += 1
        jump += params.lag
        jump_excitations *= tf.math.exp(-params.excitation_rate * params.lag) # with lag this time
        jump_suppressions *= tf.math.exp(-params.suppression_rate * params.lag)
        exciting_effect = tf.gather(params.excitation_coef, occurring_type, axis=0) * params.excitation_rate # gathering by a scalar index is a way to index the row by a Tensor
        suppressing_effect = tf.gather(params.suppression_coef, occurring_type, axis=0) * params.suppression_rate
        jump_excitations += exciting_effect
        jump_suppressions += suppressing_effect
      current_time += jump
      excitations = tf.ensure_shape(jump_excitations, excitations.shape)
      suppressions = tf.ensure_shape(jump_suppressions, suppressions.shape)
    return event_types[n_throwaway:], event_times[n_throwaway:]

  @classmethod
  def serialize(cls, params):
    return {
      "excitation_coef": params.excitation_coef.numpy(), "excitation_rate": params.excitation_rate.numpy(),
      "suppression_coef": params.suppression_coef.numpy(), "suppression_rate": params.suppression_rate.numpy(),
      "background": params.background.numpy(), "lag": params.lag.numpy()}

  @classmethod
  def deserialize(cls, params):
    return cls.Parameters(params["excitation_coef"], params["excitation_rate"],
      params["suppression_coef"], params["suppression_rate"], params["background"], params["lag"])

@tf.function
def sample_beta_variate(alpha, beta, shape, dtype):
  x = tf.random.gamma(shape, alpha, 1.0, dtype=dtype)
  y = tf.random.gamma(shape, beta,  1.0, dtype=dtype)
  return tf.math.divide_no_nan(x, x + y)
