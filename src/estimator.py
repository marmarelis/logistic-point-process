import numpy as np
import pandas as pd
import tensorflow as tf
import types
from tqdm import tqdm
import pprint # pretty printing
from matplotlib import pyplot as plt
plt.ion()

class UnivariateProcess(tf.Module):
  def __init__(self, excitation_coef, suppression_coef, excitation_rate, suppression_rate, background):
    self.dtype = tf.float64
    self.zero = tf.zeros([], dtype=self.dtype)
    self.unit = tf.ones([], dtype=self.dtype)
    self.excitation_coef = tf.Variable(excitation_coef, dtype=self.dtype)
    self.suppression_coef = tf.Variable(suppression_coef, dtype=self.dtype)
    self.excitation_rate = tf.Variable(excitation_rate, dtype=self.dtype)
    self.suppression_rate = tf.Variable(suppression_rate, dtype=self.dtype)
    self.background = tf.Variable(background, dtype=self.dtype)

  @tf.function
  def compute_log_likelihood(self, event_times):
    intensity = self.infer_intensity(event_times, event_times)
    score = tf.math.reduce_sum(tf.math.log(intensity), axis=0)
    penalty = self.integrate_intensity(event_times)
    return score - penalty

  @tf.function
  def infer_causalities(self, event_times):
    intensity = self.infer_intensity(event_times, event_times)
    excitation = self.infer_excitation(event_times, event_times)
    suppression = self.infer_suppression(event_times, event_times)
    total_excitation = tf.math.reduce_sum(excitation, axis=0)
    total_suppression = tf.math.reduce_sum(suppression, axis=0)
    total_unsuppressed = tf.math.minimum(self.unit, tf.math.maximum(self.zero, (1.0 - total_suppression)))
    #p = excitation * total_unsuppressed[None, :] / intensity[None, :]
    #q = suppression * total_excitation[None, :] / intensity[None, :]
    #b = self.background / intensity
    #q_sum = b #tf.math.reduce_sum(p, axis=0) + b - 1.0
    #corrected_q = q * tf.math.divide_no_nan(q_sum, tf.math.reduce_sum(q, axis=0))[None, :]
    p = excitation * total_unsuppressed / intensity
    #q = suppression * total_excitation / intensity # let's get this straight. High Q -> suppression at actual event occurrence -> PUNISH, as a counterweight to the savings in the intensity integral. If estimated suppression coefficient is too high, then not enough punishing happened. Tiny Q's: goes negative. Giant Q's: grows larger.
    #corrected_q = tf.math.divide_no_nan(q, tf.maximum(unit, tf.math.reduce_sum(q, axis=0)))
    b = self.background / intensity
    return p, b

  @tf.function
  def filter_deltas(self, event_times, times, filter=np.inf):
    deltas = times[None, :] - event_times[:, None]
    filtered_deltas = tf.where(deltas > 0.0, deltas, filter)
    return filtered_deltas

  @tf.function
  def infer_excitation(self, event_times, times):
    filtered_deltas = self.filter_deltas(event_times, times)
    excitations = self.excitation_coef * self.excitation_rate * tf.exp(-self.excitation_rate * filtered_deltas)
    return excitations

  @tf.function
  def infer_suppression(self, event_times, times):
    filtered_deltas = self.filter_deltas(event_times, times)
    suppressions = self.suppression_coef * self.suppression_rate * tf.exp(-self.suppression_rate * filtered_deltas)
    return suppressions

  @tf.function
  def infer_response(self, event_times, times):
    excitation = tf.math.reduce_sum(self.infer_excitation(event_times, times), axis=0)
    suppression = tf.math.reduce_sum(self.infer_suppression(event_times, times), axis=0)
    return excitation * tf.math.maximum(self.zero, (1.0 - suppression))

  @tf.function
  def infer_intensity(self, event_times, times):
    return self.infer_response(event_times, times) + self.background

  @tf.function
  def infer_excitational_intensity(self, event_times, times):
    return tf.math.reduce_sum(self.infer_excitation(event_times, times), axis=0) + self.background

  @tf.function
  def integrate_intensity(self, event_times): # for now, this DOES make the approximation that holds as long as the record is large and the process is stable (i.e., relatively few points near T)
    deltas = self.filter_deltas(event_times, event_times) # positive side is upper-triangular
    residual_times = event_times[-1] - event_times
    time_interval = event_times[-1] - event_times[0]
    combined_rate = self.excitation_rate + self.suppression_rate
    summand = (tf.math.exp(-self.suppression_rate * deltas) + tf.math.exp(-self.excitation_rate * deltas)) * (1.0 - tf.exp(-combined_rate*residual_times))[None, :]
    summation = tf.math.reduce_sum(summand, axis=1) # keep one axis in the end
    scale = self.suppression_coef * self.excitation_coef * (self.suppression_rate * self.excitation_rate) / (self.suppression_rate + self.excitation_rate)
    correction = scale * ((1.0 - tf.exp(-combined_rate*residual_times)) + summation)
    core = self.excitation_coef * (1.0 - tf.exp(-self.excitation_rate * residual_times))
    main_integral = tf.math.reduce_sum(core - correction, axis=0) # this is the main contribution
    integral = tf.math.maximum(self.zero, main_integral) + time_interval * self.background # hard to account for an overwhelming suppression in the integral. Make sure, with the proper bias, that this doesn't hit the bottom too often.
    return integral

  @tf.function
  def estimate_excitation_coefficient(self, p, event_times):
    time_residue = event_times[-1] - event_times
    tf.debugging.assert_greater_equal(time_residue, self.zero)
    denominator = self.integrate_intensity(event_times) / self.excitation_coef # LINEAR DERIVATIVE. (1.0 - tf.math.exp(-self.excitation_rate * time_residue)) leads to an underestimated coefficient
    return tf.math.reduce_sum(p, axis=(0, 1)) / denominator

  @tf.function
  def estimate_excitation_rate(self, p, event_times):
    deltas = self.filter_deltas(event_times, event_times, filter=self.zero)
    return tf.math.reduce_sum(p, axis=(0, 1)) / tf.math.reduce_sum(p * deltas, axis=(0, 1))

  @tf.function
  def sample_excitation_rate(self, event_times, alpha, beta, n_samples, n_moments):
    prior = tf.random.gamma([n_samples], alpha, beta, dtype=self.dtype)
    return self.estimate_posterior_moments(event_times, prior, lambda model: model.excitation_rate, n_moments)

  @tf.function
  def estimate_suppression_coefficient(self, p, event_times, prior_alpha, prior_beta, epsilon=1e-7, magic_number=0.0): # THIS IS FRAMED AS A PROBLEM (\sum_ij q_ij/(x + w_ij) = C, with all w_ij, C < 0)
    total_p = tf.math.reduce_sum(p, axis=0)
    deltas = self.filter_deltas(event_times, event_times)
    inv_shifts = -self.suppression_rate * tf.exp(-self.suppression_rate * deltas)
    shifts = 1.0 / tf.math.reduce_sum(inv_shifts, axis=0) # shifts ARE zero when time is inf. we don't consider those as contributing to the equation. 1/inf -> 0
    constant_scale = -self.excitation_coef * (self.excitation_rate*self.suppression_rate/(self.excitation_rate + self.suppression_rate))
    combined_rate = self.excitation_rate + self.suppression_rate
    residual_times = event_times[-1] - event_times
    constant_content = (tf.exp(-self.excitation_rate * deltas) + tf.exp(-self.suppression_rate * deltas)) * (1.0 - tf.exp(-combined_rate*residual_times))[None, :]
    constant = constant_scale * tf.math.reduce_sum(tf.math.reduce_sum(constant_content, axis=0) + (1.0 - tf.exp(-combined_rate*residual_times)), axis=0) # precise axes don't matter here, just to count the 1.0 N times
    suppression_coef = solve_fractional_equation(total_p, shifts, constant, epsilon, prior_alpha, prior_beta, magic_number=magic_number)
    return suppression_coef

  @tf.function
  def sample_suppression_coefficient(self, event_times, alpha, beta, n_samples, n_moments):
    prior = sample_beta_variate(alpha, beta, n_samples, self.dtype)
    return self.estimate_posterior_moments(event_times, prior, lambda model: model.suppression_coef, n_moments)

  @tf.function
  def estimate_suppression_rate(self, p, event_times, current_guess, epsilon=1e-7):
    total_p = tf.math.reduce_sum(p, axis=0)
    deltas = self.filter_deltas(event_times, event_times)
    contributions = tf.exp(-current_guess[:, None, None] * deltas) # FOR GRID SEARCH
    scales = 1.0 - self.suppression_coef * current_guess[:, None] * tf.math.reduce_sum(contributions, axis=1)
    filtered_scales = tf.maximum(tf.cast(epsilon, self.dtype), scales)
    numerator = total_p * tf.math.reduce_sum(contributions, axis=1) / filtered_scales
    filtered_deltas = tf.where(tf.math.is_inf(deltas), self.zero, deltas)
    denominator = total_p * tf.math.reduce_sum(contributions * filtered_deltas, axis=1) / filtered_scales
    return tf.math.reduce_sum(numerator, axis=1) / tf.math.reduce_sum(denominator, axis=1)

  @tf.function
  def sample_suppression_rate(self, event_times, alpha, beta, n_samples, n_moments): # Gamma prior. Redundant with the below. Streamline at some point.
    prior = tf.random.gamma([n_samples], alpha, beta, dtype=self.dtype)
    return self.estimate_posterior_moments(event_times, prior, lambda model: model.suppression_rate, n_moments)
    #old_suppression_rate = tf.identity(self.suppression_rate) # effectively copy it, to break off update propagation
    #def eval_log_likelihood(suppression_rate):
    #  self.suppression_rate.assign(suppression_rate)
    #  return self.compute_log_likelihood(event_times)
    #log_likelihoods = tf.map_fn(eval_log_likelihood, prior, parallel_iterations=1) # note: assignments, which always synchronize in TensorFlow, would interfere with each other in parallel. This is why it's good to decouple data from code!
    #self.suppression_rate.assign(old_suppression_rate)
    #log_rates = tf.math.log(prior)
    #moments = tf.range(1, n_moments+1, dtype=self.dtype)
    #log_numerator = tf.math.reduce_logsumexp(moments[:, None] * log_rates + log_likelihoods, axis=1)
    #log_denominator = tf.math.reduce_logsumexp(log_likelihoods, axis=0)
    #return tf.math.exp(log_numerator - log_denominator)

  @tf.function
  def estimate_posterior_moments(self, event_times, prior_samples, get_param, n_moments): # It would be a hassle to add extra dimensions to the model's variables, so it performs a serial computation now
    old_param = tf.identity(get_param(self)) # effectively copy it, to break off update propagation
    def eval_log_likelihood(parameter):
      get_param(self).assign(parameter)
      return self.compute_log_likelihood(event_times)
    log_likelihoods = tf.map_fn(eval_log_likelihood, prior_samples, parallel_iterations=1) # note: assignments, which always synchronize in TensorFlow, would interfere with each other in parallel. This is why it's good to decouple data from code!
    get_param(self).assign(old_param)
    log_samples = tf.math.log(prior_samples)
    moments = tf.range(1, n_moments+1, dtype=self.dtype)
    log_numerator = tf.math.reduce_logsumexp(moments[:, None] * log_samples + log_likelihoods, axis=1)
    log_denominator = tf.math.reduce_logsumexp(log_likelihoods, axis=0)
    return tf.math.exp(log_numerator - log_denominator)

  @tf.function
  def simulate_events_thinning(self, n_events, n_last_events_to_use, time_epsilon=1e-8, time_bound=1e30): # taken from project "polynomial-hawkes" HOW THE SCOPE HAS CHANGED
    time = self.zero
    event_times = tf.fill([n_events], tf.cast(time_bound, self.dtype)) # can't be inf because we have to be able to multiply by 0
    event_index = tf.zeros([], dtype=tf.int32) # signal that this is a tensor
    while event_index < n_events:
      first_event_to_use = tf.maximum(event_index-n_last_events_to_use, 0)
      last_event_to_use  = tf.maximum(event_index, n_last_events_to_use) # so that the window is always of the same size
      last_event_times = event_times[first_event_to_use:last_event_to_use] - time # NOTICE I FIX THE REFERENCE LOCALLY, to preserve stability
      next_time = tf.convert_to_tensor([time_epsilon], dtype=self.dtype) # since 0 refers to our current point in time
      intensity = self.infer_excitational_intensity(last_event_times, next_time)[0] # we need this to always be an UPPER BOUND
      quantile = tf.random.uniform([], dtype=self.dtype)
      possible_jump = -tf.math.log(quantile) / intensity
      possible_spike = next_time + possible_jump
      later_intensity = self.infer_intensity(last_event_times, possible_spike)[0]
      probability = later_intensity / intensity
      occurrence_time = time + possible_jump
      gamble = tf.random.uniform([], dtype=self.dtype)
      if gamble <= probability:
        selection = tf.one_hot(event_index, n_events, dtype=self.dtype)
        event_times *= 1.0 - selection # remove the present large number
        event_times += occurrence_time * selection
        event_index += 1
      time = occurrence_time # always advance. in next_time, small delta is to include current spike in next intensity. delta can't be too small that addition does nothing!
    return event_times

  def serialize(self):
    return types.SimpleNamespace(
      excitation_coef=self.excitation_coef.numpy(), excitation_rate=self.excitation_rate.numpy(),
      suppression_coef=self.suppression_coef.numpy(), suppression_rate=self.suppresison_rate.numpy(),
      background=self.background.numpy())

  def deserialize(self, params):
    self.excitation_coef.assign(params.excitation_coef); self.excitation_rate.assign(params.excitation_rate)
    self.suppression_coef.assign(params.suppression_coef); self.suppression_rate.assign(params.suppression_rate)
    self.background.assign(params.background)

@tf.function
def sample_beta_variate(alpha, beta, n_samples, dtype):
  x = tf.random.gamma([n_samples], alpha, 1.0, dtype=dtype)
  y = tf.random.gamma([n_samples], beta,  1.0, dtype=dtype)
  return tf.math.divide_no_nan(x, x + y)

@tf.function
def get_gamma_prior_strength(alpha, beta):
  entropy = alpha - tf.math.log(beta) + tf.math.lgamma(alpha) + (1.0-alpha)*tf.math.digamma(alpha) # looks a little similar to below..
  mean = alpha/beta
  maximal_entropy = 1.0 + tf.math.log(mean) # follows from exponential distribution
  return maximal_entropy - entropy

@tf.function
def get_beta_prior_strength(alpha, beta):
  entropy = tf.math.lbeta(tf.convert_to_tensor([alpha, beta])) - (alpha-1.0)*tf.math.digamma(alpha) - (beta-1.0)*tf.math.digamma(beta) + (alpha+beta-2.0)*tf.math.digamma(alpha+beta)
  maximal_entropy = 0.0 # follows from the uniform distribution. is differential entropy always non-positive?
  return maximal_entropy - entropy

@tf.function
def solve_fractional_equation(weights, shifts, constant, epsilon, prior_alpha, prior_beta, verbose=False, magic_number=1e-7): # pertains to the estimation of the suppression coefficient
  def get_position(guess):
    return tf.math.reduce_sum(tf.math.divide_no_nan(weights, (shifts + guess))) + (prior_alpha - 1.0)/tf.math.abs(guess) - (prior_beta - 1.0)/tf.math.abs(1.0 - guess) # Beta distribution as prior. absolute value is to throw out-of-bounds back in the right direction
  def get_velocity(guess):
    return -tf.math.reduce_sum(tf.math.divide_no_nan(weights, tf.square(shifts + guess))) - (prior_alpha - 1.0)/tf.square(guess) - (prior_beta - 1.0)/tf.square(1.0 - guess)
  filtered_shifts = tf.where(shifts < 0.0, shifts, -np.inf) # for the line below solely
  upper_bound = tf.math.reduce_min(-filtered_shifts)
  current_guess = upper_bound * (1.0 - magic_number) # weights are negative! go slightly lower, by some magic number. start SUPER close to prevent verging onto the other side
  current_position = get_position(current_guess)     # Now if we set our initial guess EQUAL to the upper bound, it won't crash because we're using divide_no_nan. It will treat it as if we removed that one singular component.
  init_attempt = 0
  while current_position > constant:
    if init_attempt == 10: # Houston, we've got a problem.
      tf.print("Could not find a starting position for Newton's method.")
      break
    current_guess = tf.math.sqrt(current_guess * upper_bound) # geometric mean is more numerically stable
    current_position = get_position(current_guess)
    init_attempt += 1
  #tf.assert_less(current_position, constant) # upper limit (negative). # TRY IT ANYWAYS. Got nothing to lose at this point... make sure we're not too close to the upper_bound singularity.
  while tf.math.abs((current_position - constant) / constant) > epsilon:
    current_velocity = get_velocity(current_guess)
    new_movement = (current_position - constant) / current_velocity
    current_guess -= new_movement
    current_position = get_position(current_guess)
    if verbose: tf.print(current_position - constant, current_guess)
  return current_guess

def perform_experiment(ground_truth, initialization, n_events, n_last_events_to_use, n_samples, n_epochs, gamma_alpha, gamma_beta, beta_alpha, beta_beta, bayesian_coef, min_suppression_coef=1e-20):
  print("Beta strength: {}; gamma strength: {}.".format(get_beta_prior_strength(beta_alpha, beta_beta), get_gamma_prior_strength(gamma_alpha, gamma_beta)))
  event_times = ground_truth.simulate_events_thinning(n_events, n_last_events_to_use)
  model = initialization
  for epoch in tqdm(range(n_epochs)):
    p, b = model.infer_causalities(event_times)
    excitation_coef = model.estimate_excitation_coefficient(p, event_times)
    excitation_rate = model.estimate_excitation_rate(p, event_times)
    if bayesian_coef:
      suppression_coef = model.sample_suppression_coefficient(event_times, beta_alpha, beta_beta, n_samples, 1)[0]
    else:
      suppression_coef = tf.maximum(tf.cast(min_suppression_coef, model.dtype), model.estimate_suppression_coefficient(p, event_times, beta_alpha, beta_beta))
    suppression_rate = model.sample_suppression_rate(event_times, gamma_alpha, gamma_beta, n_samples, 1)[0]
    parameters = types.SimpleNamespace(
      excitation_coef=excitation_coef.numpy(), excitation_rate=excitation_rate.numpy(), suppression_coef=suppression_coef.numpy(), suppression_rate=suppression_rate.numpy()
    )
    tqdm.write(pprint.pformat(parameters))
    model.deserialize(parameters)
  return model
