import numpy as np
import pandas as pd
import tensorflow as tf # REQUIRES TF 2.2
#import tensorflow_probability as tfp # for tfp.stats.quantiles
import sys
from types import SimpleNamespace
from tqdm import tqdm
from termcolor import colored # colorful printing

# Stein variational gradient descent. It would be pleasant to sample from the reference distribution deterministically for reproducibility's sake.
# It doesn't really matter what parametric form it takes in the beginning, as long as it's reasonably spread out. For instance, how about an
# evenly-spaced spaced sample (i.e. tf.linspace) on the Gaussian inverse cumulative distribution, also known as the quantile function? Or even uniform.

# Operate on a list of differently shaped variables, and take gradients with respect to all of them. The Hessian would be more complicated this way.

@tf.function
def sample_deterministic_normal(mean, sigma, resolution): # sadly, it's not clear what we must to in the multivariate setting without combinatorial samples
  dtype = tf.convert_to_tensor(mean).dtype # if it's not already a tensor
  quantile = tf.linspace(tf.cast(0.0, dtype), tf.cast(1.0, dtype), resolution + 2)[1:-1] # stop short of exact 0 and 1
  return sigma*tf.math.sqrt(2.0) * tf.math.erfinv(2.0*quantile - 1.0) + mean

def flow_particles(positions, rate, evaluate_kernel, n_iterations,
    compute_log_likelihood=None, compute_log_likelihood_gradient=None, optimize=None, do_newton=False, parallel_bundle_size=1,
    set_bandwidths=None, explore_bandwidths=None, particle_subsample_for_bandwidth_selection=None, bandwidth_iteration_interval=1,
    report_progress=None, kernel_verbosity_freq=0.0, tag="these"): # positions is a list of parameters batched along the first dimension. use the 'positions' format here to speed up graph contruction at large scale
  dtype = positions[0].dtype
  particles = [tf.Variable(parameters) for parameters in positions]
  n_particles = particles[0].shape[0]
  n_components = len(particles) # does Stein only make sense with a large number of particles, and doubly so when estimating the gradient's covariance? If so, the present construction and the assumption that a particle's likelihood should be evaluated one at a time do not make sense.
  #particles = [[tf.Variable(p) for p in particle] for particle in zip(*[tf.unstack(position, axis=0) for position in positions])] # transpose the list of lists for easier handling. now it's a list of batches, which are lists of parameter variables

  @tf.function
  def evaluate_heat_equation_bandwidth_objective(particles, bandwidths, particle_subsample): # see "Understanding and Accelerating Particle-Based Variational Inference" (2019)
    def smooth_density(particles): # all gradients are welcome in here
      def eval_kernels(particle): # unbiased because mean is taken over number of particles? is this a stochastic Laplace estimator? is it risky?
        kernels = tf.convert_to_tensor([
          evaluate_kernel(particle, other_particle, bandwidths) for other_particle in particles]) # n_particles x (n_bandwidth_samples)
        subsampled_kernels = tf.gather(kernels, tf.random.shuffle(tf.range(n_particles))[:particle_subsample], axis=0) # workaround tf.random.shuffle not supporting gradients. I hope that, as gradients pass through, tensorflow can avoid doing the exhaustive computation for all particles before trimming down. I'm not very sure, given some weird gradient behaviors with tf.where in the past
        return subsampled_kernels
      def statically_eval_kernels(particle):
        indices = list(range(n_particles))
        np.random.shuffle(indices)
        subsampled_particles = [particles[i] for i in indices[:particle_subsample]]
        kernels = tf.convert_to_tensor([
          evaluate_kernel(particle, other_particle, bandwidths) for other_particle in subsampled_particles]) # n_particles x (n_bandwidth_samples)
        return kernels # unlike the above, this method STATICALLY sets a subsample of the particles on which to compute the kernel each time it is called in the following code. no room for tensorflow blunders here.
      return tf.stack([
          tf.math.reduce_mean(statically_eval_kernels(particle), axis=0) for particle in particles
        ], axis=1) # (n_bandwidth_samples) x n_particles
    particle_jacobians = []
    with tf.GradientTape(persistent=True) as first_tape:
      first_tape.watch(particles)
      with tf.GradientTape() as second_tape:
        second_tape.watch(particles)
        density = smooth_density(particles)
      jacobians = second_tape.jacobian(density, particles, unconnected_gradients=tf.UnconnectedGradients.NONE) # a double list by particles, then particle components: (n_bandwidth_samples) x n_particles x component... SHOULD THERE BE UNCONNECTED GRADIENTS?
      for i, particle_jacobian in enumerate(jacobians):
        component_jacobians = []
        for j, component_jacobian in enumerate(particle_jacobian):
          if component_jacobian is not None: # some components (ie the Hawkes background) may not have any effect. Exclude them.
            component_jacobians.append(component_jacobian[:, i, ...])
        particle_jacobians.append(component_jacobians) # must create these INSIDE the first_tape context
    particle_laplacians = []
    for i, particle_jacobian in enumerate(particle_jacobians):
      particle_laplacian = tf.zeros([tf.shape(density)[0]], dtype=density.dtype) # n_bandwidth_samples
      for j, component_jacobian in enumerate(particle_jacobian):
        second_deriv = first_tape.jacobian(component_jacobian, particles[i][j], unconnected_gradients=tf.UnconnectedGradients.NONE, experimental_use_pfor=False) # I hope this slicing prevents the second Jacobian from computing across all the dimensions needlessly. My growing suspicion is that it does. HERE n_bandwidth_samples becomes necessary because [component] can be multiple dimensions. n_bandwidth_samples x [component...] x [component...]. the pfor thing is for reduced memory usage
        component_laplacian = tf.stack([ # LOOK INTO tfp.math.diag_jacobian
            tf.math.reduce_sum(tf.linalg.tensor_diag_part(band_second_deriv if tf.rank(band_second_deriv) > 0 else band_second_deriv[None, None])) # weird hack wherein the INPUT must be valid, since tensorflow checks both branches always
          for band_second_deriv in tf.unstack(second_deriv, axis=0)],
          axis=0) # (n_bandwidth_samples), Laplace operator at particles[i]. Double check because this is complex logic, and a little up in the air.
        particle_laplacian += component_laplacian
      particle_laplacians.append(particle_laplacian)
    laplacian = tf.stack(particle_laplacians, axis=1) # (n_bandwidth_samples) x n_particles
    dot_product = tf.zeros_like(laplacian) # both this and the above are batched by the overall sum_k objective(x_k)
    for j, particle_jacobian in enumerate(jacobians): # second term in the heat-equation objective. j corresponds to the j in the aforementioned paper
      for component_jacobian in particle_jacobian:
        if component_jacobian is None:
          continue
        component_rank = len(component_jacobian.shape) - 2
        expanded_density = density
        for _ in range(component_rank):
          expanded_density = tf.expand_dims(expanded_density, axis=-1) # add a component_rank number of Nones to the end
        jacobian_of_log = component_jacobian[:, j, ...] / expanded_density[:, j, ...] # derivative of log is the normal derivative divided by the value. n_bandwidth_samples x component...
        component_dot = tf.math.reduce_sum(component_jacobian * jacobian_of_log[:, None, ...], axis=[r+2 for r in range(component_rank)]) # the dot product, batched along n_particles (the k sum) of which the jacobian_of_log is invariant
        dot_product += component_dot
    objective = tf.math.reduce_sum(tf.square(laplacian + dot_product), axis=-1)
    return objective

  @tf.function
  def select_best_bandwidth():
    if explore_bandwidths is None: return
    trans_particles = [[p[particle, ...] for p in particles] for particle in range(n_particles)]
    possible_bandwidths = explore_bandwidths()
    #with tf.device("/cpu:1"): # second CPU. TF disallows more flexible assignments like "use any available CPU." Also, there are 'jobs' and 'tasks' that can be assigned in this command that group operations together.
    scores = evaluate_heat_equation_bandwidth_objective(trans_particles, possible_bandwidths, particle_subsample_for_bandwidth_selection)
    best_score = tf.math.argmin(scores)
    set_bandwidths([bandwidths[best_score] for bandwidths in possible_bandwidths]) # bandwidths have many components

  @tf.function
  def get_log_likelihood_gradient(particle):
    if compute_log_likelihood is not None:
      with tf.GradientTape() as tape:
        tape.watch(particle)
        log_likelihood = compute_log_likelihood(*particle)
      return tape.gradient(log_likelihood, particle)
    else:
      return compute_log_likelihood_gradient(*particle)

  @tf.function
  def get_log_likelihood_gradient_and_hessian(particle):
    if compute_log_likelihood is not None:
      log_likelihood = compute_log_likelihood(*particle)
      gradient = tf.gradients(log_likelihood, particle)
      hessian = tf.hessians(log_likelihood, particle)
      flat_hessian = [
        tf.reshape(p_hessian, [tf.size(p_gradient), tf.size(p_gradient)])
        for p_gradient, p_hessian in zip(gradient, hessian)
      ]
      return gradient, flat_hessian
    else:
      return compute_log_likelihood_gradient(*particle)

  def clear_null(x, shape_like):
    return x if x is not None else tf.zeros_like(shape_like)

  @tf.function
  def estimate_hessians_from_covariance(gradients): # as the empirical covariance of the gradient. correct for nonzero mean because we're not necessarily integrating over the whole posterior.
    hessians = []
    for component in range(n_components):
      dimension = tf.size(gradients[0][component])
      product = tf.zeros([dimension, dimension], dtype)
      mean = tf.zeros([dimension], dtype)
      for gradient in gradients:
        flat_gradient = tf.reshape(gradient[component], [-1])
        product += tf.tensordot(flat_gradient, flat_gradient, axes=0) # no axes implies outer product
        mean += flat_gradient
      product /= n_particles
      mean /= n_particles
      covariance = product - tf.tensordot(mean, mean, axes=0)
      hessians.append(-covariance) # without the mean correction, we'd be performing the Gauss-Newton approximation of the Hessian. Neal, however, argues for the usage of the empirical covariance.
    return hessians

  @tf.function
  def estimate_kernelized_hessians_from_covariance(particles, gradients):
    all_hessians = [] # per particle
    for particle in particles:
      hessians = []
      for component in range(n_components):
        dimension = tf.size(particles[0][component])
        if gradients[0][component] is None:
          hessians.append(tf.zeros([dimension, dimension], dtype))
          continue
        product = tf.zeros([dimension, dimension], dtype)
        mean = tf.zeros([dimension], dtype)
        kernel_product = tf.zeros([dimension], dtype)
        for gradient, reference_particle in zip(gradients, particles):
          flat_gradient = tf.reshape(gradient[component], [-1])
          with tf.GradientTape() as tape: # some of this is redundant with the main loop. Hopefully it doesn't occupy the brunt of the computing time.
            tape.watch(reference_particle)
            kernel = evaluate_kernel(reference_particle, [tf.stop_gradient(p) for p in particle])
          kernel_gradient = tape.gradient(kernel, reference_particle[component])
          if kernel_gradient[0] is not None:
            flat_kernel_gradient = tf.reshape(kernel_gradient, [-1])
          else:
            flat_kernel_gradient = tf.zeros_like(flat_gradient)
          product += tf.tensordot(flat_gradient, flat_gradient, axes=0) * tf.square(kernel)
          mean += flat_gradient * kernel
          kernel_product += tf.tensordot(flat_kernel_gradient, flat_kernel_gradient, axes=0)
        product /= n_particles
        mean /= n_particles
        kernel_product /= n_particles
        covariance = product - tf.tensordot(mean, mean, axes=0) + kernel_product #TRY UNCENTERED COVARIANCE FOR LOWER VARIANCE, HIGHER BIAS OF ESTIMATE
        hessians.append(-covariance)
      all_hessians.append(hessians)
    return all_hessians # will employ the block-diagonal approximation as suggested in the original "Stein variational Newton" paper. will additionally separate by parameter "components" here. h_ij is a kernelized version of all this, the correct form in the present context, which the Bayesian changepoint Hawkes paper omits.

  @tf.function
  def estimate_kernelized_hessians(particles, hessians):
    particle_pack = [tf.stack([op[c] for op in particles], axis=0) for c in range(n_components)]
    inflated_particle_pack = [pc[:, None, ...] for pc in particle_pack]
    all_hessians = [] # per particle
    for particle in particles:
      smooth_hessians = []
      for component in range(n_components):
        dimension = tf.size(particles[0][component])
        if hessians[0][component] is None:
          smooth_hessians.append(tf.zeros([dimension, dimension], dtype))
          continue
        with tf.GradientTape(watch_accessed_variables=False) as tape:
          tape.watch(inflated_particle_pack)
          kernels = evaluate_kernel(inflated_particle_pack, [tf.stop_gradient(p) for p in particle]) # attempt to make the number of gradient calls linear as opposed to quadratic in the number of particles
        component_jacobian = tape.batch_jacobian(kernels, inflated_particle_pack[component]) # venture over to iterate_particle for an explanation
        kernel_gradients = component_jacobian[:, 0, 0, ...]
        kernels = kernels[:, 0]
        product = tf.zeros([dimension, dimension], dtype)
        kernel_product = tf.zeros([dimension], dtype)
        for hessian, reference_particle in zip(hessians, range(n_particles)): # runtime is dwarfed by actual likelihood computations, but the initialization here is hefty even with the linearization of the number of gradient calls
          #kernel = evaluate_kernel(reference_particle, [tf.stop_gradient(p) for p in particle])
          #kernel_gradient = tf.gradients(kernel, reference_particle[component])
          kernel = kernels[reference_particle]
          kernel_gradient = kernel_gradients[reference_particle, ...]
          #if kernel_gradient[0] is not None: # returns a single-element list. this branch is for when the Hessian is nonzero---i.e. we do descent on this compartment---but its kernel does not depend on it
          flat_kernel_gradient = tf.reshape(kernel_gradient, [-1])
          #else: COMMENTED OUT BECAUSE WE NOW IMPLORE THAT ALL GRADIENTS ARE CONNECTED
          #  flat_kernel_gradient = tf.zeros_like(hessian[component][..., 0]) # get just one column because we want it vector-sized
          product += hessian[component] * tf.square(kernel)
          kernel_product += tf.tensordot(flat_kernel_gradient, flat_kernel_gradient, axes=0) # why did I concoct this instead of flat_kernel_gradient[:, None] * flat_kernel_gradient[None, :] at the time?
        covariance = (-product + kernel_product) / n_particles # FUNDAMENTAL BUG WENT UNDER THE RADAR FOR MONTHS: missed the minus sign on the product. Only because I waited that long before running elucidative "unit" tests in verification.py. This turned out to not have been that much of an issue because in multivariate.py I had flipped the sign once more.
        # tf.print(tf.math.real(tf.linalg.eigvals(-covariance))) ALL OF THESE LOOKED NEGATIVE (GOOD)
        smooth_hessians.append(-covariance) # If the original Hessian is positie semidefinite, then this smoothed version should be too.
      all_hessians.append(smooth_hessians)
    return all_hessians

  @tf.function #(experimental_compile=True) # XLA has limited support for TensorFlow operations, e.g. it can't do cumprod.
  def iterate_particle(*args): # compile this chunk as a single node to avoid bloating the computation graph
    particle = args[:n_components]
    particle_hessian = args[n_components:(2*n_components)]
    gradients = args[2*n_components]
    old_particles = args[2*n_components + 1]
    verbose = args[2*n_components + 2]
    particle_pack = [tf.stack([op[c] for op in old_particles], axis=0) for c in range(n_components)] # a lot of redundant transposing going on...
    #kernels = [0.0 for _ in range(n_particles)]
    #kernel_gradient_sum = [] # no longer automatically assumes the shape of particles via tape.gradient(kernel_sum, particles)
    #for i, reference_particle in enumerate(old_particles):
    #  with tf.GradientTape() as tape:
    #    tape.watch(reference_particle) # make sure this DOES NOT indirectly end up watching 'particle'
    #    kernels[i] = evaluate_kernel(reference_particle, [tf.stop_gradient(p) for p in particle])
    #  kernel_gradient_sum.append(tape.gradient(kernels[i], reference_particle)) # gradient of a kernel is only nonzero for the reference_particle it used
    kernel_gradient_sum = [[] for p in range(n_particles)]
    inflated_particle_pack = [pc[:, None, ...] for pc in particle_pack]# to ensure we have a distinguishable batch dimension, and we still don't throw off the kernel evaluator
    for particle_pack_component in inflated_particle_pack:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(particle_pack_component)
        kernels = evaluate_kernel(inflated_particle_pack, [tf.stop_gradient(p) for p in particle]) # should be a vector.
      component_gradient = tape.batch_jacobian(kernels, particle_pack_component) # clever way to attain the batched gradient
      for p in range(n_particles):
        kernel_gradient_sum[p].append(component_gradient[p, 0, 0, ...]) # clear out the Nones that have propagated and multiplied. a drop-in replacement for the previous definition and structure?
    unstacked_kernels = tf.unstack(kernels[:, 0], axis=0)
    if verbose: tf.print(f"{tag} kernels",
      tf.strings.reduce_join(tf.strings.as_string(tf.cast(kernels, tf.float32), precision=3), separator=" "), output_stream=sys.stdout) # cast because if dtype happens to be float16, tf.strings doesn't accept it
    first_perturbations = []
    particle_coefficients = [] # for do_newton
    for component in range(n_components):
      first_variation = tf.zeros_like(particle[component])
      for kernel, gradient, kernel_gradient in zip(unstacked_kernels, gradients, kernel_gradient_sum):
        perturbation = kernel * clear_null(gradient[component], particle[component]) + clear_null(kernel_gradient[component], particle[component]) # FIRST VARIATION
        first_variation += perturbation / tf.cast(n_particles, dtype)
      first_perturbations.append(first_variation)
      if do_newton:
        #with tf.device("/cpu:0"): # force this pinv on the CPU, because cuSolverDN is not that great (crashes if matrix is degenerate?). Am I forcing one particular core here?
        safe_hessian = tf.debugging.check_numerics(particle_hessian[component], "particle Hessian") # do NaNs occasionally lead to a segfault, if not caught before flowing into the pinv solver?
        flat_coefficients = tf.linalg.matvec(tf.linalg.pinv(-safe_hessian), tf.reshape(first_variation, [-1])) # pinv's validate_args does static type checking, basically
        coefficients = tf.reshape(flat_coefficients, tf.shape(particle[component]))
        particle_coefficients.append(coefficients)
    return first_perturbations, particle_coefficients, unstacked_kernels

  def select_particle(particle, component): # adds a dimension to the front of component with one_hot(particle, n_particles)
    selector = tf.reshape(tf.one_hot(particle, n_particles, dtype=dtype), [-1] + ([1] * len(component.shape)))
    return component * selector

  @tf.function
  def iterate(t, learning_rate):
    verbose = tf.random.uniform([]) < kernel_verbosity_freq # nondeterministic logging! avoids harmonizing/synchronizing with other processes
    trans_particles = [[p[particle, ...] for p in particles] for particle in range(n_particles)]
    if parallel_bundle_size > 1:
      assert n_particles % parallel_bundle_size == 0
      bundled_trans_particles = [
        [p[particle:(particle+parallel_bundle_size), ...] for p in particles]
        for particle in range(0, n_particles, parallel_bundle_size)]
      def unbundle(quantities, bundle_size=parallel_bundle_size): # fancy python multi-order list-comprehension syntax
        return [[bundle[p, ...] for bundle in var_bundle] for var_bundle in quantities for p in range(bundle_size)]
      assert unbundle(np.array([[[1, 2], [11, 12]], [[3, 4], [13, 14]], [[5, 6], [15, 16]]]), 2) == [[1, 11], [2, 12], [3, 13], [4, 14], [5, 15], [6, 16]] # quick sanity check
    else:
      bundled_trans_particles = trans_particles
    if do_newton:
      gradients_and_hessians = [get_log_likelihood_gradient_and_hessian(particle) for particle in bundled_trans_particles]
      #with tf.control_dependencies([gradients_and_hessians[0][1][2]]): tf.print("stein 1") # debugging paradigm
      gradients = [p[0] for p in gradients_and_hessians]
      raw_hessians = [p[1] for p in gradients_and_hessians]
      if parallel_bundle_size > 1: # unpack the results, so that it all looks the same to our core Stein implementation
        gradients = unbundle(gradients)
        raw_hessians = unbundle(raw_hessians) # no more use for tf.unstack(bundle, num=parallel_bundle_size), where we force the first dimension to be of that size.
      hessians = estimate_kernelized_hessians(trans_particles, raw_hessians)
    #elif do_newton and estimate_covariance: # use particles as empirical histogram of distributional approximation. RENDERED INADVISABLE
    else:
      gradients = [get_log_likelihood_gradient(particle) for particle in bundled_trans_particles]
      if parallel_bundle_size > 1:
        gradients = unbundle(gradients)
      hessians = [[tf.eye(tf.size(gradient), dtype=dtype) if gradient is not None else None
        for gradient in p_gradients] for p_gradients in gradients]
    old_particles = [[tf.identity(p) for p in particle] for particle in trans_particles] # copy old values
    first_variation_perturbations = []
    second_variation_coefficients = [] # for do_newton
    all_kernels = []
    for particle, particle_hessian in zip(range(n_particles), hessians):
      first_perturbations, particle_coefficients, particle_kernels = iterate_particle(
        *trans_particles[particle], *particle_hessian, gradients, old_particles, verbose) # AHH: iterate_particle actually retraces here. TF only reported this when I enabled experimental_compile? Can't do tf.ragged when ranks differ. Expand all into a long sequence of arguments?
      first_variation_perturbations.append(first_perturbations)
      second_variation_coefficients.append(particle_coefficients)
      all_kernels.append(particle_kernels)
      if optimize is not None: # like post-processing for each step. Make sure to do stop_gradients on the likelihood function for the parameters that are optimized here instead.
        opt_particle = optimize(*trans_particles[particle])
        for component in range(n_components):
          delete = select_particle(particle, -particles[component][particle, ...])
          particles[component].assign_add(delete)
          update = select_particle(particle, opt_particle[component])
          particles[component].assign_add(update)
    get_learning_rate = lambda component: tf.broadcast_to(learning_rate, [n_components])[component] if len(learning_rate.shape) > 0 else learning_rate
    if do_newton: # Newton replacement for the gradient descent step
      for particle, particle_kernels in zip(range(n_particles), all_kernels):
        for component in range(n_components):
          our_learning_rate = get_learning_rate(component)
          second_variation = tf.zeros_like(second_variation_coefficients[0][component])
          for particle_coefficients, kernel in zip(second_variation_coefficients, particle_kernels):
            second_variation += particle_coefficients[component] * kernel
          step = our_learning_rate * tf.reshape(second_variation, tf.shape(particles[component][particle, ...]))
          particles[component].assign_add(select_particle(particle, step)) # heretofore flattened
    else:
      for particle, perturbation in zip(range(n_particles), first_variation_perturbations):
        for component in range(n_components):
          our_learning_rate = get_learning_rate(component)
          step = our_learning_rate * perturbation[component]
          particles[component].assign_add(select_particle(particle, step))
    #if verbose: tf.print("Done with iteration", t, "with particles\n",
    #  [[tf.strings.reduce_join(tf.strings.as_string(p, precision=2, scientific=True), separator=" ") for p in particle] for particle in old_particles],
    #  "\nand gradients (min, 25%, 50%, 75%, max)\n",
    #  [[tf.strings.reduce_join(tf.strings.as_string(tfp.stats.quantiles(gradient, 4), precision=2, scientific=True), separator=" ")
    #    if gradient is not None else gradient for gradient in p_gradients] for p_gradients in gradients], "\b.")

  start_time = None
  for t in tf.range(n_iterations): # Make t a Tensor, although this is still not traced.
    if callable(rate):
      learning_rate = tf.convert_to_tensor(rate(t))
    else:
      rate = tf.convert_to_tensor(rate)
      if tf.rank(rate) > 0:
        if tf.shape(rate)[0] > 1:
          learning_rate = rate[t, ...]
        else:
          learning_rate = rate[0, ...]
      else:
        learning_rate = rate
    iterate(t, learning_rate) # compile part. it recompiles each time we call this flow_particles function, but that's okay. keyboard interrupts work with this setup
    if t % bandwidth_iteration_interval == 0: # if bandwidth sample size can't go lower without degeneracies, we can speed things up this way
      select_best_bandwidth()
    if report_progress is not None:
      report_progress(t, *particles)
    if start_time is not None:
      seconds_passed = tf.cast(tf.timestamp() - start_time, dtype)
      tf.print(colored("Progress.", "green"), tf.cast(seconds_passed, tf.int32), "seconds to reach iteration", t, "out of", n_iterations, "\b.",
        "The estimated total number of hours is", tf.cast(seconds_passed / tf.cast(t, dtype) * tf.cast(n_iterations, dtype) / 3600.0, tf.int32), "\b.", # t is implicitly the number of iterations run minus one
        output_stream=sys.stdout)
    else:
      start_time = tf.timestamp() # ignore the first iteration in order to get a more accurate representation of running time sans compilation time
  return particles, seconds_passed

## NOTES
# Yielding does not work with tf.functions because they cannot return generators. Can be done statically, though.
# Neither variables nor tensors support indexed item assignment...
