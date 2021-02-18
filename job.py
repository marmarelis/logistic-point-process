import sys
import os
import subprocess
import shlex
import argh
from termcolor import colored
from configparser import ConfigParser

@argh.arg("-m", "--modes", nargs="*", type=str)   # perhaps, when it suits me, I can interleave commands like --modes cross --config <cross-specific flags>
@argh.arg("-g", "--gpu-indices", nargs="*", type=int)
def run(import_dir, prefix, config_file, export_dir="results", gpu_indices=[], # for now we define a mode to mean a very SPECIFIC thing, with options hardcoded. only the homogeneous flags may vary if necessary
    homogeneous_flags="--infer-n-event-types --excess-suppression-reg=1e-5 --horseshoe-precision=1e3 --initial-excitation=0.01",
    validation_flags="--n-events=1000 --n-test-events=1000", nonvalidation_flags="--n-test-events=4000", additional_flags="", extra_tag="", out_prefix="",
    n_particles=10, modes=None, do_newton=False, batch_size=None, fisher_subsample=None, learning_rates=None, parallel_bundle_size=None,
    truncate_record=-1, validate=False, dtype=""):

  config = ConfigParser()
  config.read(f"configs/{config_file}.cfg")
  homogeneous_flags += " --excite-kernel={}".format(config["kernel"]["excite"])
  homogeneous_flags += " --supp-kernel={}".format(config["kernel"]["supp"])
  homogeneous_flags += " --rate-kernel={}".format(config["kernel"]["rate"])


  mode_flags = {
    "": "",
    "cross": "--cross-suppressions",
    "crossF": "--cross-suppressions --learning-rates 1e1 1e1 1e1 1e1 1e1",
    "learnhawkes": "--learn-hawkes --hawkes-stability-reg=0",
    "learnhawkesS": "--learn-hawkes",
    "learnhawkesN": "--permanent-suppression-rate='1e-15'" # I suppose this one is not in a dire need of stability like normal Hawkes is
  }

  if modes is None:
    modes = ["", "cross", "crossF", "learnhawkes", "learnhawkesS"] if not do_newton else ["", "cross", "learnhawkes", "learnhawkesS"] # crossF is irrelevant with Newton's method
  base_command = "python3 src/experiment.py stein --silent"
  homogeneous_flags += f" --n-particles={n_particles}"
  if len(additional_flags) > 0:
    homogeneous_flags += f" {additional_flags}"
  if do_newton:
    homogeneous_flags += " --do-newton"
    if fisher_subsample is not None:
      homogeneous_flags += " --fisher-subsample={} --conservative-fisher-subsample".format(int(fisher_subsample))
  if batch_size is not None:
    homogeneous_flags += " --batch-size={}".format(int(batch_size)) # common use-case to alter this without wanting to repeat the other flags
  if parallel_bundle_size is not None:
    homogeneous_flags += " --parallel-bundle-size={}".format(int(parallel_bundle_size))
  if learning_rates is not None: # simple bypasses
    homogeneous_flags += f" --learning-rates {learning_rates}"
  if len(dtype) > 0:
    homogeneous_flags += f" --use-dtype={dtype}"

  current_gpu = 0 # if number of GPUs exceeds the number of modes, then a validation may be kicked off on a result that has yet to complete
  assert 0 < len(gpu_indices) <= len(modes)
  running_processes = [None] * len(gpu_indices)

  def recover_process(process):
    if process is not None:
      print(f"Waiting for process @ slot {current_gpu} to finish.")
      process.wait() # wait for the guy on our scheduled GPU to complete. TODO; more flexible slots. also, sadly, process.wait() is degenerate when shell=True
      stdout, stderr = process.communicate()
      if stderr is not None:
        print("{}: {}".format(colored("ERROR", "red"), stderr))

  for validation in ([False, True] if validate else [False]):
    for mode in modes:
      running_process = running_processes[current_gpu]
      recover_process(running_process)
      environment = os.environ.copy()
      environment["CUDA_VISIBLE_DEVICES"] = str(gpu_indices[current_gpu]) # also TODO: detect colorful output and print that?
      flags = "{} {} --tag='{}'".format(mode_flags[mode], homogeneous_flags, (mode if mode != "" else "vanilla") + (" validation" if validation else ""))
      experimental_setting = f"{mode}_" if mode != "" else ""
      if do_newton:
        experimental_setting = "newton_" + experimental_setting
      if len(extra_tag) > 0:
        experimental_setting = f"{extra_tag}_{experimental_setting}"
      if truncate_record > 0:
        experimental_setting = "trunc_" + experimental_setting
      if validation:
        input_filename = f"{export_dir}/{prefix}_{experimental_setting}result.msgpack" # validation *ONLY* occurs after a normal experimental trial
        output_filename = f"{export_dir}/{prefix}_{experimental_setting}result_validation.msgpack"
        flags += f" {validation_flags}"
        command = f"{base_command} --validate-other-result={input_filename} --export-filename={output_filename} {flags}"
      else:
        if truncate_record > 0:
          flags += f" --n-events={truncate_record} --truncate-record"
        input_filename = f"{import_dir}/{prefix}.msgpack"
        out_name = f"{out_prefix}-{prefix}" if len(out_prefix) > 0 else prefix
        output_filename = f"{export_dir}/{out_name}_{experimental_setting}result.msgpack"
        flags += f" {nonvalidation_flags}"
        command = f"{base_command} --import-filename={input_filename} --export-filename={output_filename} {flags}"
      print(colored("RUNNING", "green"), command, colored(f"on slot {current_gpu}.", "green"))
      process = subprocess.Popen(shlex.split(command), env=environment, shell=False, stdout=None, stderr=None, universal_newlines=True) # can do stderr=subprocess.PIPE; there is absolutely NO documentation on what the "errors" optional argument does
      running_processes[current_gpu] = process
      current_gpu += 1
      if current_gpu == len(gpu_indices):
        current_gpu = 0

  for running_process in running_processes:
    recover_process(running_process)

argh.dispatch_commands([run])
