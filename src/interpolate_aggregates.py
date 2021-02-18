# taken from polynomial-hawkes, with a bug fixed and the rest polished
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
from os import path
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import json
if path.isfile("../style.mplstyle"):
  from matplotlib import pyplot as plt
  from matplotlib import colors as clrs
  plt.style.use("../style.mplstyle")
  plt.ion()
  import seaborn as sb
  from cmcrameri import cm

def upsample(cases, scale, unit=1): # smoothing by logarithmic interpolation between days. must quantize to discrete number of upticks. higher unit, e.g. 10, considers one event after 10 new cases. MUST BE CUMULATIVE CURVES
  upsampled = np.zeros([cases.shape[0], (cases.shape[1]-1) * scale], np.float64)
  for day in range(1, cases.shape[1]):
    first, last = np.float64(cases[:, day-1]), np.float64(cases[:, day])
    interp = np.exp(np.linspace(np.log(first), np.log(last), scale+1, axis=-1)) # if first, last are 0 then the whole thing is nan. if last is nonzero then all but the last entry are nan.
    upsampled[:, ((day-1)*scale):(day*scale)] = interp[:, 1:] / unit
  upsampled = pd.DataFrame(upsampled).fillna(method="ffill", axis="columns").values # forward-fill intermediate values
  np.nan_to_num(upsampled, nan=0.0, copy=False) # there might be some, e.g. at the beginning, that are still NaN
  return np.int32(upsampled) # floors

# NOTE: take np.diff of what's returned from above.

def demarcate_events(cases, n_divisions=1, cumulative=True): # cases are stored as region x time. probably upsampled before calling this. n_divisions tells us how much this was upsampled beforehand.
  rates = np.diff(cases, axis=1) if cumulative else cases
  n_types, n_ticks = rates.shape
  types = np.concatenate([
    np.concatenate([np.full(np.maximum(rates[j, tick], 0), j) for j in range(n_types)], axis=0) for tick in tqdm(range(n_ticks), desc="step one out of two")], axis=0)
  times = np.concatenate([
    np.concatenate([np.full(np.maximum(rates[j, tick], 0), tick/n_divisions) for j in range(n_types)], axis=0) for tick in tqdm(range(n_ticks), desc="step two out of two")], axis=0)
  sorted_ind = np.argsort(times) # THIS IS WHAT I HADN'T DONE BEFORE. With leaky batches, not only is accuracy jumbled: some event influences will flat-out not be seen at all.
  event_types, event_times = types[sorted_ind], times[sorted_ind]
  return event_types, event_times

def parse_tycho_record(filename, states):
  df = pd.read_csv(filename)
  delta = dt.timedelta(days=7)
  tiny_delta = dt.timedelta(days=1) # if off by a few days due to imperfect alignment, move backwards slightly until an entry is found
  state_df = df[df["Admin1Name"].isin(states)].copy() # silence those silly warnings about assigning to an implicit copy
  state_df["PeriodStartDate"] = state_df["PeriodStartDate"].map(lambda date: dt.date.fromisoformat(date))
  min_date, max_date = state_df["PeriodStartDate"].min(), state_df["PeriodStartDate"].max()
  date_indices = {} # lazy but works
  current_date = min_date
  n_slots = 0
  while current_date <= max_date:
    date_indices[current_date] = n_slots
    current_date += delta
    n_slots += 1
  rates = np.zeros((0, n_slots))
  regions = []
  for region, region_df in tqdm(state_df.groupby(["Admin1Name", "Admin2Name"], dropna=True)): # the Admin2Name==NAN entries are "PartOfCumulativeCountSeries," not weekly counts
    regions.append(region)
    tqdm.write("{}".format(region))
    start_dates = region_df["PeriodStartDate"]
    counts = region_df["CountValue"]
    region_rates = np.zeros((1, n_slots))
    current_adjustment = dt.timedelta(days=0)
    for start_date, count in zip(start_dates, counts):
      date = start_date + current_adjustment
      if date not in date_indices: # might not need this branching (can have this loop simply always run), but can help for targeted print statements
        current_date = date
        while current_date not in date_indices:
          current_date -= tiny_delta
          current_adjustment -= tiny_delta
          if current_adjustment <= -delta:
            current_adjustment = dt.timedelta(days=0) # reset if we veer too far off
            current_date = start_date
        i = date_indices[current_date]
        tqdm.write("{} not found, so went back to {}".format(date, current_date))
      else:
        i = date_indices[date]
      region_rates[0, i] = count
    rates = np.concatenate([rates, region_rates], axis=0)
  cases = np.cumsum(rates, axis=1)
  return regions, cases
