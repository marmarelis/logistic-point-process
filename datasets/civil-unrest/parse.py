import pandas as pd
import numpy as np
import datetime as dt
import argh
import itertools as itt
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import json
from termcolor import colored
from tqdm import tqdm

def parse_events(df, heading_size, top_k, reference_date):
  action_id, spatial_id = "EVENT_TYPE", "ADMIN1" # more refined identifiers: "SUB_EVENT_TYPE" and ("ADMIN2", "LOCATION")
  type_id = [action_id, spatial_id]
  heading = df.sort_values(by="EVENT_DATE").iloc[:heading_size, :]
  frequencies = heading.groupby(type_id)["EVENT_DATE"].aggregate(len) # pick an arbitrary column and get lengths of each group. (multi-)index of this new series is type_id
  if len(frequencies) == 0:
    return [], [], []
  if len(frequencies) < top_k:
    top_k = len(frequencies)
  frequency_threshold = frequencies.sort_values()[-top_k]
  event_type_counter = itt.count(start=0)
  event_types, event_times = [], []
  type_labels = []
  for label, type_df in df.groupby(type_id): # groupby is always sorted, but we can't guarantee that all elements here also appear in the header
    #label = type_df[action_id].iloc[0], type_df[spatial_id].iloc[0]
    if label not in frequencies.index:
      continue
    freq = frequencies[label[0], label[1]]
    if freq < frequency_threshold:
      continue
    event_type = next(event_type_counter) # I remember it was a big deal to me in the past that I prefer these fundamental constructs like next(..) to be functions and not member methods. Julia does it this way too
    times = type_df["EVENT_DATE"].apply(lambda d: d.toordinal())
    event_types.extend([event_type]*len(times))
    event_times.extend(times)
    type_labels.append(label)
  sorted_indices = np.argsort(event_times)
  final_event_types = np.int32(event_types)[sorted_indices]
  final_event_times = np.float32(event_times)[sorted_indices] - reference_date.toordinal()
  return final_event_types, final_event_times, type_labels

"""  heading_size_for_statistics:
       to avoid taking a putative test set into account when determining which event types are worthwhile in the composition of the record
"""
def sort_out(filename, from_year, heading_size_for_statistics, top_k, min_n_events=2000, sheet_name="Sheet1"):
  # why is xlrd being deprecated in favor of openpyxl when the latter cannot even open certain .xlsx files??
  df = pd.read_excel(f"input/{filename}", sheet_name=sheet_name, engine="xlrd")
  for country, country_df in tqdm(df.groupby("COUNTRY")):
    current_entries = country_df["YEAR"] >= int(from_year)
    reference_date = dt.datetime(year=int(from_year), month=1, day=1)
    event_types, event_times, labels = parse_events(
      country_df[current_entries], int(heading_size_for_statistics), int(top_k), reference_date)
    prefix = filename.split('.')[0]
    n_types = len(labels)
    n_events = len(event_types)
    if n_events < min_n_events:
      continue
    tqdm.write(' '.join([colored(country, "red"), ';  '.join(', '.join(label) for label in labels)]))
    name = f"{prefix}-{country}-since{from_year}-top{top_k}_{n_types}-{n_events//1000}k-events"
    with open(f"output/{name}.msgpack", "wb") as f:
      msgpack.dump([event_types, event_times], f)
    with open(f"output/{name}-labels.json", "w") as f:
      json.dump(labels, f)


argh.dispatch_commands([sort_out])
