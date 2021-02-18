# retrieved from
#  https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0/data
# on 2/10/2021

import numpy as np
import pandas as pd
import datetime as dt
import json
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import sys
sys.path.append("../../")

import interpolate_aggregates as interp

n_types = int(sys.argv[1])
subsample_unit = int(sys.argv[2])
bundesland = sys.argv[3]
test_set_size = 1000

df = pd.read_csv("RKI_COVID19.csv")

land_df = df[df["Bundesland"] == bundesland].sort_values(by=["Refdatum"])
days = land_df["Refdatum"].map(
  lambda s: dt.date.fromisoformat(
    s.replace('/', '-').split(' ')[0]).toordinal())
places_sorted = np.sort(land_df["Landkreis"].unique())
days_sorted = np.arange(days.min(), days.max()+1)
# summing because there are many matches. I believe they differentiate by age group as well (Altersgruppe2)
cases = [[land_df[(land_df["Landkreis"]==j) & (days == i)]["AnzahlFall"] for j in places_sorted] for i in days_sorted]
clean_cases = np.array([[x.sum() if len(x)>0 else np.nan for x in a] for a in cases])
cumulative_cases = np.nancumsum(clean_cases, axis=0)
upsampled = interp.upsample(cumulative_cases.T, 24*60, unit=subsample_unit)
ranked_places = np.argsort(-upsampled[:, -test_set_size])
filtered_series = upsampled[ranked_places[:n_types], :]
event_types, event_times = interp.demarcate_events(filtered_series, 24*60)
type_labels = places_sorted[ranked_places[:n_types]]

end_date = land_df["Refdatum"].values[-1].replace('/', '_').split(' ')[0]

with open(f"germany-{bundesland}-{n_types}places-until{end_date}-labels.json", "w") as f:
  json.dump(type_labels.tolist(), f)
with open(f"germany-{bundesland}-{n_types}places-until{end_date}.msgpack", "wb") as f:
  msgpack.dump([event_types, event_times], f)
