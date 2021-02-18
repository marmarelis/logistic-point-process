import numpy as np
import pandas as pd
import datetime as dt
import json
from tqdm import tqdm
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import sys
sys.path.append("../../")

import interpolate_aggregates as interp

df = pd.read_csv(sys.argv[1])
n_types = int(sys.argv[2])
subsample_unit = int(sys.argv[3])
test_set_size = 1000

la_df = df[df['county']=='Los Angeles']
days = la_df['date'].map(lambda s: dt.date.fromisoformat(s).toordinal())
places_sorted = np.sort(df[df['county'] == 'Los Angeles']['place'].unique())
days_sorted = np.arange(days.min(), days.max()+1)
cases = [[la_df[(la_df['place']==j) & (days == i)]['confirmed_cases'] for j in places_sorted] for i in tqdm(days_sorted)]
clean_cases = np.array([[x.iloc[0] if len(x)>0 else np.nan for x in a] for a in cases])
upsampled = interp.upsample(clean_cases.T, 24*60, unit=subsample_unit) # was 16
ranked_places = np.argsort(-upsampled[:, -test_set_size])
filtered_series = upsampled[ranked_places[:n_types], :]
event_types, event_times = interp.demarcate_events(filtered_series, 24*60)
type_labels = places_sorted[ranked_places[:n_types]]

end_date = df['date'].sort_values().values[-1].replace('-', '_')

with open(f"los-angeles-{n_types}places-until{end_date}-labels.json", "w") as f:
  json.dump(type_labels.tolist(), f)
with open(f"los-angeles-{n_types}places-until{end_date}.msgpack", "wb") as f:
  msgpack.dump([event_types, event_times], f)
