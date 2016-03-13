#!/usr/bin/env python

import numpy as np
import pandas as pd
import collections
from datetime import datetime as dt

## read in
## tell it which columns to treat like date-objects
df = pd.read_csv("2013-3rd-quarter.csv", parse_dates=[1, 4])
df.head()

## take a slice of the dataframe, based on the function
afternoon = df[df["Start date"].map(lambda v: v.hour >= 12)]
print(len(afternoon))
afternoon.head()

aft_starts = afternoon.groupby("Start Station")
aft_ends = afternoon.groupby("End Station")

aft_counts = collections.defaultdict(int)
for station in aft_starts:
    aft_counts[station[0]] += len(station[1])
for station in aft_ends:
    aft_counts[station[0]] -= len(station[1])


## same thing for the morning data: dataframe slice
morning = df[df["Start date"].map(lambda x:x.hour < 12)]
morn_starts = morning.groupby("Start Station")
morn_ends = morning.groupby("End Station")

morn_counts = collections.defaultdict(int)
for station in morn_starts:
    morn_counts[station[0]] += len(station[1])
for station in morn_ends:
    morn_counts[station[0]] -= len(station[1])

print("am max", max(morn_counts.items(), key=lambda x:x[1]))
print("pm max", max(aft_counts.items(), key=lambda x:x[1]))

