import pandas as pd
import numpy as np

META_NON = ['participant_id', 'visit', 'pwr_current',
            'vig_activity', 'race_1', 'race_2',
            'wt_gain', 'weight_change', 'age_delivery_range2',
            'num_alc_weekly', 'caffeine', 'smoke_current',
            'ppg_BMI', 'first_BMI']

for visit in [2, 3, 4]:
    old_df = pd.read_csv(f"../../result/data/meta+nmr/DF{visit}_filtered.csv")
    new_df = pd.read_csv(f"../test/result/data/meta+2dnmr/DF{visit}_filtered_2d.csv")

    nmr_cols = [c for c in old_df.columns if c not in META_NON]
    diff = (old_df[nmr_cols] - new_df[nmr_cols]).abs()

    print(f"\n===== V{visit} =====")
    print("max abs diff:", diff.max().max())
    print("median abs diff:", diff.stack().median())
    print("num cells diff > 1e-8:", (diff > 1e-8).sum().sum())
    print("num cells diff > 1e-4:", (diff > 1e-4).sum().sum())
    print("num cells diff > 1e-2:", (diff > 1e-2).sum().sum())