import pandas as pd

# Load dataframe
#df = pd.read_pickle("Monsoon_Data/Processed_Data/Models/dry_spell/imd_clim_mok_date_wide.pkl")
df = pd.read_pickle("Monsoon_Data/Processed_Data/Models/dry_spell_strict/imd_clim_mok_date_2mm_wide.pkl")

#df.to_csv("Monsoon_Data/Processed_Data/Models/mr_onset_idx.csv", index=False)
df.to_csv("Monsoon_Data/Processed_Data/Models/dry_spell_strict/mr_onset_idx.csv", index=False)

# onset date for specific year
#df_2020 = df[df["year"] == 2020]
#df_2020 = (
#        df_2020.rename(columns={"id": "adm3_name"})
#        .rename(columns={"onset_day": "median_onset_idx"})
#)
#
#df_2020.to_pickle("Monsoon_Data/Processed_Data/Models/mr_onset_idx_2020_by_id.pkl")


# Calculate median onset_idx for each id
stats_df = (
    #df.groupby("id", as_index=False)["onset_idx"]
    df.groupby("id", as_index=False)["onset_day"]
      .median()
      .rename(columns={"id": "adm3_name"})
      #.rename(columns={"onset_idx": "median_onset_idx"})
      .rename(columns={"onset_day": "median_onset_idx"})
      #.rename(columns={"onset_idx": "tp"})
)

# Save output
#stats_df.to_csv("Monsoon_Data/Processed_Data/Models/mr_onset_idx_median_by_id.csv", index=False)
stats_df.to_csv("Monsoon_Data/Processed_Data/Models/dry_spell_strict/mr_onset_idx_median_by_id.csv", index=False)

# Optional: save as pickle too
#stats_df.to_pickle("Monsoon_Data/Processed_Data/Models/mr_onset_idx_median_by_id.pkl")
stats_df.to_pickle("Monsoon_Data/Processed_Data/Models/dry_spell_strict/mr_onset_idx_median_by_id.pkl")

print(stats_df.head())
