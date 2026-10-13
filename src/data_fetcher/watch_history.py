import polars as pl

from config import WATCH_HISTORY_PATH

def get_participant_ids(n):
    return pl.read_parquet(WATCH_HISTORY_PATH)["participant_id"].unique()[:n]   

def get_participant_vids(p_id : str, n = 500):
    pipe = (
        pl.scan_parquet(WATCH_HISTORY_PATH, try_parse_dates=True)
        .filter(pl.col("participant_id") == p_id)
        .sort("time", descending=True)
        .head(n)
    )
    return pipe.collect()["video_id"].to_list()