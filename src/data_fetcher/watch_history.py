import polars as pl

from config import WATCH_HISTORY_PATH
from logger import setup_logger

logger = setup_logger("Database")

def get_participant_ids(n):
    logger.info(f"Fetching {n} participant IDs")
    return pl.read_parquet(WATCH_HISTORY_PATH)["participant_id"].unique()[:n]   

def get_participant_vids(p_id : str, n = 500):
    logger.info(f"Fetching {n} videos for participant {p_id}")
    pipe = (
        pl.scan_parquet(WATCH_HISTORY_PATH)
        .filter(pl.col("participant_id") == p_id)
        .sort("time", descending=True)
        .head(n)
    )
    return pipe.collect()["video_id"].to_list()