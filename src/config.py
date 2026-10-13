from pathlib import Path
import os
from dotenv import load_dotenv
import json

HEADLESS = False

ROOT = Path(__file__).parents[1]

load_dotenv(ROOT / ".env")
API_KEY = json.loads(os.environ.get("API_KEY")) #YT Data API Key
MAX_API_ERRORS : int = 5

DB_DIR = ROOT / "data" / "db"
DB_PATH = DB_DIR / "db.sqlite"
WATCH_HISTORY_PATH = ROOT / "data" / "watch_histories" / "video_histories_filtered.parquet"

SESSION_DIR = ROOT / "data" / "session"
LOG_DIR = ROOT / "data" / "logs"
PUPPETS_DIR = ROOT / "data" / "puppets"

UBLOCK_PATH = ROOT / "src" / "puppet" / "extensions"

SLANT_DIR = ROOT / "data" / "slant"
SLANT_ESTIMATIONS_CSV = SLANT_DIR / "slant_estimations.csv"
SLANTS_TRAIN = SLANT_DIR / "slants_train.pkl"

MODELS_DIR = ROOT / "data" / "models"

if not PUPPETS_DIR.exists():
    os.mkdir(PUPPETS_DIR)

if not MODELS_DIR.exists():
    os.mkdir(MODELS_DIR)

if not SESSION_DIR.exists():
    os.mkdir(SESSION_DIR)

if not DB_DIR.exists():
    os.mkdir(DB_DIR)

if not LOG_DIR.exists():
    os.mkdir(LOG_DIR)

if not UBLOCK_PATH.exists():
    os.mkdir(UBLOCK_PATH)