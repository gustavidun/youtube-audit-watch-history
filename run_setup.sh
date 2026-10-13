cd youtube-audit-watch-history
source .venv/bin/activate
git pull
pip install -r requirements.txt
python -m patchright install chrome --with-deps
cd src
python -m batch_run --n 100 --train 100 --drift 0