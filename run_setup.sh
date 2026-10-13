cd SockPuppetTest
cd YouTube-Algorithm-Audit
source .venv/bin/activate
git pull
pip install -r requirements.txt
python -m patchright install chrome --with-deps
cd src
python -m batch_run --n 40 --train 100 --drift 0 --debug