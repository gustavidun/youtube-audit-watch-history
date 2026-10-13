import asyncio
import argparse

from puppet import YTPuppet
from data_fetcher import get_participant_ids

import random
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=45)
parser.add_argument("--train", type=int, default=100)
parser.add_argument("--drift", type=int, default=200)
parser.add_argument("--headless", type=bool, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

N = args.n
TRAIN = args.train
DRIFT = args.drift
HEADLESS = args.headless
LOGGER = args.debug

crashes = 0

async def safe_run(puppet : YTPuppet):
    global crashes
    await asyncio.sleep(random.uniform(0,30))
    try:
        await puppet.run()
    except Exception as e:
        puppet.logger.exception("Puppet crashed", exc_info=e)
        crashes += 1

async def main():

    puppets = []
    ids = get_participant_ids(N)

    for i in range(N):
        puppets.append(
            YTPuppet(
                ids[i],
                slant=0,
                target_slant=0,
                train_depth=TRAIN,
                drift_depth=DRIFT,
                headless=HEADLESS,
                utility=False,
                logger_level=logging.DEBUG if LOGGER else logging.INFO
            )
        )

    async with asyncio.TaskGroup() as tg:
        for p in puppets:
            tg.create_task(safe_run(p))

    return puppets

if __name__ == "__main__":
    puppets = asyncio.run(main())
    print(f"Finished run. Lost {crashes} puppets")

