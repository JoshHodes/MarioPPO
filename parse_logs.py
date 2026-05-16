import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = './logs/'
dirs = [d for d in glob.glob(os.path.join(log_dir, '*')) if os.path.isdir(d)]
latest_run = max(dirs, key=os.path.getmtime) if dirs else None

if not latest_run:
    print("No logs found.")
    exit(0)

print(f"Reading from {latest_run}")
event_acc = EventAccumulator(latest_run)
event_acc.Reload()

tags = event_acc.Tags()['scalars']

metrics = ['curriculum/flag_rate', 'rollout/ep_rew_mean', 'rollout/ep_len_mean']

print("Latest Metrics:")
for m in metrics:
    if m in tags:
        events = event_acc.Scalars(m)
        print(f"  {m}: {events[-1].value:.2f} (at step {events[-1].step})")
    else:
        print(f"  {m}: Not found")
