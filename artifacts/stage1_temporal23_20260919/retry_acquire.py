import sys,time
sys.path.insert(0,'scripts/data')
import experiment_stage1_temporal23 as e
pool=e.acq.concurrent.futures.ThreadPoolExecutor
e.acq.concurrent.futures.ThreadPoolExecutor=lambda max_workers:pool(max_workers=2)
e.acq.VD += '&temporal_retry='+str(time.time_ns())
e.acquire()
