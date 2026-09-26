"""Select only the official mini archive; stop on completion or ten minutes without bytes."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/data_pilot_20260916'
sys.path.insert(0,str(OUT/'download_tools'))
import libtorrent as lt

def main():
    target=OUT/'zod';target.mkdir(exist_ok=True)
    info=lt.torrent_info(str(OUT/'zod_sequences.torrent'))
    files=info.files();chosen=[i for i in range(files.num_files()) if files.file_path(i).endswith('/sequences_mini.tar.gz')]
    assert len(chosen)==1
    priorities=[0]*files.num_files();priorities[chosen[0]]=4
    session=lt.session({'listen_interfaces':'0.0.0.0:0','upload_rate_limit':1024,'connections_limit':20})
    h=session.add_torrent({'ti':info,'save_path':str(target),'file_priorities':priorities})
    last_bytes=-1;last_change=time.monotonic()
    try:
        while True:
            s=h.status();done=h.file_progress()[chosen[0]]
            if done!=last_bytes:last_change=time.monotonic();last_bytes=done
            record={'selected_file':files.file_path(chosen[0]),'expected_bytes':files.file_size(chosen[0]),'completed_bytes':done,'peers':s.num_peers,'download_rate':s.download_rate,'state':str(s.state),'error':str(s.errc),'status':'running'}
            if done==record['expected_bytes'] and s.is_finished:record['status']='complete'
            elif time.monotonic()-last_change>600:record['status']='stopped_no_download_progress'
            (target/'download_status.json').write_text(json.dumps(record,indent=2))
            print(json.dumps(record),flush=True)
            if record['status']!='running':break
            time.sleep(30)
    finally:h.pause();session.pause()
if __name__=='__main__':main()
