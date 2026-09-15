"""Select new source clips before prediction, and index only their RGB members."""
from pathlib import Path
import hashlib
import json
import mmap
import struct
import xml.etree.ElementTree as E
import zipfile
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/v5_external/dada'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
CATEGORIES = ['8','9','10','11','49','50']


def annotations():
    with zipfile.ZipFile(OUT/'official_annotation.xlsx') as z:
        ss=[''.join(x.itertext()) for x in E.fromstring(z.read('xl/sharedStrings.xml'))]
        rows=[]
        for row in list(E.fromstring(z.read('xl/worksheets/sheet2.xml')).find('s:sheetData',NS))[1:]:
            d={}
            for c in row:
                v=c.find('s:v',NS);v=v.text if v is not None else ''
                column=''.join(ch for ch in c.attrib['r'] if ch.isalpha())
                d[column]=ss[int(v)] if c.attrib.get('t')=='s' else v
            if d.get('F') in CATEGORIES and d.get('G')=='1':
                try:
                    n=int(float(d['K'])); collision=int(float(d['I']))
                except (ValueError,KeyError):
                    continue
                if 100<=n<=500 and 1<=collision<=n:
                    key=d['F']+'/'+str(int(d['A'])).zfill(3)
                    rows.append({'source_key':key,'category':d['F'],'clip':str(int(d['A'])).zfill(3),
                        'provided_accident_frame':collision,'provided_abnormal_start':d.get('H'),
                        'provided_frame_count':n,'provided_description':d.get('Q'),
                        'selection_hash':hashlib.sha256(('v5-dada-validation-20260913:'+key).encode()).hexdigest()})
        return rows


def main():
    selection_path=OUT/'selection_frozen.json'
    if selection_path.exists() and '--resume-index' not in sys.argv:
        raise FileExistsError('Existing selection is frozen')
    rows=annotations();selected=[]
    for category in CATEGORIES:
        eligible=sorted((r for r in rows if r['category']==category),key=lambda r:r['selection_hash'])
        for rank,row in enumerate(eligible[:4]):
            row['rank_in_category']=rank
            row['role']='primary' if rank<2 else 'reserve'
            selected.append(row)
    selection={'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Twelve primary new clips from six categories; two reserves/category selected before any image inspection or prediction.',
        'protocol':{'categories':CATEGORIES,'eligibility':'Provided accident=1, 100..500 frames, valid provided accident frame.',
            'ranking':'SHA256 of v5-dada-validation-20260913:<category>/<clip>; first two primary, next two reserves.',
            'replacement':'Only corrupt/missing data, public duplicate, or no discernible ego-vehicle versus vehicle contact; record reason before predictions. Never replace based on candidate performance.',
            'labels':'Author accident frame is not automatically DACON first-contact GT. Blind scene review required. Missing entry/side/space remain unknown unless separately evidenced. AI review is not official GT.',
            'independence':'New clips, but unknown original video/channel/camera grouping. Similarity audit cannot prove all hidden duplicates absent.',
            'use':'No training, threshold selection or ID-specific rules. Frozen V3/V5 paired external validation after blind review.'},
        'sources':selected}
    if selection_path.exists():
        original=json.loads(selection_path.read_text(encoding='utf-8'))
        if original['sources']!=selected or original['protocol']!=selection['protocol']:
            raise AssertionError('Resume would alter frozen source selection')
        selection=original
    else:
        selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    targets={r['source_key']:[] for r in selected}
    total=0
    with (OUT/'central_directory.bin').open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as b:
        pos=0
        while pos<len(b):
            v=struct.unpack_from('<4s6H3L5H2L',b,pos)
            if v[0]!=b'PK\x01\x02':
                raise AssertionError(f'Invalid central record at {pos}')
            name=b[pos+46:pos+46+v[10]].decode('utf-8' if v[3]&0x800 else 'cp437',errors='strict')
            parts=name.split('/')
            if (len(parts)==5 and parts[0]=='DADA2000' and parts[3]=='images'
                    and parts[4].lower().endswith(('.png','.jpg','.jpeg'))
                    and '/'.join(parts[1:3]) in targets):
                uncompressed,compressed,offset,disk=v[9],v[8],v[16],v[13]
                extra=bytes(b[pos+46+v[10]:pos+46+v[10]+v[11]])
                p=0
                while p+4<=len(extra):
                    kind,size=struct.unpack_from('<HH',extra,p);data=extra[p+4:p+4+size];p+=4+size
                    if kind==1:
                        cursor=0
                        values=[uncompressed,compressed,offset,disk]
                        for i,width in enumerate([8,8,8,4]):
                            if values[i]==(0xffff if i==3 else 0xffffffff):
                                values[i]=int.from_bytes(data[cursor:cursor+width],'little');cursor+=width
                        uncompressed,compressed,offset,disk=values
                targets['/'.join(parts[1:3])].append({'path':name,'filename':parts[4],
                    'flags':v[3],'method':v[4],'crc32':v[7],'compressed_bytes':compressed,
                    'uncompressed_bytes':uncompressed,'disk':disk,'local_offset':offset})
            pos+=46+v[10]+v[11]+v[12];total+=1
    for r in selected:
        members=targets[r['source_key']]
        if len(members)!=r['provided_frame_count']:
            print(json.dumps({'warning':'annotation/image count mismatch','source':r['source_key'],'annotation':r['provided_frame_count'],'members':len(members)}),flush=True)
    report={'central_records':total,'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            'members_by_source':targets}
    (OUT/'selected_members.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'records':total,'selection_sha256':report['selection_sha256'],
        'sources':[{'source':r['source_key'],'role':r['role'],'frames':len(targets[r['source_key']])} for r in selected]}),flush=True)


if __name__=='__main__':
    main()
