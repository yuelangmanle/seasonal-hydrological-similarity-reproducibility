"""Import the five study regions and provider notices from an official Caravan tarball."""
from pathlib import Path, PurePosixPath
import argparse,tarfile,io,hashlib,json
from urllib.request import urlopen
import pandas as pd
REGIONS={'camels','lamah','camelscl','camelsbr','hysets'}
COLS=['date','streamflow','total_precipitation_sum','temperature_2m_mean']

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,required=True,help='Caravan-csv.tar.gz from release 15530022')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    with urlopen('https://zenodo.org/api/records/15530022',timeout=30) as response:meta=json.load(response)
    entry=next(f for f in meta['files'] if f['key']=='Caravan-csv.tar.gz')
    algorithm,expected=entry['checksum'].split(':',1);h=hashlib.new(algorithm)
    with args.archive.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('Archive checksum differs from official release')
    args.out.mkdir(parents=True,exist_ok=True);counts={r:0 for r in REGIONS}
    with tarfile.open(args.archive,'r|gz') as archive:
        for member in archive:
            if not member.isfile():continue
            parts=PurePosixPath(member.name).parts
            if '..' in parts or member.name.startswith('/'):raise ValueError('Unsafe archive path')
            if len(parts)>=5 and parts[:3]==('Caravan-csv','timeseries','csv') and parts[3] in REGIONS and member.name.endswith('.csv'):
                dst=args.out/parts[3]/parts[-1];dst.parent.mkdir(exist_ok=True)
                with archive.extractfile(member) as source:
                    frame=pd.read_csv(io.BytesIO(source.read()),usecols=COLS)
                frame.to_csv(dst,index=False);counts[parts[3]]+=1
                if sum(counts.values())%200==0:print(counts,flush=True)
            elif len(parts)>=3 and parts[0]=='Caravan-csv' and parts[1]=='attributes' and parts[2] in REGIONS and member.name.endswith('.csv'):
                dst=args.out/'_attributes'/(parts[2]+'__'+parts[-1]);dst.parent.mkdir(exist_ok=True)
                dst.write_bytes(archive.extractfile(member).read())
            elif len(parts)>=3 and parts[:2]==('Caravan-csv','licenses'):
                dst=args.out/'_licenses'/Path(*parts[2:]);dst.parent.mkdir(parents=True,exist_ok=True)
                dst.write_bytes(archive.extractfile(member).read())
    if any(n==0 for n in counts.values()):raise ValueError(f'Missing regions: {counts}')
    (args.out/'source_record.json').write_text(json.dumps({'doi':meta['metadata']['doi'],'checksum':entry['checksum'],'counts':counts},indent=2))
    print(counts)
if __name__=='__main__':main()
