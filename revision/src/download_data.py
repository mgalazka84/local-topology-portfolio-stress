"""Retrieve the original French industry files, record hashes, and parse one panel.
Raw source files are not included in the code distribution.
"""
from pathlib import Path
import urllib.request,zipfile,io,re,hashlib,json,datetime,warnings
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'data').mkdir(exist_ok=True)

def download(n):
    url=f'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{n}_Industry_Portfolios_daily_CSV.zip'
    path=ROOT/'data'/f'{n}_industry_original.zip'
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as r:path.write_bytes(r.read())
    raw=path.read_bytes()
    archive=ROOT/'data'/'archived_provenance.json'
    if archive.exists() and str(n) in json.loads(archive.read_text()):
        expected=json.loads(archive.read_text())[str(n)]['sha256_original_zip']
        if hashlib.sha256(raw).hexdigest()!=expected:
            warnings.warn('Source-data vintage differs from the archived experiment; historical results can change.')
    z=zipfile.ZipFile(io.BytesIO(raw));text=z.read(z.namelist()[0]).decode('utf-8-sig')
    lines=text.splitlines()
    label=next(i for i,x in enumerate(lines) if 'Average Value Weighted Returns' in x)
    header_index=next(i for i in range(label+1,len(lines)) if lines[i].startswith(','))
    rows=[]
    for line in lines[header_index+1:]:
        if not re.match(r'^\s*\d{8},',line):
            if rows:break
            continue
        rows.append(line)
    df=pd.read_csv(io.StringIO(lines[header_index]+'\n'+'\n'.join(rows)),index_col=0)
    df.index=pd.to_datetime(df.index.astype(str),format='%Y%m%d');df.columns=df.columns.str.strip()
    df=df.mask(df<=-99)/100.
    intended=df.loc['1998-01-01':'2025-12-31'].copy()
    # Use only years ending with an observed December, never synthesize dates.
    last=df.index.max()
    final_year=min(2025,last.year if last.month==12 else last.year-1)
    intended=intended.loc[:f'{final_year}-12-31']
    if intended.isna().any().any():raise ValueError('Missing industry returns in selected period')
    intended.to_csv(ROOT/'data'/f'industry_{n}.csv',float_format='%.9f')
    metadata={'url':url,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'sha256_original_zip':hashlib.sha256(raw).hexdigest(),'source_last_date':str(last.date()),
       'selected_start':str(intended.index.min().date()),'selected_end':str(intended.index.max().date()),
       'rows':len(intended),'assets':list(df.columns),'header':lines[:min(label+3,20)],
       'interpretation':'Daily value-weighted research industry portfolios; returns divided by 100; not directly tradeable instruments.'}
    (ROOT/'data'/f'industry_{n}_provenance.json').write_text(json.dumps(metadata,indent=2))
    print(n,metadata['rows'],metadata['selected_start'],metadata['selected_end'],flush=True)

def download_risk_free():
    url='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
    path=ROOT/'data/factors_daily_original.zip'
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response:path.write_bytes(response.read())
    raw=path.read_bytes();archive=zipfile.ZipFile(io.BytesIO(raw))
    lines=archive.read(archive.namelist()[0]).decode('utf-8-sig').splitlines()
    header=next(i for i,line in enumerate(lines) if 'Mkt-RF' in line and 'RF' in line)
    rows=[line for line in lines[header+1:] if re.match(r'^\s*\d{8},',line)]
    df=pd.read_csv(io.StringIO(lines[header]+'\n'+'\n'.join(rows)),index_col=0)
    df.index=pd.to_datetime(df.index.astype(str));df.columns=df.columns.str.strip()
    (df[['RF']]/100).loc['1998':'2025'].to_csv(ROOT/'data/risk_free.csv')
    metadata={'url':url,'sha256_original_zip':hashlib.sha256(raw).hexdigest(),
        'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'units':'Daily RF percent divided by 100'}
    (ROOT/'data/risk_free_provenance.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':
    for n in [10,30,49]:download(n)
    download_risk_free()
