"""Bounded-memory raw CSV scan with original, pre-filter record ordinals."""
from pathlib import Path
import json
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.compute as pc
import pyarrow.parquet as pq
import numpy as np

def extract(path,out,columns,itemids=None,key=None,keys=None):
 path=Path(path);out=Path(out)
 if out.exists():raise FileExistsError(out)
 reader=csv.open_csv(path,read_options=csv.ReadOptions(use_threads=False,block_size=16*1024*1024),convert_options=csv.ConvertOptions(column_types=columns,include_columns=list(columns),strings_can_be_null=True))
 writer=None;offset=0;kept=0
 try:
  for batch in reader:
   table=pa.Table.from_batches([batch]);n=len(table)
   table=table.append_column('source_row_number',pa.array(np.arange(offset+1,offset+n+1,dtype=np.int64)));offset+=n
   if itemids is not None:table=table.filter(pc.is_in(table['itemid'],value_set=pa.array(itemids,type=pa.int64())))
   if key is not None:table=table.filter(pc.is_in(table[key],value_set=pa.array(keys,type=pa.int64())))
   if writer is None:writer=pq.ParquetWriter(out,table.schema,compression='zstd')
   if len(table):writer.write_table(table);kept+=len(table)
 finally:
  if writer is not None:writer.close()
 out.with_suffix('.scan.json').write_text(json.dumps({'raw_records_scanned':offset,'records_retained':kept,'source_ordinal':'1-based data record before filtering','chunk_bytes':16*1024*1024},indent=2))
 print('RAW SCAN COMPLETE',path.name,'records',offset,'retained',kept,flush=True)

def vitals(raw,out,cohort,code_status=False):
 keys=pq.read_table(cohort,columns=['stay_id'])['stay_id'].to_pylist()
 cols={'stay_id':pa.int64(),'charttime':pa.timestamp('us'),'itemid':pa.int64()}
 if code_status:cols['value']=pa.string();items=[223758,229784,228687]
 else:cols.update(valuenum=pa.float64(),valueuom=pa.string());items=[220052,220181,220045,220277,220210,224690,223761,223762]
 extract(raw,out,cols,items,'stay_id',keys)

def labs(raw,out,cohort):
 keys=pq.read_table(cohort,columns=['hadm_id'])['hadm_id'].to_pylist()
 cols={'hadm_id':pa.int64(),'charttime':pa.timestamp('us'),'itemid':pa.int64(),'valuenum':pa.float64(),'valueuom':pa.string()}
 extract(raw,out,cols,key='hadm_id',keys=keys)
