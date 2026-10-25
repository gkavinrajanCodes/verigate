"""Inspect the Verigate SQLite store (debug helper)."""
import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "data/verigate.db"
c = sqlite3.connect(db)
q = lambda s: c.execute(s).fetchall()
print("audits by provider/status:", q("select provider,status,count(*) from audits group by 1,2"))
print("tokenization:", q("select tokenization,count(*) from audits group by 1"))
print("mean s2/s3/s2_early by provider:",
      q("select provider,round(avg(s2),3),round(avg(s3),3),round(avg(s2_early),3),round(avg(s4),4) from audits group by 1"))
print("scored_tokens avg:", q("select round(avg(scored_tokens),1), round(avg(prompt_tokens),1) from audits"))
print("null samples:", q("select signal,count(*),round(avg(value),3) from null_samples group by 1"))
print("verdict rows:", q("select count(*) from verdicts"))
print("latest verdicts:", q("select provider,state,n_audits,combined_p from verdicts order by changed_at desc limit 4"))
