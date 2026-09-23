"""Regenerate the six submission figures from current aggregate results."""
from pathlib import Path
import argparse,csv
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor,black
from pypdf import PdfReader,PdfWriter,Transformation
from pypdf.generic import RectangleObject
from matplotlib import font_manager
import pandas as pd
ap=argparse.ArgumentParser()
for name in ['clinical','mimic-results','eicu-results','posthoc','display','out']:ap.add_argument('--'+name,type=Path,required=True)
args=ap.parse_args();UP=args.out.resolve();UP.mkdir(parents=True,exist_ok=True);W=UP/'figure_work';W.mkdir(exist_ok=True);A=args.clinical
pdfmetrics.registerFont(TTFont('ArialEmbedded',font_manager.findfont('DejaVu Sans')))
pdfmetrics.registerFont(TTFont('ArialBoldEmbedded',font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))))
def rows(n):
 with (A/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def label(c,x,y,s,size=15,bold=False,align='left'):
 c.setFillColor(black);c.setFont('ArialBoldEmbedded' if bold else 'ArialEmbedded',size)
 {'left':c.drawString,'center':c.drawCentredString,'right':c.drawRightString}[align](x,y,s)
def axes(c,x,y,w,h,mx,ticks):
 for t in ticks:
  yy=y+t/mx*h;c.setStrokeColor(HexColor('#dddddd'));c.setLineWidth(.7);c.line(x,yy,x+w,yy);label(c,x-10,yy-5,f'{t:g}',14,align='right')
 c.setStrokeColor(black);c.setLineWidth(1);c.line(x,y,x,y+h);c.line(x,y,x+w,y)
def finish(src,out,rect=None):
 r=PdfReader(src);pg=r.pages[0]
 if rect:
  x,y,x1,y1=rect;pg.add_transformation(Transformation().translate(-x,-y));pg.mediabox=RectangleObject((0,0,x1-x,y1-y));pg.cropbox=pg.mediabox
 pg.scale_to(170/25.4*72,float(pg.mediabox.height)*170/25.4*72/float(pg.mediabox.width))
 # ReportLab installs an unused Helvetica resource by default; remove only that unused font.
 if '/Font' in pg['/Resources']:
  fonts=pg['/Resources']['/Font']
  if '/F1' in fonts and fonts['/F1'].get_object().get('/BaseFont')=='/Helvetica':
   # Its initial 12-point Tf is a setup operation, followed by embedded-font selection before text.
   from pypdf.generic import ContentStream
   cs=ContentStream(pg.get_contents(),r)
   cs.operations=[(v,o) for v,o in cs.operations if not (o==b'Tf' and str(v[0])=='/F1')]
   pg.replace_contents(cs);del fonts['/F1']
 writer=PdfWriter();writer.add_page(pg);writer.add_metadata({'/Title':out.stem,'/Author':'Fuyang Cao'});writer.write(out)

c=canvas.Canvas(str(W/'timing.pdf'),pagesize=(960,425),initialFontName='ArialEmbedded')
bins=rows('restart_time_fixed_intervals.csv');summ=rows('restart_time_summary.csv')
for i,db in enumerate(['MIMIC-IV','eICU']):
 x=68+i*465;y=68;w=390;h=225;mx=75
 s=next(r for r in summ if r['database']==db)
 label(c,x+w/2,395,db,22,True,'center')
 label(c,x+w/2,369,f"Median {float(s['median_hours']):.1f} h [Q1 {float(s['q1_hours']):.1f}, Q3 {float(s['q3_hours']):.1f}]",16,align='center')
 label(c,x+w/2,344,f"Valid times {s['valid_time_n']}/{s['restart_first_total']}",15,align='center')
 label(c,x+w/2,322,'Missing/invalid 0',15,align='center')
 axes(c,x,y,w,h,mx,list(range(0,71,10)))
 for j,r in enumerate(r for r in bins if r['database']==db):
  v=float(r['percent_of_all_restart_first']);cx=x+(j+.5)*w/4
  c.setFillColor(HexColor('#27577f' if i==0 else '#527da7'));c.rect(cx-31,y,62,v/mx*h,fill=1,stroke=0)
  label(c,cx,y+v/mx*h+10,str(int(r['events'])),16,True,'center');label(c,cx,y-23,r['interval_hours'],15,align='center')
 label(c,x+w/2,15,'Time after cessation (h)',16,align='center')
c.saveState();c.translate(19,183);c.rotate(90);label(c,0,0,'Restart-first records (%)',16,align='center');c.restoreState()
c.save();finish(W/'timing.pdf',UP/'Figure_2.pdf')

c=canvas.Canvas(str(W/'ablation.pdf'),pagesize=(960,390),initialFontName='ArialEmbedded')
pts=rows('source_ablation_performance_points.csv');cis=rows('source_ablation_hospital_cluster_CI.csv')
for i,(metric,title,mx,ticks) in enumerate([
 ('mean_predicted_risk','Mean predicted risk',1.05,[0,.2,.4,.6,.8,1]),
 ('brier','Brier score',.65,[0,.1,.2,.3,.4,.5,.6]),
 ('auroc','AUROC',.65,[0,.1,.2,.3,.4,.5,.6])]):
 x=54+i*316;y=92;w=261;h=240
 label(c,x+w/2,364,title,18,True,'center');axes(c,x,y,w,h,mx,ticks)
 for j,r in enumerate(pts):
  ci=next(t for t in cis if t['model']==r['model'] and t['metric']==metric);v=float(r[metric]);cx=x+(j+.5)*w/2
  c.setFillColor(HexColor('#a94e42' if j==0 else '#3e7b59'));c.rect(cx-38,y,76,h*v/mx,fill=1,stroke=0)
  lo=y+h*float(ci['ci_low'])/mx;hi=y+h*float(ci['ci_high'])/mx
  c.setStrokeColor(black);c.setLineWidth(1.3);c.line(cx,lo,cx,hi);c.line(cx-6,lo,cx+6,lo);c.line(cx-6,hi,cx+6,hi)
  label(c,cx,hi+11,f'{v:.3f}',16,True,'center')
  label(c,cx,y-24,'Full source' if j==0 else 'Dose/agent',15,align='center')
  if j:label(c,cx,y-43,'ablation',15,align='center')
 if metric=='mean_predicted_risk':
  yy=y+h*float(pts[0]['prevalence'])/mx;c.setDash(4,3);c.line(x,yy,x+w,yy);c.setDash()
label(c,480,14,'Dashed line: observed event rate 34.89%',15,align='center')
c.save();finish(W/'ablation.pdf',UP/'Figure_5.pdf')

impact=pd.read_csv(args.posthoc/'endpoint_correction_impact_compact.csv').set_index('database')
c=canvas.Canvas(str(W/'flow.pdf'),pagesize=(960,490),initialFontName='ArialEmbedded')
def box(x,y,w,h,lines):
 c.setFillColor(HexColor('#e8f0f7'));c.setStrokeColor(HexColor('#45647d'));c.roundRect(x,y,w,h,7,fill=1,stroke=1)
 for j,line in enumerate(lines):label(c,x+w/2,y+h/2+(len(lines)-1)*11-j*22,line,18,align='center')
for i,db in enumerate(['MIMIC-IV','eICU']):
 x=20+i*480;r=impact.loc[db];label(c,x+225,458,db,24,True,'center');box(x+93,345,270,68,[f'{int(r.fixed_candidates):,} candidates'])
 box(x+8,209,260,85,[f'{int(r.primary_analysis):,} primary','analysis records']);box(x+282,209,175,85,[f'{int(r.excluded_or_indeterminate):,}','excluded or','indeterminate'])
 c.line(x+185,345,x+138,294);c.line(x+275,345,x+368,294)
 if i==0:
  counts=pd.read_csv(args.mimic_results/'cohort_counts.csv');counts=counts[counts.analysis=='primary'].groupby('corrected_analysis_split').n.sum()
  partitions=[('Development',f'{int(counts["development"]):,}'),('Temporal validation',f'{int(counts["temporal_validation"]):,}')]
 else:
  counts=pd.read_csv(args.eicu_results/'cohort_counts.csv');counts=counts[counts.variant=='primary_traceable'].groupby('partition').included.sum()
  partitions=[('Adaptation',f'{int(counts["adaptation_development"]):,}'),('Recalibration',f'{int(counts["recalibration"]):,}'),('Locked\nevaluation',f'{int(counts["locked_evaluation"]):,}')]
 width=210 if i==0 else 145
 for j,(name,n) in enumerate(partitions):
  bx=x+8+j*(width+8);box(bx,54,width,76,[*name.split('\n'),n]);c.line(x+138,209,x+138,169);c.line(x+138,169,bx+width/2,169);c.line(bx+width/2,169,bx+width/2,130)
c.save();finish(W/'flow.pdf',UP/'Figure_1.pdf')
mc=pd.read_csv(args.mimic_results/'mimic_primary_main_patient_bootstrap_ci.csv');ec=pd.read_csv(args.eicu_results/'primary_traceable_hospital_CI.csv')
records=[('MIMIC temporal dynamic logistic',mc[(mc.model=='dynamic_multinomial_logistic')&(mc.outcome=='restart_first')&(mc.metric=='auroc')].iloc[0]),('eICU source-only',ec[(ec.model=='corrected_primary_source_raw')&(ec.metric=='auroc')].iloc[0]),('eICU target-native',ec[(ec.model=='target_native_refit')&(ec.metric=='auroc')].iloc[0])]
c=canvas.Canvas(str(W/'auroc.pdf'),pagesize=(960,390),initialFontName='ArialEmbedded');x=390;y=76;w=500;h=240
for tick in [.45,.50,.55,.60,.65,.70,.75]:
 xx=x+(tick-.42)/.36*w;c.setStrokeColor(HexColor('#dddddd'));c.line(xx,y,xx,y+h);label(c,xx,y-25,f'{tick:.2f}',17,align='center')
for j,(name,r) in enumerate(records):
 yy=y+200-j*80;label(c,x-20,yy-6,name,19,align='right');lo=x+(r.ci_low-.42)/.36*w;hi=x+(r.ci_high-.42)/.36*w;xx=x+(r.estimate-.42)/.36*w;c.setStrokeColor(HexColor('#27577f'));c.setFillColor(HexColor('#27577f'));c.setLineWidth(2);c.line(lo,yy,hi,yy);c.line(lo,yy-7,lo,yy+7);c.line(hi,yy-7,hi,yy+7);c.circle(xx,yy,5,fill=1);label(c,xx,yy+19,f'{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]',16,align='center')
label(c,640,20,'AUROC (95% cluster-bootstrap CI)',19,align='center');c.save();finish(W/'auroc.pdf',UP/'Figure_3.pdf')
finish(args.display/'Figure_3_calibration_and_risk_distribution.pdf',UP/'Figure_4.pdf')
finish(args.posthoc/'Figure_4_eICU_hospital_heterogeneity.pdf',UP/'Figure_6.pdf',(0,0,606.016279,650))
print('Generated six current submission figures from aggregate results.')
