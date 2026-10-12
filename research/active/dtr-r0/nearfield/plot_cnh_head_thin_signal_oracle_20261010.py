"""Standalone scientific figure from the sealed expectation-only oracle CSVs."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np


COLORS=('#0072B2','#D55E00','#009E73','#CC79A7')


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def draw(out):
    began=time.monotonic()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180,'font.family':'DejaVu Sans'})
    single=read_csv(out/'single_frame.csv');temporal=read_csv(out/'accumulation.csv')
    plan=json.loads((out/'PLAN.json').read_text(encoding='utf-8-sig'))
    figures=out/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    rhos=(.19,.57)
    for column,rho in enumerate(rhos):
        ax=axes[0,column]
        for thick,color in zip(plan['thickness_m'],COLORS):
            values=[r for r in single if abs(float(r['rho'])-rho)<1e-9 and abs(float(r['thickness_m'])-thick)<1e-9]
            frames=range(15,-1,-1);x=[];median=[];minimum=[];maximum=[]
            for frame in frames:
                rows=[r for r in values if int(r['frame'])==frame]
                v=np.array([float(r['target_best_bin_snr']) for r in rows])
                x.append(float(rows[0]['front_distance_m']));median.append(float(np.median(v)))
                minimum.append(float(v.min()));maximum.append(float(v.max()))
            ax.plot(x,median,color=color,lw=2,label=f'{100*thick:g} cm')
            ax.fill_between(x,minimum,maximum,color=color,alpha=.10,lw=0)
        ax.set_title(f'Single frame, reflectivity {rho:g}')
        ax.set_xlabel('Forward distance to target front (m)')
        ax.set_ylabel('Best target-bin SNR')
        ax.grid(alpha=.2);ax.legend(title='Square cross section',frameon=False)
        ax=axes[1,column]
        styles=(('moving_fixed_bin_snr','Fixed native bin, moving','-','#0072B2'),
                ('privileged_moving_peak_snr','Target-aligned best bins, moving','--','#D55E00'),
                ('static_repeat_peak_snr','Identical static repeats',':','#009E73'))
        values=[r for r in temporal if abs(float(r['rho'])-rho)<1e-9 and abs(float(r['thickness_m'])-.017)<1e-9 and int(r['end_frame'])==13]
        for field,label,linestyle,color in styles:
            median=[];minimum=[];maximum=[]
            for k in plan['windows']:
                v=np.array([float(r[field]) for r in values if int(r['requested_frames'])==k])
                median.append(float(np.median(v)));minimum.append(float(v.min()));maximum.append(float(v.max()))
            ax.plot(plan['windows'],median,color=color,ls=linestyle,lw=2,marker='o',label=label)
            ax.fill_between(plan['windows'],minimum,maximum,color=color,alpha=.08,lw=0)
        ax.set_title(f'1.7 cm bar, current front 0.97 m, reflectivity {rho:g}')
        ax.set_xlabel('Independent frames / repeated exposures')
        ax.set_ylabel('Single-bin sum SNR')
        ax.set_xticks(plan['windows']);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('HEAD horizontal bars: expected return relative to simulated shot noise',fontsize=15)
    fig.supxlabel('Line: median; band: min-max over 16 consumed fixture templates (not a confidence interval).\n'
                  'Oracle target/background knowledge; no detector false-positive or hardware performance claim.',fontsize=10)
    png=figures/'thin_bar_signal_and_accumulation.png';pdf=figures/'thin_bar_signal_and_accumulation.pdf'
    if png.exists() or pdf.exists():raise FileExistsError('Retain prior scientific figures')
    fig.savefig(png);fig.savefig(pdf);plt.close(fig)
    receipt=dict(status='COMPLETE',seconds=time.monotonic()-began,
        source_tables=['single_frame.csv','accumulation.csv'],outputs=[str(png),str(pdf)],
        panel_contract='top: four cross sections at fixed x/y/span/background over original approach frames; bottom: 1.7cm frame13 only, single-bin sum estimators; reflectivity.19/.57',
        distribution='Median/minmax over16 fixtures, repeated template source factors; not independent confidence interval')
    with (out/'plot_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print('ORACLE_PLOT',png,'seconds',round(time.monotonic()-began,2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    draw(parser.parse_args().output)
