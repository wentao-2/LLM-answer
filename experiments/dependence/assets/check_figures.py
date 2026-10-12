"""Check actual figure values, vector text, embedded fonts and final dimensions.

Pass --visual-inspection-confirmed only after inspecting both rendered previews.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from common import load,sha,now,atomic_json


def same(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=0,abs_tol=1e-12)),(a,b)


def check(visual):
    assert load(ROOT/'audit/AGGREGATES.json')['state']=='passed'
    analysis=load(ROOT/'analysis/RESULTS.json')
    digest=sha(ROOT/'analysis/RESULTS.json')
    source_digest=sha(ROOT/'assets/plot_results.py')
    figures=[]
    points=0
    for name in ('partial_dependence','source_intervention'):
        provenance=load(ROOT/'figures'/f'{name}.data.json')
        assert provenance['analysis_sha256']==digest
        rendering=provenance['rendering']
        assert rendering['source_sha256']==source_digest
        assert rendering['width_inches']==6.3 and rendering['minimum_font_points']>=9.5
        assert rendering['text_within_canvas'] is True and rendering['font']=='Times New Roman'
        module='partial_dependence' if name=='partial_dependence' else 'source_intervention'
        for point in provenance['points']:
            match=[row for row in analysis['pair_summaries'] if row['module']==module
                   and all(row[key]==point[key] for key in ('history_arm','branch'))
                   and (row['model']==point['model'] and row['rho']==point['rho']
                        if module=='partial_dependence' else row['condition']==point['condition'])]
            assert len(match)==1
            row=match[0]
            selected=row['common_all'] if row['common_all']['effective_weight']['n_families']==6 else row
            mode='common complete parents' if selected is row['common_all'] else 'finite valid pairs'
            assert point['selection']==mode and point['n']==selected['effective_weight']['n']
            assert point['planned']==row['n_planned']
            same(point['estimate'],selected['effective_weight']['equal_family_mean'])
            assert point['family_range']==selected['effective_weight']['family_range']
            if module=='partial_dependence':
                members=[p for p in analysis['pair_rows'] if p['module']==module
                         and all(p[key]==point[key] for key in ('model','history_arm','branch','rho'))
                         and p['effective_weight'] is not None
                         and (mode!='common complete parents' or p['common_parent_all_cells'])]
                families=defaultdict(list)
                for p in members:families[p['family_id']].append(p['normative_weight'])
                target=math.fsum(math.fsum(v)/len(v) for v in families.values())/6 if len(families)==6 else None
            else:target=0 if point['branch']=='copy' else 1
            same(point['normative'],target)
            points+=1
        svg=ET.parse(ROOT/'figures'/f'{name}.svg')
        tags=[node.tag.rsplit('}',1)[-1] for node in svg.iter()]
        assert tags.count('text')>10 and 'image' not in tags
        reader=PdfReader(ROOT/'figures'/f'{name}.pdf')
        assert len(reader.pages)==1
        page=reader.pages[0]
        same(float(page.mediabox.width),6.3*72)
        font_info=[]
        for resource in page['/Resources']['/Font'].values():
            font=resource.get_object()
            assert 'TimesNewRoman' in str(font['/BaseFont'])
            concrete=font['/DescendantFonts'][0].get_object() if '/DescendantFonts' in font else font
            descriptor=concrete['/FontDescriptor'].get_object()
            assert '/FontFile2' in descriptor or '/FontFile' in descriptor or '/FontFile3' in descriptor
            font_info.append(dict(name=str(font['/BaseFont']),embedded=True))
        assert font_info
        figures.append(dict(name=name,width_points=float(page.mediabox.width),height_points=float(page.mediabox.height),
            live_svg_text=tags.count('text'),raster_images=0,fonts=font_info,
            outputs={ext:sha(ROOT/'figures'/f'{name}.{ext}') for ext in ('pdf','svg','png')}))
    report=dict(state='passed' if visual else 'awaiting_visual_inspection',checked_at=now(),
        analysis_sha256=digest,plot_source_sha256=source_digest,points_checked=points,
        visual_inspection_confirmed=visual,figures=figures,
        interval='range of six family means, not a confidence interval')
    atomic_json(ROOT/'audit/FIGURES.json',report)
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--visual-inspection-confirmed',action='store_true')
    args=parser.parse_args()
    check(args.visual_inspection_confirmed)
