"""Check page contract, text presence, geometry, and missing glyphs in the LBD PDF."""
import json
from pathlib import Path
import sys
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]

def main():
    pdfpath = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'output/pdf/main.pdf'
    with pdfplumber.open(pdfpath) as doc:
        texts=[p.extract_text(x_tolerance=1) or '' for p in doc.pages]
        compact=[''.join(t.split()).upper() for t in texts]
        checks={
            'three_pages':len(doc.pages)==3,
            'a4':all(abs(p.width-595.28)<2 and abs(p.height-841.89)<2 for p in doc.pages),
            'references_only_start_on_page_3': len(texts)==3 and 'REFERENCES' in texts[2] and all('REFERENCES' not in t for t in texts[:2]),
            'scientific_conclusion_on_page_2':len(texts)>1 and 'DISCUSSIONANDCONCLUSION' in compact[1],
            'p1_p2_semip3_present':all(s in ''.join(texts[:2]) for s in ['P1:', 'P2:', 'Semi-P3:']),
            'no_unresolved_citations':all('[?]' not in t and '??' not in t for t in texts),
            'no_replacement_glyphs':all('\ufffd' not in t and '(cid:' not in t for t in texts),
            'under_10mb':pdfpath.stat().st_size < 10_000_000,
        }
        pages=[]
        for i,p in enumerate(doc.pages,1):
            chars=[c for c in p.chars if c.get('text','').strip()]
            outside=[c['text'] for c in chars if c['x0']<0 or c['x1']>p.width or c['top']<0 or c['bottom']>p.height]
            pages.append({'page':i,'characters':len(chars),'outside_page':outside,'min_x':min(c['x0'] for c in chars),'max_x':max(c['x1'] for c in chars),'max_bottom':max(c['bottom'] for c in chars)})
        checks['all_characters_inside_page']=not any(p['outside_page'] for p in pages)
        report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'pages':pages,'author_metadata_pending':'AUTHORINFORMATIONPENDING' in ''.join(compact),'visual_review_required':True}
        (ROOT/'paper/lbd2026/pdf_qa.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2))
        for i,t in enumerate(texts,1):
            print(f'PAGE {i}: starts {t[:85]!r}; ends {t[-180:]!r}')
        if report['status']!='PASS': raise SystemExit(1)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
