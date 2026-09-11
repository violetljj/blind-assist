"""Focused CPU acceptance fixtures; --output is a fresh evidence directory."""
import argparse
import json
from pathlib import Path
from PIL import Image
from compact_rgb_view import build, sha, write_json


def check(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    image=Image.new('RGBA',(2,2))
    image.putdata([(17,23,91,0),(3,4,5,1),(100,50,2,128),(4,6,8,255)])
    source=output/'original.png';image.save(source);image.close()
    original=source.read_bytes()
    row=dict(frame_id='rgba-alpha-edge',source=dict(path=str(source),sha256=sha(source)),output='model/rgb/example.webp')
    manifest=output/'input.json';write_json(manifest,dict(schema='compact-rgb-view-input-v1',images=[row]))
    result=build(manifest,output/'view')
    assert result['exact_pixels_all'] and result['frames']==1 and source.read_bytes()==original
    # Refuse an existing view, including a differing file; no overwrite or silent resume.
    prior=(output/'view/model/rgb/example.webp').read_bytes()
    try:build(manifest,output/'view')
    except FileExistsError:pass
    else:raise AssertionError('Existing output accepted')
    assert (output/'view/model/rgb/example.webp').read_bytes()==prior
    bad=output/'bad-sha.json';write_json(bad,dict(schema='compact-rgb-view-input-v1',images=[dict(row,source=dict(path=str(source),sha256='0'*64))]))
    try:build(bad,output/'bad-sha-output')
    except ValueError:pass
    else:raise AssertionError('Bad source hash accepted')
    assert (output/'bad-sha-output/failure.json').exists() and source.read_bytes()==original
    for name,images in [('escape',[dict(row,output='../escaped.webp')]),
                        ('collision',[dict(row,output='A.webp'),dict(row,frame_id='other',output='a.webp')])]:
        path=output/f'{name}.json';write_json(path,dict(schema='compact-rgb-view-input-v1',images=images))
        try:build(path,output/f'{name}-output')
        except ValueError:pass
        else:raise AssertionError(f'{name} accepted')
        assert not (output/f'{name}-output').exists()
    receipt=dict(status='PASS',checks=['RGBA including invisible RGB under alpha0 exact',
        'existing output immutable','wrong original SHA rejected with failure evidence',
        'path traversal rejected','case-fold output collision rejected'],code_sha256=sha(__file__),
        builder_sha256=sha(Path(__file__).with_name('compact_rgb_view.py')),source_unchanged=source.read_bytes()==original)
    write_json(output/'tests.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    check(p.parse_args().output)
