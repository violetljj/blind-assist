"""Make an ONNX graph acceptable to qairt-converter when the exporter shares Constant nodes across layers
("Duplicate node name ... Constant_output_0"): turn every Constant node into an initializer with a unique name per consumer,
and give every node a unique name. Numerics are unchanged. Usage: onnx_unique_constants_dev_20261117.py <in.onnx> <out.onnx>
"""
import sys

import onnx
from onnx import numpy_helper


def main(src, dst):
    m = onnx.load(src)
    g = m.graph
    consts = {}
    keep = []
    for n in g.node:
        if n.op_type == 'Constant' and len(n.output) == 1:
            attr = {a.name: a for a in n.attribute}
            if 'value' in attr:
                consts[n.output[0]] = numpy_helper.to_array(attr['value'].t)
                continue
        keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    k = 0
    for n in g.node:
        for i, name in enumerate(n.input):
            if name in consts:
                new = f'{name}__c{k}'; k += 1
                g.initializer.append(numpy_helper.from_array(consts[name], new))
                n.input[i] = new
    for i, n in enumerate(g.node):
        n.name = f'n{i:05d}_{n.op_type}'
    for o in g.output:
        if o.name in consts:
            raise SystemExit('graph output is a constant: ' + o.name)
    onnx.save(m, dst, save_as_external_data=False)
    print('constants moved', len(consts), 'uses', k, 'nodes', len(g.node))


if __name__ == '__main__':
    main(*sys.argv[1:3])
