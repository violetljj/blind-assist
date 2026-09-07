"""Inspect and prepare the Poly Haven modular seating FBX without Unreal."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib


class Node:
    def __init__(self, name, properties, raw, children, sentinel=False):
        self.name, self.properties, self.raw, self.children = name, properties, raw, children
        self.sentinel = sentinel


def parse(data):
    version = struct.unpack_from("<I", data, 23)[0]
    if version != 7400:
        raise ValueError("Only observed FBX 7400 format supported")

    def node(offset):
        end, count, size, name_size = struct.unpack_from("<IIIB", data, offset)
        if not end:
            return None, offset + 13
        cursor = offset + 13
        name = data[cursor:cursor + name_size].decode()
        cursor += name_size
        raw = data[cursor:cursor + size]
        prop_cursor, properties = cursor, []
        for _ in range(count):
            kind = chr(data[prop_cursor]); prop_cursor += 1
            if kind in "YCFDI L".replace(" ", ""):
                fmt = {"Y": "h", "C": "?", "F": "f", "D": "d", "I": "i", "L": "q"}[kind]
                value = struct.unpack_from("<" + fmt, data, prop_cursor)[0]
                prop_cursor += struct.calcsize(fmt)
            elif kind in "SR":
                length = struct.unpack_from("<I", data, prop_cursor)[0]; prop_cursor += 4
                value = data[prop_cursor:prop_cursor + length]; prop_cursor += length
                if kind == "S": value = value.decode(errors="replace")
            elif kind in "fdlibc":
                length, encoding, payload_size = struct.unpack_from("<III", data, prop_cursor); prop_cursor += 12
                value = data[prop_cursor:prop_cursor + payload_size]; prop_cursor += payload_size
                if encoding == 1: value = zlib.decompress(value)
                fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b", "c": "b"}[kind]
                value = list(struct.unpack("<" + str(length) + fmt, value))
            else:
                raise ValueError(f"Unsupported property {kind}")
            properties.append(value)
        children, cursor = [], cursor + size
        sentinel = cursor < end
        while cursor < end:
            child, cursor = node(cursor)
            if child is None: break
            children.append(child)
        if cursor != end:
            raise ValueError("Invalid FBX node boundary")
        return Node(name, properties, raw, children, sentinel), end

    nodes, offset = [], 27
    while True:
        item, offset = node(offset)
        if item is None: break
        nodes.append(item)
    return nodes, data[offset:]


def inspect(nodes):
    objects = next(node for node in nodes if node.name == "Objects")
    result = []
    for node in objects.children:
        if node.name not in ("Geometry", "Model", "Material"): continue
        entry = {"kind": node.name, "id": node.properties[0], "name": node.properties[1]}
        for child in node.children:
            if child.name == "Vertices":
                values = child.properties[0]
                entry["vertex_count"] = len(values) // 3
                entry["bounds"] = [[min(values[axis::3]), max(values[axis::3])] for axis in range(3)]
            if child.name == "Properties70":
                entry["transform"] = {row.properties[0]: row.properties[4:] for row in child.children if row.properties[0].startswith("Lcl")}
        result.append(entry)
    return result


def serialize(nodes, header, tail):
    def encode(node, offset):
        name = node.name.encode()
        cursor = offset + 13 + len(name) + len(node.raw)
        body = bytearray(node.raw)
        for child in node.children:
            payload = encode(child, cursor)
            body.extend(payload); cursor += len(payload)
        if node.sentinel:
            body.extend(bytes(13)); cursor += 13
        return struct.pack("<IIIB", cursor, len(node.properties), len(node.raw), len(name)) + name + body
    result = bytearray(header)
    for node in nodes:
        result.extend(encode(node, len(result)))
    result.extend(bytes(13))
    # Blender footer: magic, alignment padding, version/reserved/footer magic.
    # The original terminal block is preserved with its alignment regenerated.
    footer_body = tail[-140:]
    result.extend(tail[:16])
    result.extend(bytes((-len(result)) % 16 + 16))
    result.extend(footer_body)
    return bytes(result)


def prepare(source, output, assembled=False):
    original = source.read_bytes()
    if hashlib.md5(original).hexdigest() != "a46a21ac975563abb2a11e977400b7a6":
        raise ValueError("Unexpected asset revision; inspect before choosing removal nodes")
    nodes, tail = parse(original)
    objects = next(node for node in nodes if node.name == "Objects")
    connections = next(node for node in nodes if node.name == "Connections")
    names = {"connector_30", "connector_45", "connector_60", "connector_90"}
    if assembled:
        names |= {"seat_bench", "suspended_support_01", "suspended_support_02"}
    model_ids = {node.properties[0] for node in objects.children
                 if node.name == "Model" and node.properties[1].split("\x00")[0] in names}
    geometry_ids = {node.properties[1] for node in connections.children
                    if node.properties[0] == "OO" and node.properties[2] in model_ids}
    # Other incoming connections are materials: remove only connected geometry.
    geometry_ids &= {node.properties[0] for node in objects.children if node.name == "Geometry"}
    if len(model_ids) != len(names) or len(geometry_ids) != len(names):
        raise ValueError("Expected exact selected spare mesh/geometry pairs")
    removed = model_ids | geometry_ids
    before = inspect(nodes)
    objects.children = [node for node in objects.children if node.properties[0] not in removed]
    connections.children = [node for node in connections.children if not any(value in removed for value in node.properties[1:3])]
    definitions = next(node for node in nodes if node.name == "Definitions")
    for node in definitions.children:
        if node.name == "Count":
            node.properties[0] -= len(removed)
            node.raw = b"I" + struct.pack("<i", node.properties[0])
        elif node.name == "ObjectType" and node.properties[0] in ("Geometry", "Model"):
            count = next(child for child in node.children if child.name == "Count")
            count.properties[0] -= len(names)
            count.raw = b"I" + struct.pack("<i", count.properties[0])
    assembly = {}
    if assembled:
        # FBX global coordinates are centimetres with Y up. These translations
        # replace the downloaded exploded display layout with a single bench.
        translations = {"crossbar": [0., 16.5, 0.], "legs_double": [88., 0., 0.],
                        "legs_single": [-88., 0., 0.], "back_support_r": [-84., 45., 0.],
                        "back_support_l": [84., 45., 0.], "arm_rest_01": [-88., 45., 0.],
                        "arm_rest_02": [88., 45., 0.], "seat": [0., 45., 0.],
                        "seat_back": [0., 44.5, 0.]}
        for node in objects.children:
            if node.name != "Model": continue
            name = node.properties[1].split("\x00")[0]
            props = next(child for child in node.children if child.name == "Properties70")
            prop = next(child for child in props.children if child.properties[0] == "Lcl Translation")
            assert len(prop.properties) == 7 and all(prop.raw[-27 + j * 9] == ord("D") for j in range(3))
            prop.properties[4:] = translations[name]
            prop.raw = prop.raw[:-27] + b"".join(b"D" + struct.pack("<d", value) for value in translations[name])
        geometries = {node.properties[0]: node for node in objects.children if node.name == "Geometry"}
        bounds = {}
        for node in objects.children:
            if node.name != "Model": continue
            name = node.properties[1].split("\x00")[0]
            geom_id = next(row.properties[1] for row in connections.children
                           if row.properties[2] == node.properties[0] and row.properties[1] in geometries)
            vertices = next(child.properties[0] for child in geometries[geom_id].children if child.name == "Vertices")
            props = next(child for child in node.children if child.name == "Properties70")
            trans = {child.properties[0]: child.properties[4:] for child in props.children if child.properties[0].startswith("Lcl")}
            rx, ry, rz = trans["Lcl Rotation"]
            assert abs(ry) < 1e-6 and abs(rz) < 1e-6
            c, s = math.cos(math.radians(rx)), math.sin(math.radians(rx))
            points = []
            for index in range(0, len(vertices), 3):
                x, y, z = [vertices[index + axis] * trans["Lcl Scaling"][axis] for axis in range(3)]
                points.append([x + translations[name][0], c * y - s * z + translations[name][1],
                               s * y + c * z + translations[name][2]])
            bounds[name] = [[min(point[axis] for point in points), max(point[axis] for point in points)] for axis in range(3)]
        edges = [[left, right] for left in bounds for right in bounds if left < right
                 and all(max(bounds[left][axis][0], bounds[right][axis][0]) <=
                         min(bounds[left][axis][1], bounds[right][axis][1]) + .01 for axis in range(3))]
        connected = {"seat"}
        for _ in bounds:
            for left, right in edges:
                if left in connected or right in connected: connected.update((left, right))
        assert connected == set(bounds), f"Disconnected assembly AABBs: {set(bounds) - connected}"
        assert all(abs(bounds[name][1][0]) < .01 for name in ("legs_double", "legs_single"))
        assembly = {"translations_fbx_cm_y_up": translations, "bounds_fbx_cm_y_up": bounds,
                    "aabb_connection_edges": edges, "grounded_legs": "PASS",
                    "component_aabb_connectivity": "PASS_PROXY_ONLY_REQUIRES_RENDER_INSPECTION"}
    cleaned = serialize(nodes, original[:27], tail)
    reparsed, _ = parse(cleaned)
    after = inspect(reparsed)
    assert len([row for row in after if row["kind"] == "Model"]) == 16 - len(names)
    # Preserved property payloads include every UV, normal, polygon/material index
    # array, texture connection, and kept model transform. Compare recursively.
    def flatten(items):
        return [(node.name, node.raw, flatten(node.children)) for node in items]
    assert flatten(nodes) == flatten(reparsed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(cleaned)
    receipt = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(original).hexdigest(),
               "output": str(output.resolve()), "output_sha256": hashlib.sha256(cleaned).hexdigest(),
               "removed": [row for row in before if row["id"] in removed],
               "retained": after, "property_payload_preservation": "PASS",
               "assembly": assembly,
               "reason": "Assemble exploded seating via model translations; remove extension and spare connectors/supports." if assembled else "Four angular connectors are spare parts displayed at 75/95cm above adjacent 45cm seat, not attached seating.",
               "source_license": "https://polyhaven.com/license", "license": "CC0"}
    output.with_suffix(".preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"output": str(output.resolve()), "bytes": len(cleaned), "removed_models": sorted(names),
            "retained_models": [row["name"].split("\x00")[0] for row in after if row["kind"] == "Model"],
            "retained_materials": [row["name"].split("\x00")[0] for row in after if row["kind"] == "Material"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts.local/unreal/sample-materials-v2/modular_street_seating/modular_street_seating_4k.fbx")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assembled", action="store_true", help="Assemble the single backed bench from exploded model parts")
    args = parser.parse_args()
    if args.output:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts.local"
        if not args.output.resolve().is_relative_to(artifact_root.resolve()):
            raise ValueError("Prepared assets must stay under artifacts.local")
        if args.output.resolve() == args.input.resolve():
            raise ValueError("Keep the original downloaded asset intact")
        print(json.dumps(prepare(args.input, args.output, args.assembled), indent=2))
    else:
        nodes, _ = parse(args.input.read_bytes())
        print(json.dumps(inspect(nodes), indent=2))


if __name__ == "__main__":
    main()
