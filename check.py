"""The community library's automatic checks (R17, R20, KTD10): one module for the app's upload check and the library
repository's pull-request workflow, which vendors this file. It imports only the standard library, numpy, trimesh and
manifold3d, never the app.

    python check.py piece piece.stl metadata.json
    python check.py train car.glb metadata.json
    python check.py furniture table.json metadata.json

prints what was checked and exits 0, or prints why the item is refused and exits 1. In code, `check(kind, data,
metadata)` returns a summary or raises `Refused` with a message for the contributor.

Metadata, per kind (an `id`, when given, is a library id: `community_` then lowercase letters, digits and `_`):
- piece: `name` and `ends`, the two marked ends `{x, y, z, heading}` in the STL's frame (README "Community library"); a
  piece adapted from a CC BY model also has that model's `source` `{url, authors}` (a piece is always CC BY 4.0).
- train: `name`, `year`, `power`, `gauge`, `car` (the index's car measurements) and `licence`; a model adapted from
  another CC BY model also has `source` `{url, authors}` and keeps that model's CC BY licence (R17).
- furniture: `name`; the file is the `FurnitureType` JSON (src/domain/furniture.ts) and its `family` is the `id`.

A piece is checked in this order, cheapest first: the byte length against the triangle count (binary STL only), the
mesh (closed, consistently wound, positive volume, one part), its size (10 mm up to what the A1 mini prints: 166 mm
across in plan, 180 mm high), then its marked ends against its end faces. The connector grid: a marked end sits where a
track connector's centre does, in the middle of its end face (a connector's stud columns are symmetric about the
centreline; the face is at least two studs wide), at rail height (the highest point of the piece within a connector's
half-width of it), facing straight out of the face.
An end face is everything in the end's plane facing out of it, taken together: a connector's tongue and socket cut it
into separate pieces, and its width is their combined span across the track. The app's ends editor
(src/contribute/ends.ts) snaps a click to that point by the same rules.
"""
from __future__ import annotations

import json
import math
import re
import struct
import sys
from collections.abc import Callable

import numpy as np

PIECE_BYTES = 20_000_000
PIECE_TRIANGLES = 200_000
PIECE_SMALLEST = 10.0
# The A1 mini's bed, the smallest printer Plateway prepares for (mm).
BED = 180.0
# The widest plan footprint its arrange places: the bed less a 7 mm margin each side (cad/service/plates.py `_shelves`;
# tests/test_library_check.py holds them together). A piece prints as uploaded, never turned to fit.
PIECE_PLAN = BED - 2 * 7.0
# A train car's model budget: the built-in catalogue's (tests/test_trains.py, scripts/trains/validate.py).
TRAIN_BYTES = 250_000
TRAIN_TRIANGLES = 12_000
TRAIN_WIDEST = 80.0
TRAIN_FIT = 3.0
TRAIN_LOWEST = -12.0
FURNITURE_BYTES = 512_000
FURNITURE_LARGEST = 10_000.0
TEXT_MAX = 120
OUTLINE_MAX = 40
LIBRARY_ID = re.compile(r'community_[a-z0-9_]{1,60}')
ORIGINAL_LICENCE = 'CC-BY-4.0'
ADAPTED_LICENCES = ('CC-BY-2.0', 'CC-BY-2.5', 'CC-BY-3.0', 'CC-BY-4.0')
POWER = ('none', '4.5V', '12V', '9V', 'RC', 'PF', 'PU', 'monorail')
CAR_KINDS = ('loco', 'tender', 'wagon', 'coach', 'caboose', 'other')

# Marked ends (src/domain/pieces/families/community.ts and src/contribute/ends.ts).
END_FACE_COSINE = math.cos(math.radians(2))
END_FACE_OFFSET = 0.3
# Two studs: an end face is a connector's width, not one facet of a curved side wall.
END_FACE_WIDTH = 16.0
END_TOLERANCE = 0.5
# Half a LEGO track's width (cad/profiles/lego-compatible-v1.json `bodyWidth`): the rails at an end lie within it.
RAIL_REACH = 32.0
LEVEL = 0.5
SHORTEST = 16.0

GLB_MAGIC, GLB_JSON, GLB_BIN = 0x46546C67, 0x4E4F534A, 0x004E4942
FLOAT, UBYTE, USHORT, UINT = 5126, 5121, 5123, 5125
COMPONENT_BYTES = {5120: 1, UBYTE: 1, 5122: 2, USHORT: 2, UINT: 4, FLOAT: 4}
TYPE_SIZES = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


class Refused(ValueError):
    """The item can't join the library; the message tells the contributor why."""


def is_finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def required_text(value: object, label: str, limit: int = TEXT_MAX) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise Refused(f'{label} must be 1–{limit} characters.')
    return value.strip()


def _common(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        raise Refused('The metadata must be an object.')
    required_text(metadata.get('name'), 'The name')
    if 'id' in metadata and (not isinstance(metadata['id'], str) or not LIBRARY_ID.fullmatch(metadata['id'])):
        raise Refused('The id must be "community_" then lowercase letters, digits and _.')
    return metadata


def _mm(value: float) -> str:
    return f'{value:,.1f}'.removesuffix('.0') + ' mm'


# Pieces ---------------------------------------------------------------------------------------------------------


def stl_triangles(data: bytes) -> np.ndarray:
    """The triangles (n, 3, 3) of a binary STL, refused before parsing when its size or byte length is wrong."""
    if len(data) > PIECE_BYTES:
        raise Refused(f'The STL is {len(data) / 1e6:.1f} MB; the limit is {PIECE_BYTES / 1e6:.0f} MB.')
    count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else None
    if count is None or len(data) != 84 + 50 * count:
        if data[:5].lower() == b'solid' and b'facet' in data[:1024]:
            raise Refused('Only binary STL is accepted, and this is ASCII STL. Export it again as binary STL.')
        raise Refused("The STL's length doesn't match its triangle count: it is damaged or not a binary STL.")
    if count == 0:
        raise Refused('The STL has no triangles.')
    if count > PIECE_TRIANGLES:
        raise Refused(f'The STL has {count:,} triangles; the limit is {PIECE_TRIANGLES:,}. Simplify the mesh.')
    records = np.frombuffer(data, np.dtype([('normal', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')]), count, 84)
    triangles = records['v'].astype(np.float64)
    if not np.isfinite(triangles).all():
        raise Refused('The STL has coordinates that are not numbers.')
    return triangles


def _solid(triangles: np.ndarray):
    """The triangles as one closed, consistently wound, positive-volume mesh with merged vertices."""
    import manifold3d
    import trimesh

    mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3), faces=np.arange(len(triangles) * 3).reshape(-1, 3),
                           process=True)
    if not mesh.is_watertight:
        raise Refused('The mesh has holes: it must be a closed solid. Repair it in your CAD program or slicer.')
    if not mesh.is_winding_consistent:
        raise Refused('Some faces of the mesh point inwards. Repair it in your CAD program or slicer.')
    if mesh.volume <= 0:
        raise Refused('The mesh is inside out (its volume is negative). Flip its normals and export it again.')
    solid = manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                                tri_verts=np.asarray(mesh.faces, np.uint32)))
    if solid.status() != manifold3d.Error.NoError or solid.is_empty():
        raise Refused('The mesh is not one clean solid: it has stray, doubled or self-touching faces.')
    parts = len(solid.decompose())
    if parts != 1:
        raise Refused(f'The mesh is {parts} separate parts; a piece is one part.')
    return mesh


def _size_problem(size: np.ndarray) -> None:
    largest = float(size.max())
    if largest < PIECE_SMALLEST:
        raise Refused(f'The piece is only {_mm(largest)} across: was it exported in inches or metres? '
                      'Plateway reads STL in millimetres.')
    if (size > BED + 1e-6).any():
        raise Refused(f'The piece is {" × ".join(_mm(float(value)) for value in size)}: larger than the A1 mini bed '
                      f'({BED:.0f} mm). Was it exported in another unit?')
    if (size[:2] > PIECE_PLAN + 1e-6).any():
        raise Refused(f'The piece is {" × ".join(_mm(float(value)) for value in size)}: the A1 mini fits at most '
                      f'{_mm(PIECE_PLAN)} × {_mm(PIECE_PLAN)} in plan (its {_mm(BED)} bed less a 7 mm margin each side).')


def _ends(metadata: dict) -> list[dict]:
    ends = metadata.get('ends')
    if not isinstance(ends, list) or len(ends) != 2 or not all(isinstance(end, dict) for end in ends):
        raise Refused('Mark both ends of the piece.')
    for index, end in enumerate(ends, 1):
        if not all(is_finite(end.get(key)) for key in ('x', 'y', 'z', 'heading')):
            raise Refused(f'End {index} is not a number.')
    return [{key: float(end[key]) for key in ('x', 'y', 'z', 'heading')} for end in ends]


def _normalize(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _far_end(start: dict, end: dict) -> tuple[float, float, float]:
    """Where `end` lies seen from `start` facing into the piece: `meshCentreline`'s last point (community.ts)."""
    turn = -(start['heading'] + math.pi)
    c, s = math.cos(turn), math.sin(turn)
    dx, dy = end['x'] - start['x'], end['y'] - start['y']
    x, y, heading = c * dx - s * dy, s * dx + c * dy, _normalize(end['heading'] + turn)
    if abs(heading) < 1e-4 and abs(y) < 0.05:
        return x, 0.0, 0.0
    return x, y, heading


def ends_problem(start: dict, end: dict) -> str | None:
    """`meshEndsProblem` (community.ts): level, apart, and each end ahead of the other, turning at most 90°."""
    if abs(start['z'] - end['z']) > LEVEL:
        return 'The ends are not at the same rail height.'
    for one, other in ((start, end), (end, start)):
        x, y, heading = _far_end(one, other)
        if math.hypot(x, y) < SHORTEST:
            return 'The ends are too close together.'
        if x <= 0 or abs(heading) > math.pi / 2 + 1e-6:
            return 'The ends do not face away from each other.'
    return None


def end_face(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, heading: float,
             offset: float) -> tuple[float, float] | None:
    """The end face in the plane at `offset` along `heading`: every face in that plane facing out along `heading`,
    taken together, since a connector's tongue and socket cut it into separate pieces. Returns its lateral middle and its
    width (its span across `heading`), or None when nothing in that plane faces that way."""
    d = np.array([math.cos(heading), math.sin(heading)])
    across = np.array([-d[1], d[0]])
    planar = normals[:, :2] @ d >= END_FACE_COSINE
    along = vertices[:, :2] @ d
    planar &= (np.abs(along[faces] - offset) <= END_FACE_OFFSET).all(axis=1)
    if not planar.any():
        return None
    side = vertices[np.unique(faces[planar])][:, :2] @ across
    low, high = float(side.min()), float(side.max())
    return (low + high) / 2, high - low


def rail_top(vertices: np.ndarray, x: float, y: float) -> float:
    """Rail height at an end: the highest vertex within `RAIL_REACH` of it in plan (or the nearest, when none is)."""
    distance = np.hypot(vertices[:, 0] - x, vertices[:, 1] - y)
    return float(vertices[distance <= max(RAIL_REACH, float(distance.min())), 2].max())


def _check_end(mesh, index: int, end: dict) -> None:
    vertices, faces, normals = np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.face_normals)
    d = (math.cos(end['heading']), math.sin(end['heading']))
    offset, lateral = end['x'] * d[0] + end['y'] * d[1], -end['x'] * d[1] + end['y'] * d[0]
    face = end_face(vertices, faces, normals, end['heading'], offset)
    if face is None:
        if end_face(vertices, faces, normals, end['heading'] + math.pi, -offset) is not None:
            raise Refused(f'End {index} points into the piece: flip its arrow.')
        raise Refused(f'End {index} is not on the boundary of the mesh: mark it on an end face.')
    middle, width = face
    if width < END_FACE_WIDTH:
        raise Refused(f'End {index} is on a face only {_mm(width)} wide: mark the end face where track leaves the piece, '
                      'not a side.')
    if abs(middle - lateral) > END_TOLERANCE:
        raise Refused(f'End {index} is {_mm(abs(middle - lateral))} off the middle of its end face: mark it again.')
    top = rail_top(vertices, end['x'], end['y'])
    if abs(top - end['z']) > END_TOLERANCE:
        raise Refused(f'End {index} is at {_mm(end["z"])} high, but the rails there are {_mm(top)} high: mark it at '
                      'rail height.')


def check_piece(data: bytes, metadata: object) -> dict:
    triangles = stl_triangles(data)
    mesh = _solid(triangles)
    low, high = mesh.bounds
    _size_problem(high - low)
    metadata = _common(metadata)
    ends = _ends(metadata)
    for index, end in enumerate(ends, 1):
        _check_end(mesh, index, end)
    problem = ends_problem(*ends)
    if problem:
        raise Refused(problem)
    unknown = set(metadata) - {'id', 'name', 'ends', 'source'}
    if unknown:
        raise Refused(f'Unknown piece metadata: {", ".join(sorted(unknown))}.')
    if metadata.get('source') is not None:
        _source(metadata['source'])
    return {'triangles': len(triangles), 'size': [round(float(v), 2) for v in high - low]}


# Trains ---------------------------------------------------------------------------------------------------------


def _glb(data: bytes) -> tuple[dict, bytes]:
    """The JSON and binary chunks of a self-contained glTF binary, in the shape `write_glb` (scripts/trains/glb.py)
    writes: one embedded buffer, no URIs, images, textures or extensions."""
    if len(data) > TRAIN_BYTES:
        raise Refused(f'The model is {len(data) / 1000:.0f} KB; the limit is {TRAIN_BYTES // 1000} KB.')
    if len(data) < 28:
        raise Refused('The file is not a glTF binary (.glb).')
    magic, version, length = struct.unpack_from('<III', data)
    if magic != GLB_MAGIC or version != 2 or length != len(data):
        raise Refused('The file is not a glTF 2.0 binary (.glb).')
    json_length, json_type = struct.unpack_from('<II', data, 12)
    if json_type != GLB_JSON or 20 + json_length + 8 > len(data):
        raise Refused('The glTF binary is damaged.')
    try:
        gltf = json.loads(data[20:20 + json_length])
    except (UnicodeDecodeError, ValueError):
        raise Refused('The glTF binary is damaged.') from None
    bin_length, bin_type = struct.unpack_from('<II', data, 20 + json_length)
    end = 28 + json_length + bin_length
    if bin_type != GLB_BIN or end != len(data):
        raise Refused('The model must be one glTF binary with one embedded buffer and nothing after it.')
    if not isinstance(gltf, dict):
        raise Refused('The glTF binary is damaged.')
    buffers = gltf.get('buffers')
    if not isinstance(buffers, list) or len(buffers) != 1 or not isinstance(buffers[0], dict):
        raise Refused('The model must have exactly one buffer, embedded in the file.')
    if any(isinstance(item, dict) and 'uri' in item for key in ('buffers', 'images') for item in gltf.get(key) or []):
        raise Refused('The model refers to an external file (a uri): embed everything in the .glb.')
    for key, label in (('images', 'images'), ('textures', 'textures'), ('samplers', 'texture samplers'),
                       ('extensionsUsed', 'glTF extensions'), ('extensionsRequired', 'glTF extensions')):
        if gltf.get(key):
            raise Refused(f'The model has {label}: use vertex colours only, as Plateway\'s train models do.')
    binary = data[28 + json_length:end]
    if buffers[0].get('byteLength') != len(binary) and not 0 <= len(binary) - buffers[0].get('byteLength', -9) < 4:
        raise Refused('The glTF binary is damaged.')
    return gltf, binary


def _accessor(gltf: dict, binary: bytes, index: object, kinds: dict[str, tuple[int, ...]]) -> np.ndarray:
    accessors, views = gltf.get('accessors') or [], gltf.get('bufferViews') or []
    if type(index) is not int or not 0 <= index < len(accessors) or not isinstance(accessors[index], dict):
        raise Refused('The glTF binary is damaged.')
    accessor = accessors[index]
    kind, component, count = accessor.get('type'), accessor.get('componentType'), accessor.get('count')
    if kind not in kinds or component not in kinds[kind] or type(count) is not int or count < 0 or 'sparse' in accessor:
        raise Refused('The model uses a vertex layout Plateway does not read.')
    view_index = accessor.get('bufferView')
    if type(view_index) is not int or not 0 <= view_index < len(views) or not isinstance(views[view_index], dict):
        raise Refused('The glTF binary is damaged.')
    view = views[view_index]
    size = COMPONENT_BYTES[component] * TYPE_SIZES[kind]
    start = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    stride = view.get('byteStride', size)
    if view.get('buffer') != 0 or type(start) is not int or start < 0 or stride != size or start % COMPONENT_BYTES[component]:
        raise Refused('The model uses a vertex layout Plateway does not read.')
    if start + size * count > view.get('byteOffset', 0) + view.get('byteLength', 0) or start + size * count > len(binary):
        raise Refused('The glTF binary is damaged.')
    dtype = {5120: '<i1', UBYTE: '<u1', 5122: '<i2', USHORT: '<u2', UINT: '<u4', FLOAT: '<f4'}[component]
    return np.frombuffer(binary, dtype, count * TYPE_SIZES[kind], start).reshape(count, TYPE_SIZES[kind])


def glb_triangles(data: bytes) -> np.ndarray:
    """Every triangle (n, 3, 3) of every primitive of every mesh, checked against the model shape and budget."""
    gltf, binary = _glb(data)
    nodes, meshes = gltf.get('nodes'), gltf.get('meshes')
    if (not isinstance(nodes, list) or len(nodes) != 1 or not isinstance(nodes[0], dict) or nodes[0].get('mesh') != 0
            or {'matrix', 'rotation', 'scale', 'translation', 'children'} & set(nodes[0])):
        raise Refused('The model must be one node holding one mesh, already in the car frame (no transform).')
    if not isinstance(meshes, list) or not meshes or not all(isinstance(mesh, dict) for mesh in meshes):
        raise Refused('The model has no mesh.')
    triangles = []
    for mesh in meshes:
        for primitive in mesh.get('primitives') or []:
            if not isinstance(primitive, dict) or primitive.get('mode', 4) != 4 or 'targets' in primitive:
                raise Refused('The model must be plain triangles.')
            attributes = primitive.get('attributes') or {}
            if not isinstance(attributes, dict) or not {'POSITION', 'COLOR_0'} <= set(attributes):
                raise Refused('Every part of the model needs positions and vertex colours.')
            positions = _accessor(gltf, binary, attributes['POSITION'], {'VEC3': (FLOAT,)})
            _accessor(gltf, binary, attributes['COLOR_0'], {'VEC3': (UBYTE, USHORT, FLOAT), 'VEC4': (UBYTE, USHORT, FLOAT)})
            indices = _accessor(gltf, binary, primitive.get('indices'), {'SCALAR': (UBYTE, USHORT, UINT)}).ravel()
            if len(indices) % 3 or (len(indices) and int(indices.max()) >= len(positions)):
                raise Refused('The glTF binary is damaged.')
            triangles.append(positions.astype(np.float64)[indices].reshape(-1, 3, 3))
    count = sum(len(part) for part in triangles)
    if not count:
        raise Refused('The model has no triangles.')
    if count > TRAIN_TRIANGLES:
        raise Refused(f'The model has {count:,} triangles; the limit is {TRAIN_TRIANGLES:,}. Simplify it.')
    result = np.concatenate(triangles)
    if not np.isfinite(result).all():
        raise Refused('The model has coordinates that are not numbers.')
    return result


def _hull(points: np.ndarray) -> list[tuple[float, float]]:
    """The convex hull of plan points, anticlockwise (monotone chain)."""
    unique = sorted(set(map(tuple, np.round(points, 3).tolist())))
    if len(unique) < 3:
        return unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def measure_train(data: bytes, couplers: dict) -> dict:
    """A car's measured fields from its model (the index's `length`, `width`, `height`, `outline`, `triangles`): the
    plan outline is the model's convex hull, cut to at most 40 corners."""
    triangles = glb_triangles(data)
    points = triangles.reshape(-1, 3)
    low, high = points.min(0), points.max(0)
    outline = _hull(points[:, :2])
    while len(outline) > OUTLINE_MAX:
        # Drop the corner whose removal loses the least area.
        areas = [abs((outline[i - 1][0] - outline[i][0]) * (outline[(i + 1) % len(outline)][1] - outline[i][1]) -
                     (outline[i - 1][1] - outline[i][1]) * (outline[(i + 1) % len(outline)][0] - outline[i][0]))
                 for i in range(len(outline))]
        outline.pop(int(np.argmin(areas)))
    front = couplers.get('front') if isinstance(couplers, dict) and is_finite(couplers.get('front')) else float(high[0])
    rear = couplers.get('rear') if isinstance(couplers, dict) and is_finite(couplers.get('rear')) else float(low[0])
    return {'length': round(max(front, float(high[0])) - min(rear, float(low[0])), 1),
            'width': round(float(high[1] - low[1]), 1), 'height': round(float(high[2]), 1),
            'outline': [[round(x, 1), round(y, 1)] for x, y in outline], 'triangles': len(triangles)}


def _source(source: object) -> None:
    """A CC BY model's web address and authors, for an item adapted from it (`source` in a piece's or train's metadata)."""
    if not isinstance(source, dict) or set(source) - {'url', 'authors'}:
        raise Refused('The source must give its URL and authors.')
    url = source.get('url')
    if not isinstance(url, str) or not re.fullmatch(r'https://[^\s/]+\.[^\s/]+(/\S*)?', url) or len(url) > 500:
        raise Refused("An adapted model needs its source's web address (https://…).")
    authors = source.get('authors')
    if (not isinstance(authors, list) or not authors or len(authors) > 20 or
            not all(isinstance(author, str) and author.strip() and len(author) <= TEXT_MAX for author in authors)):
        raise Refused("An adapted model needs its source's authors.")


def _licence(metadata: dict) -> None:
    source, licence = metadata.get('source'), metadata.get('licence')
    if source is None:
        if licence != ORIGINAL_LICENCE:
            raise Refused('A model of your own is shared under CC BY 4.0.')
        return
    _source(source)
    if licence not in ADAPTED_LICENCES:
        raise Refused(f'An adapted model keeps its source\'s CC BY licence: one of {", ".join(ADAPTED_LICENCES)}.')


def check_train(data: bytes, metadata: object) -> dict:
    triangles = glb_triangles(data)
    metadata = _common(metadata)
    unknown = set(metadata) - {'id', 'name', 'year', 'power', 'gauge', 'car', 'licence', 'source'}
    if unknown:
        raise Refused(f'Unknown train metadata: {", ".join(sorted(unknown))}.')
    if type(metadata.get('year')) is not int or not 1900 <= metadata['year'] <= 2100:
        raise Refused('The year must be a whole year.')
    if metadata.get('power') not in POWER:
        raise Refused(f'The power must be one of {", ".join(POWER)}.')
    if metadata.get('gauge') not in ('standard', 'monorail'):
        raise Refused('The gauge must be standard or monorail.')
    _licence(metadata)
    car = metadata.get('car')
    if not isinstance(car, dict) or set(car) - {'kind', 'motorised', 'length', 'width', 'height', 'bogies', 'wheelbase',
                                                 'couplers', 'outline', 'triangles'}:
        raise Refused('The car measurements are incomplete.')
    if car.get('kind') not in CAR_KINDS or type(car.get('motorised')) is not bool:
        raise Refused('The car needs its kind and whether it is motorised.')
    sizes = [car.get(key) for key in ('length', 'width', 'height', 'wheelbase')]
    if not all(is_finite(value) and value > 0 for value in sizes):
        raise Refused('The car needs its length, width, height and wheelbase.')
    bogies = car.get('bogies')
    if not isinstance(bogies, list) or len(bogies) < 2 or not all(is_finite(value) for value in bogies) or bogies != sorted(bogies):
        raise Refused('The car needs at least two bogie pivots, rear to front.')
    if abs(bogies[0] + bogies[-1]) > 0.5:
        raise Refused("The car's origin must be midway between its outer bogie pivots.")
    if abs(car['wheelbase'] - (bogies[-1] - bogies[0])) > 0.11:
        raise Refused('The wheelbase must be the distance between the outer bogie pivots.')
    couplers = car.get('couplers')
    if not isinstance(couplers, dict) or not is_finite(couplers.get('front')) or not is_finite(couplers.get('rear')):
        raise Refused('The car needs its front and rear coupler positions.')
    if not couplers['front'] > bogies[-1] or not couplers['rear'] < bogies[0]:
        raise Refused('The couplers must be outside the outer bogie pivots.')
    outline = car.get('outline')
    if (not isinstance(outline, list) or not 3 <= len(outline) <= OUTLINE_MAX or
            not all(isinstance(point, list) and len(point) == 2 and all(is_finite(v) for v in point) for point in outline)):
        raise Refused(f'The car outline must be 3 to {OUTLINE_MAX} points.')
    if car.get('triangles') != len(triangles):
        raise Refused(f'The car says {car.get("triangles")} triangles, but its model has {len(triangles)}.')
    if car['width'] > TRAIN_WIDEST:
        raise Refused(f'The car is {_mm(car["width"])} wide; LEGO trains are at most {_mm(TRAIN_WIDEST)}.')
    points = triangles.reshape(-1, 3)
    low, high = points.min(0), points.max(0)
    if (abs(float(high[1] - low[1]) - car['width']) >= TRAIN_FIT or abs(float(high[2]) - car['height']) >= TRAIN_FIT or
            abs(max(couplers['front'], float(high[0])) - min(couplers['rear'], float(low[0])) - car['length']) >= TRAIN_FIT):
        raise Refused("The car's measurements don't match its model: is the model in the car frame, in millimetres?")
    if float(low[2]) <= TRAIN_LOWEST:
        raise Refused('The model reaches too far below the rails: put the railhead top at z = 0.')
    return {'triangles': len(triangles), 'bytes': len(data)}


# Furniture ------------------------------------------------------------------------------------------------------


def _point(value: object) -> bool:
    return isinstance(value, dict) and is_finite(value.get('x')) and is_finite(value.get('y'))


def _name(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 200


def furniture_problem(furniture: object) -> str | None:
    """`validFurnitureType` (src/domain/furniture.ts), then the outline: every surface, cubby and panel inside it."""
    if not isinstance(furniture, dict) or 'placeholder' in furniture or 'library' in furniture:
        return 'The file is not a furniture type.'
    f = furniture
    sizes = [f.get('width'), f.get('depth'), f.get('height')]
    if not _name(f.get('family')) or not _name(f.get('label')):
        return 'The furniture needs a family and a label.'
    if not all(is_finite(size) and 0 < size <= FURNITURE_LARGEST for size in sizes):
        return f'The outline width, depth and height must be 1 to {FURNITURE_LARGEST:,.0f} mm.'
    surfaces, panels, orientations = f.get('surfaces'), f.get('panels'), f.get('orientations')
    if not isinstance(surfaces, list) or not isinstance(panels, list):
        return 'The furniture needs surfaces and panels.'
    for surface in surfaces:
        if (not isinstance(surface, dict) or not _name(surface.get('id')) or not isinstance(surface.get('name'), str) or
                not is_finite(surface.get('height')) or not isinstance(surface.get('footprint'), list) or
                len(surface['footprint']) < 3 or not all(_point(p) for p in surface['footprint']) or
                not isinstance(surface.get('entries'), list) or
                not all(isinstance(e, dict) and _name(e.get('id')) and _point(e.get('a')) and _point(e.get('b')) and
                        is_finite(e.get('heading')) for e in surface['entries'])):
            return 'A surface is incomplete.'
        cell = surface.get('cell')
        if cell is not None and not (isinstance(cell, dict) and type(cell.get('row')) is int and type(cell.get('col')) is int):
            return 'A cubby has no row and column.'
    if len({surface['id'] for surface in surfaces}) != len(surfaces):
        return 'Two surfaces share an id.'
    for panel in panels:
        if (not isinstance(panel, dict) or not all(is_finite(panel.get(key)) for key in ('x', 'y', 'z')) or
                not all(is_finite(panel.get(key)) and panel[key] > 0 for key in ('width', 'depth', 'height'))):
            return 'A panel is incomplete.'
    if (not isinstance(orientations, list) or not orientations or len(set(map(repr, orientations))) != len(orientations) or
            not all(type(turns) is int and turns in (0, 1, 2, 3) for turns in orientations)):
        return 'The orientations must be distinct quarter turns.'
    colour = f.get('colour')
    if type(colour) is not int or not 0 <= colour <= 0xFFFFFF:
        return 'The colour must be a 0xRRGGBB number.'
    plan = f.get('plan')
    if (not isinstance(plan, dict) or not isinstance(plan.get('lines'), list) or not isinstance(plan.get('caption'), str) or
            not all(isinstance(line, dict) and _point(line.get('a')) and _point(line.get('b')) for line in plan['lines'])):
        return 'The top-view drawing is incomplete.'
    if not isinstance(f.get('terms'), str) or not _name(f.get('description')):
        return 'The furniture needs its search words and description.'
    if 'prefix' in f and not _name(f['prefix']):
        return 'The name prefix must be text.'
    width, depth, height = sizes
    half_w, half_d, slack = width / 2, depth / 2, 0.01
    outline = f'{_mm(width)} × {_mm(depth)} outline'
    for surface in surfaces:
        label = (f'Cubby {surface["cell"]["row"]}-{surface["cell"]["col"]}' if surface.get('cell') else
                 f'The surface "{surface["name"] or surface["id"]}"')
        if not 0 < surface['height'] <= height + slack:
            return f'{label} is at {_mm(surface["height"])}, outside the furniture\'s {_mm(height)} height.'
        if any(abs(p['x']) > half_w + slack or abs(p['y']) > half_d + slack for p in surface['footprint']):
            return (f'The cubby grid is larger than the {outline}: {label.lower()} sticks out.' if surface.get('cell')
                    else f'{label} sticks out of the {outline}.')
    for panel in panels:
        if (abs(panel['x']) + panel['width'] / 2 > half_w + slack or abs(panel['y']) + panel['depth'] / 2 > half_d + slack
                or panel['z'] - panel['height'] / 2 < -slack or panel['z'] + panel['height'] / 2 > height + slack):
            return f'A board sticks out of the {outline}.'
    if not surfaces:
        return 'The furniture needs at least one surface for track.'
    return None


def check_furniture(data: bytes, metadata: object) -> dict:
    if len(data) > FURNITURE_BYTES:
        raise Refused(f'The furniture file is {len(data) // 1000} KB; the limit is {FURNITURE_BYTES // 1000} KB.')
    try:
        furniture = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (UnicodeDecodeError, ValueError):
        raise Refused('The furniture file is not JSON.') from None
    problem = furniture_problem(furniture)
    if problem:
        raise Refused(problem)
    metadata = _common(metadata)
    unknown = set(metadata) - {'id', 'name', 'design'}
    if unknown:
        raise Refused(f'Unknown furniture metadata: {", ".join(sorted(unknown))}.')
    if 'id' in metadata and furniture['family'] != metadata['id']:
        raise Refused('The furniture family must be its library id.')
    return {'surfaces': len(furniture['surfaces']), 'panels': len(furniture['panels'])}


CHECKS: dict[str, Callable[[bytes, object], dict]] = {
    'piece': check_piece, 'train': check_train, 'furniture': check_furniture}


def check(kind: str, data: bytes, metadata: object) -> dict:
    """Check one contribution; a summary of what was checked, or `Refused`."""
    if kind not in CHECKS:
        raise Refused('The kind must be piece, train or furniture.')
    return CHECKS[kind](data, metadata)


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in CHECKS:
        print(f'usage: {argv[0]} piece|train|furniture FILE METADATA.json', file=sys.stderr)
        return 2
    with open(argv[2], 'rb') as file, open(argv[3], encoding='utf-8') as meta:
        data, metadata = file.read(), json.load(meta)
    try:
        summary = check(argv[1], data, metadata)
    except Refused as refused:
        print(f'Refused: {refused}')
        return 1
    print(f'Passed: {json.dumps(summary)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
