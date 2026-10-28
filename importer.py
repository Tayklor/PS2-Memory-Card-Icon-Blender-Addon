import struct
from array import array

import bmesh
import bpy
from bpy.props import BoolProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

TEXTURE_SIZE = 128
TEXTURE_BYTES = TEXTURE_SIZE * TEXTURE_SIZE * 2
HEADER_BYTES = 20
FIXED_SCALE = 4096.0


def decompress_texture(data):
    if len(data) < 4:
        raise ValueError('Compressed icon texture header is missing')
    length, = struct.unpack_from('<I', data)
    if length > len(data) - 4 or length % 2:
        raise ValueError('Invalid compressed texture size')
    end = 4 + length
    cursor = 4
    words = []
    while cursor < end and len(words) < TEXTURE_SIZE * TEXTURE_SIZE:
        (run,) = struct.unpack_from('<H', data, cursor)
        cursor += 2
        if run < 0xff00:
            if cursor + 2 > end:
                raise ValueError('Truncated RLE repeat')
            (pixel,) = struct.unpack_from('<H', data, cursor)
            cursor += 2
            words.extend([pixel] * run)
        else:
            count = 0x10000 - run
            if cursor + count * 2 > end:
                raise ValueError('Truncated RLE literal block')
            words.extend(struct.unpack_from('<' + str(count) + 'H', data, cursor))
            cursor += count * 2
        if len(words) > TEXTURE_SIZE * TEXTURE_SIZE:
            raise ValueError('RLE texture expands beyond 128x128')
    if len(words) != TEXTURE_SIZE * TEXTURE_SIZE:
        raise ValueError('Compressed icon texture did not decode to 128x128')
    return struct.pack('<' + str(len(words)) + 'H', *words)


def read_icon(filepath, shape_index=0, uv_scale=None):
    with open(filepath, 'rb') as file:
        buf = file.read()

    if len(buf) < HEADER_BYTES + 20:
        raise ValueError('Icon file is too small')
    magic, animation_shapes, flags, reserved, count = struct.unpack_from('<5I', buf)
    if magic != 0x00010000:
        raise ValueError(f'Invalid PS2 icon header: 0x{magic:08X}')
    if not 1 <= animation_shapes <= 128:
        raise ValueError(f'Unsupported animation shape count: {animation_shapes}')
    if count == 0 or count % 3 or count > 65535:
        raise ValueError(f'Invalid vertex count: {count}')
    if shape_index >= animation_shapes:
        raise ValueError(f'Shape {shape_index} not found (icon contains {animation_shapes} shapes)')

    stride = 8 * animation_shapes + 16
    end_vertices = HEADER_BYTES + count * stride
    if end_vertices + 20 > len(buf):
        raise ValueError('Truncated vertex data: incorrect shape/vertex count')

    if uv_scale is None:
        uv_scale = FIXED_SCALE

    records = []
    for idx in range(count):
        base = HEADER_BYTES + idx * stride
        px, py, pz, _pad = struct.unpack_from('<3hH', buf, base + 8 * shape_index)
        shared = base + animation_shapes * 8
        nx, ny, nz, _npad, u, v, rgba = struct.unpack_from('<3hH2hI', buf, shared)
        pos = (px / FIXED_SCALE, -pz / FIXED_SCALE, -py / FIXED_SCALE)
        normal = (nx / FIXED_SCALE, -nz / FIXED_SCALE, -ny / FIXED_SCALE)
        uv = (u / uv_scale, v / uv_scale)
        color = tuple(min(1.0, ((rgba >> shift) & 255) / 128.0) for shift in (0, 8, 16, 24))
        records.append((pos, normal, uv, color))

    anim_start = end_vertices
    tag, duration, speed, play, frame_count = struct.unpack_from('<IIfII', buf, anim_start)
    if tag != 1 or frame_count > 1024:
        raise ValueError('Invalid icon animation header')
    def end_of_frames(frame_header_bytes):
        cursor = anim_start + 20
        for _ in range(frame_count):
            if cursor + frame_header_bytes > len(buf):
                return None
            _shape, keys = struct.unpack_from('<2I', buf, cursor)
            if keys > 100000:
                return None
            cursor += frame_header_bytes + keys * 8
            if cursor > len(buf):
                return None
        return cursor

    candidates = [c for c in (end_of_frames(16), end_of_frames(8)) if c is not None]
    if not candidates:
        raise ValueError('Truncated icon animation frames')
    if flags & 0x04 and not (flags & 0x08):
        valid = [c for c in candidates if c + TEXTURE_BYTES <= len(buf)]
        if not valid:
            raise ValueError('Missing uncompressed 128x128 icon texture')
        offset = min(valid, key=lambda c: abs(len(buf) - c - TEXTURE_BYTES))
    else:
        offset = candidates[0]
    animation_bytes = offset - anim_start
    if flags & 0x08:
        texture = decompress_texture(buf[offset:])
    elif flags & 0x04:
        if offset + TEXTURE_BYTES > len(buf):
            raise ValueError('Missing 128x128 texture')
        texture = buf[offset:offset + TEXTURE_BYTES]
    else:
        texture = bytes(TEXTURE_BYTES)

    return records, texture, animation_shapes, flags, animation_bytes

def decode_texture(texture, flip_y):
    pixels = array('f', [0.0]) * (TEXTURE_SIZE * TEXTURE_SIZE * 4)
    for y in range(TEXTURE_SIZE):
        source_y = TEXTURE_SIZE - 1 - y if flip_y else y
        for x in range(TEXTURE_SIZE):
            word = struct.unpack_from('<H', texture, 2 * (source_y * TEXTURE_SIZE + x))[0]
            i = 4 * (y * TEXTURE_SIZE + x)
            pixels[i] = (word & 31) / 31.0
            pixels[i + 1] = ((word >> 5) & 31) / 31.0
            pixels[i + 2] = ((word >> 10) & 31) / 31.0
            pixels[i + 3] = 1.0
    return pixels



def _merge_mesh_vertices(mesh, distance):
    corner_name = '_PS2_ICN_Original_Corner'
    corner_attr = mesh.attributes.new(
        name=corner_name, type='INT', domain='CORNER'
    )
    for loop in mesh.loops:
        corner_attr.data[loop.index].value = loop.vertex_index

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=distance)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()

    corner_attr = mesh.attributes.get(corner_name)
    if corner_attr is None or len(corner_attr.data) != len(mesh.loops):
        raise RuntimeError('Could not preserve icon face-corner data after merging')
    indices = [int(item.value) for item in corner_attr.data]
    mesh.attributes.remove(corner_attr)
    if not indices:
        raise ValueError('All icon faces disappeared during vertex merging')
    return indices


def _build_corner_uvs(mesh, records, corner_indices, flip_uv_v=False):
    if len(corner_indices) != len(mesh.loops):
        raise ValueError('ICN corner mapping does not match the imported mesh')
    flat = [0.0] * (2 * len(mesh.loops))
    for loop_idx, record_idx in enumerate(corner_indices):
        if not 0 <= record_idx < len(records):
            raise ValueError(f'Invalid ICN corner index: {record_idx}')
        u, v = records[record_idx][2]
        flat[2 * loop_idx] = float(u)
        flat[2 * loop_idx + 1] = float(1.0 - v if flip_uv_v else v)
    return flat


def _max_triangle_area(mesh, uv_floats):
    highest = 0.0
    for poly in mesh.polygons:
        a, b, c = poly.loop_indices
        ax, ay = uv_floats[a*2:a*2+2]
        bx, by = uv_floats[b*2:b*2+2]
        cx, cy = uv_floats[c*2:c*2+2]
        highest = max(highest, abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax)) / 2)
    return highest


def _read_corner_uvs(layer, count):
    output = [0.0] * (2 * count)
    layer.uv.foreach_get('vector', output)
    return output


def _write_verified_uvs(mesh, records, corner_indices, flip_uv_v=False):
    expected = _build_corner_uvs(mesh, records, corner_indices, flip_uv_v)
    layer = mesh.uv_layers.new(name='PS2 ICN UV', do_init=False)
    if layer is None:
        raise RuntimeError('Blender refused to create a UV layer')
    name = layer.name
    layer = mesh.uv_layers[name]
    if len(layer.uv) != len(mesh.loops):
        raise RuntimeError('UV corner count differs from mesh loop count')

    layer.uv.foreach_set('vector', expected)
    mesh.uv_layers.active = layer
    layer.active_render = True
    mesh.update()

    def verify():
        actual_layer = mesh.uv_layers[name]
        actual = _read_corner_uvs(actual_layer, len(mesh.loops))
        delta = max((abs(a - b) for a, b in zip(expected, actual)), default=0.0)
        area = _max_triangle_area(mesh, actual)
        return delta, area

    delta, area = verify()
    if delta > 1e-5:
        layer = mesh.uv_layers[name]
        for idx in range(len(mesh.loops)):
            layer.uv[idx].vector = (expected[2 * idx], expected[2 * idx + 1])
        mesh.update()
        delta, area = verify()
    if delta > 1e-5:
        raise RuntimeError(
            f'UV write failed in Blender: maximum coordinate error {delta:.7g}. '
            'No generated UV layout was substituted.'
        )
    mesh['ps2_icn_uv_source'] = 'Signed 16-bit / 4096 directly from ICN'
    mesh['ps2_icn_uv_largest_face_area'] = area
    mesh['ps2_icn_uv_max_coordinate_error'] = delta
    mesh['ps2_icn_uv_corner_count'] = len(mesh.loops)
    return mesh.uv_layers[name], area


def _make_material(name, texture_bytes, flip_texture_y, pack_image):
    image = bpy.data.images.new(name + '_Texture', width=TEXTURE_SIZE,
                                height=TEXTURE_SIZE, alpha=True)
    image.pixels.foreach_set(decode_texture(texture_bytes, flip_texture_y))
    image.update()
    if pack_image:
        image.pack()

    material = bpy.data.materials.new(name + '_Material')
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    tex = material.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = image
    tex.name = 'PS2 Icon Texture'
    if shader is not None:
        material.node_tree.links.new(tex.outputs['Color'], shader.inputs['Base Color'])
        shader.inputs['Alpha'].default_value = 1.0
        shader.inputs['Roughness'].default_value = 1.0
    return image, material


def _assign_vertex_colors(mesh, records, corner_indices):
    colors = mesh.color_attributes.new(name='PS2 Vertex Color',
                                       type='FLOAT_COLOR', domain='CORNER')
    for loop in mesh.loops:
        colors.data[loop.index].color = records[corner_indices[loop.index]][3]


def _apply_import_smoothing(mesh, records, corner_indices,
                            shade_smooth=True, use_icn_normals=True, flip_x=True):
    if not shade_smooth:
        return

    mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))
    if use_icn_normals:
        loop_normals = []
        for loop in mesh.loops:
            nx, ny, nz = records[corner_indices[loop.index]][1]
            if flip_x:
                nx = -nx
            length_squared = nx * nx + ny * ny + nz * nz
            if length_squared > 1.0e-12:
                factor = length_squared ** -0.5
                loop_normals.append((nx * factor, ny * factor, nz * factor))
            else:
                loop_normals.append((0.0, 0.0, 0.0))
        mesh.normals_split_custom_set(loop_normals)

    mesh.update()


def import_icon(context, filepath, scale, flip_uv_v, flip_texture_y,
                pack_image, shape_index, shade_smooth=True,
                use_icn_normals=True, merge_by_distance=True, flip_x=True):
    records, texture, shapes, flags, animation_bytes = read_icon(
        filepath, shape_index, uv_scale=FIXED_SCALE
    )
    name = bpy.path.display_name_from_filepath(filepath)

    vertices = [((-pos[0] if flip_x else pos[0]) * scale,
                 pos[1] * scale, pos[2] * scale)
                for pos, _normal, _uv, _color in records]
    faces = ([(i, i + 1, i + 2) for i in range(0, len(records), 3)]
             if flip_x else
             [(i, i + 2, i + 1) for i in range(0, len(records), 3)])
    mesh = bpy.data.meshes.new(name + '_PS2_Mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    if merge_by_distance:
        unit_scale = context.scene.unit_settings.scale_length
        distance = 0.0001 / unit_scale if unit_scale > 0 else 0.0001
        corner_indices = _merge_mesh_vertices(mesh, distance)
    else:
        corner_indices = [loop.vertex_index for loop in mesh.loops]

    obj = bpy.data.objects.new(name + '_PS2', mesh)
    context.collection.objects.link(obj)

    image = material = None
    try:
        _assign_vertex_colors(mesh, records, corner_indices)
        image, material = _make_material(name, texture, flip_texture_y,
                                         pack_image)
        mesh.materials.append(material)

        _apply_import_smoothing(mesh, records, corner_indices,
                                shade_smooth, use_icn_normals, flip_x)

        obj['ps2_icn_source_filepath'] = bpy.path.abspath(filepath)
        obj['ps2_icn_importer_version'] = '1.0.0'
        obj['ps2_icn_shade_smooth'] = bool(shade_smooth)
        obj['ps2_icn_custom_normals'] = bool(shade_smooth and use_icn_normals)
        obj['ps2_icn_merge_distance_m'] = 0.0001 if merge_by_distance else 0.0
        obj['ps2_icn_flip_x'] = bool(flip_x)

        _layer, area = _write_verified_uvs(mesh, records, corner_indices, flip_uv_v)
    except Exception:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)
        if material is not None:
            bpy.data.materials.remove(material, do_unlink=True)
        if image is not None:
            bpy.data.images.remove(image, do_unlink=True)
        raise

    for other in list(context.selected_objects):
        other.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj, shapes, animation_bytes


class IMPORT_SCENE_OT_ps2_icon(bpy.types.Operator, ImportHelper):
    bl_idname = 'import_scene.ps2_icon'
    bl_label = 'PS2 Icon (.icn)'
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = '.icn'
    filter_glob: StringProperty(default='*.icn;*.ico', options={'HIDDEN'})
    scale: FloatProperty(name='Scale', default=1.0, min=0.001, max=100.0)
    flip_x: BoolProperty(
        name='Correct Orientation',
        description='Match PS2 icon viewer left/right orientation',
        default=True,
    )
    flip_uv_v: BoolProperty(name='Flip UV V', default=False)
    flip_texture_y: BoolProperty(name='Flip Texture', default=False)
    pack_image: BoolProperty(name='Pack Texture', default=True)
    shade_smooth: BoolProperty(
        name='Smooth Shading',
        description='Smooth imported faces',
        default=True,
    )
    use_icn_normals: BoolProperty(
        name='Icon Normals',
        description='Use normals from the icon',
        default=True,
    )
    merge_by_distance: BoolProperty(
        name='Merge by Distance',
        description='Merge vertices within 0.0001 m',
        default=True,
    )
    shape_index: bpy.props.IntProperty(name='Shape', default=0, min=0, max=127)

    def execute(self, context):
        try:
            obj, shapes, anim_bytes = import_icon(
                context, self.filepath, self.scale, self.flip_uv_v,
                self.flip_texture_y, self.pack_image, self.shape_index,
                self.shade_smooth, self.use_icn_normals,
                self.merge_by_distance, self.flip_x
            )
        except (ValueError, OSError, RuntimeError, struct.error, AttributeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'},
                    f'Imported {len(obj.data.polygons)} triangles with UVs')
        if shapes > 1:
            self.report({'WARNING'}, f'Shape {self.shape_index + 1}/{shapes}; animation not imported')
        return {'FINISHED'}



def menu_func_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_ps2_icon.bl_idname,
                         text='PS2 Icon (.icn)')


def register():
    bpy.utils.register_class(IMPORT_SCENE_OT_ps2_icon)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_SCENE_OT_ps2_icon)
