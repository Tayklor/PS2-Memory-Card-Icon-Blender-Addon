import os
import struct

import bpy
import bmesh
from bpy.props import BoolProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper


def to_fixed(value):
    return max(-32768, min(32767, int(value * 4096.0)))


def to_ps2_uv(value):
    return max(-32768, min(32767, round(float(value) * 4096.0)))


def pack_color(r, g, b, a):
    rgba = [max(0, min(128, int(float(c) * 128))) for c in (r, g, b, a)]
    return rgba[0] | (rgba[1] << 8) | (rgba[2] << 16) | (rgba[3] << 24)


def pack_pixel(r, g, b, a):
    if a < 0.5:
        return 0
    rgb = [max(0, min(255, int(float(c) * 255))) for c in (r, g, b)]
    value = (rgb[0] >> 3) | ((rgb[1] >> 3) << 5) | ((rgb[2] >> 3) << 10)
    return value if value else 1 


def get_image(obj):
    mat = obj.active_material
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        raise ValueError("Active material must use nodes and contain an Image Texture")
    active = mat.node_tree.nodes.active
    if active and active.type == 'TEX_IMAGE' and active.image:
        return active.image
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image:
            return node.image
    raise ValueError("No Image Texture found in the active material")


def build_texture(image, flip_v):
    if tuple(image.size) != (128, 128):
        raise ValueError("Image Texture must be exactly 128 x 128 pixels")
    if not image.has_data:
        try:
            image.reload()
        except RuntimeError:
            pass
    pixels = image.pixels[:]
    if len(pixels) < 128 * 128 * 4:
        raise ValueError("Image does not contain readable RGBA pixels")
    out = bytearray(32768)
    for y in range(128):
        sy = 127 - y if flip_v else y
        for x in range(128):
            src = (sy * 128 + x) * 4
            dst = (y * 128 + x) * 2
            struct.pack_into('<H', out, dst, pack_pixel(*pixels[src:src + 4]))
    return out


def build_vertices(context, obj, use_modifiers, apply_transforms, scale, flip_uv_v, flip_x=True):
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get()) if use_modifiers else obj
    source = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=context.evaluated_depsgraph_get())
    if source is None:
        raise ValueError("Could not evaluate the mesh")
    try:
        bm = bmesh.new()
        try:
            bm.from_mesh(source)
            if apply_transforms:
                transform = obj.matrix_world.copy()
                bm.transform(transform)
                if transform.determinant() < 0.0:
                    bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.to_mesh(source)
        finally:
            bm.free()
        source.update()
        uv_layer = source.uv_layers.active
        if uv_layer is None:
            raise ValueError("The mesh needs an active UV map")
        col = source.color_attributes.active_color if source.color_attributes else None
        if col and col.domain not in {'CORNER', 'POINT'}:
            col = None
        if not source.polygons:
            raise ValueError("The mesh has no faces")
        count = len(source.polygons) * 3
        if count > 65535:
            raise ValueError("Too many triangle vertices (maximum 65535)")
        data = bytearray()
        for poly in source.polygons:
            if poly.loop_total != 3:
                raise ValueError("Triangulation failed")
            loops = list(range(poly.loop_start, poly.loop_start + 3))
            for li in (loops if flip_x else (loops[0], loops[2], loops[1])):
                vertex_id = source.loops[li].vertex_index
                co = source.vertices[vertex_id].co
                pos = (-scale * co.x if flip_x else scale * co.x,
                       -scale * co.z, -scale * co.y)
                data.extend(struct.pack('<3hh', *(to_fixed(v) for v in pos), 0))
                normal = source.corner_normals[li].vector
                n = (-normal.x if flip_x else normal.x,
                     -normal.z, -normal.y)
                data.extend(struct.pack('<3hh', *(to_fixed(v) for v in n), 0))
                uv = uv_layer.uv[li].vector
                u, v = float(uv.x), float(uv.y)
                if flip_uv_v:
                    v = 1.0 - v
                if col is not None:
                    rgba = col.data[li if col.domain == 'CORNER' else vertex_id].color
                else:
                    rgba = (0.5, 0.5, 0.5, 0.5)
                data.extend(struct.pack('<2hI', to_ps2_uv(u), to_ps2_uv(v), pack_color(*rgba)))
        return count, data
    finally:
        evaluated.to_mesh_clear()


def write_icon(filepath, count, vertices, texture):
    with open(filepath, 'wb') as fp:
        fp.write(struct.pack('<5I', 0x010000, 1, 0x07, 0x3F800000, count))
        fp.write(vertices)
        fp.write(struct.pack('<IIfII', 1, 31, 1.0, 0, 1))
        fp.write(struct.pack('<II', 0, 1))
        fp.write(struct.pack('<ff', 0.0, 1.0))
        fp.write(texture)


class EXPORT_SCENE_OT_ps2_icon(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.ps2_icon"
    bl_label = "PS2 Icon (.icn)"
    bl_options = {'PRESET'}

    filename_ext = ".icn"
    filter_glob: StringProperty(default="*.icn", options={'HIDDEN'})
    use_modifiers: BoolProperty(name="Apply Modifiers", description="Include modifiers", default=True)
    apply_transforms: BoolProperty(
        name="Apply All Transforms",
        description="Apply transforms to the exported mesh",
        default=True,
    )
    scale: FloatProperty(name="Scale", description="Scale the exported model", default=1.0, min=0.001, max=100.0)
    flip_x: BoolProperty(
        name="Correct Orientation",
        description="Match PS2 icon viewer left/right orientation",
        default=True,
    )
    flip_uv_v: BoolProperty(name="Flip UV V", description="Flip V coordinates", default=False)
    flip_texture_y: BoolProperty(name="Flip Texture", description="Flip texture vertically", default=False)

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        try:
            image = get_image(obj)
            count, vertices = build_vertices(
                context, obj, self.use_modifiers, self.apply_transforms,
                self.scale, self.flip_uv_v, self.flip_x,
            )
            texture = build_texture(image, self.flip_texture_y)
            write_icon(self.filepath, count, vertices, texture)
        except (ValueError, OSError, RuntimeError, struct.error) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported {count // 3} triangles to {os.path.basename(self.filepath)}")
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(EXPORT_SCENE_OT_ps2_icon.bl_idname, text="PS2 Icon (.icn)")


def register():
    bpy.utils.register_class(EXPORT_SCENE_OT_ps2_icon)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(EXPORT_SCENE_OT_ps2_icon)

