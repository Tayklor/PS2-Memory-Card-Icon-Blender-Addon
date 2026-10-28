"""PS2 icon import and export for Blender."""

bl_info = {
    "name": "PS2 Icons",
    "author": "TayKlor",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "File > Import / Export",
    "description": "Import and export PS2 icons (.icn)",
    "category": "Import-Export",
}

from . import importer, exporter


def register():
    importer.register()
    exporter.register()


def unregister():
    exporter.unregister()
    importer.unregister()
