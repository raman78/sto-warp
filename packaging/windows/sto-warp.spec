# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the sto-warp Windows installer build.

Invoked from the repo root by the GitHub Actions workflow:

    pyinstaller --noconfirm --clean packaging/windows/sto-warp.spec

Produces ``dist/sto-warp/`` (onedir layout) consumed by Inno Setup.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


datas = []
datas += collect_data_files('warp')              # resources, baseline JSONs, etc.
datas += collect_data_files('easyocr')           # character lists + model configs
datas += collect_data_files('pyclipper')
datas += collect_data_files('shapely')

hiddenimports = []
hiddenimports += collect_submodules('warp')
hiddenimports += collect_submodules('easyocr')
hiddenimports += collect_submodules('skimage')   # scikit-image submodules sometimes missed
hiddenimports += collect_submodules('scipy')

excludes = [
    'tkinter',
    'matplotlib',
    'pandas',
    'IPython',
    'pytest',
    'jupyter',
    'notebook',
    'PyQt5',
    'PyQt6',
]

a = Analysis(
    ['warp/cli.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='sto-warp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon='packaging/windows/_build/sto-warp.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='sto-warp',
)
