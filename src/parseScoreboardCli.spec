# -*- mode: python ; coding: utf-8 -*-
import shutil
import os

a = Analysis(
    ['parseScoreboardCli.py'],
    pathex=[os.path.abspath('.')],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [('W ignore', None, 'OPTION')],
    exclude_binaries=True,
    name='parseScoreboardCli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='parseScoreboardCli',
)

shutil.copyfile('config.json', os.path.join(DISTPATH, 'parseScoreboardCli', '_internal', 'config.json'))
shutil.copyfile('parseConfig.json', os.path.join(DISTPATH, 'parseScoreboardCli', '_internal', 'parseConfig.json'))
shutil.copytree('data', os.path.join(DISTPATH, 'parseScoreboardCli', '_internal', 'data'))
