# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []

datas += [
    ('file_category_mapping.pkl', '.'),
    ('black_white', 'black_white'),
    ('original', 'original'),
]

datas += collect_data_files('PySide6')
datas += collect_data_files('cv2')
datas += collect_data_files('PIL')
datas += collect_data_files('numpy')
datas += collect_data_files('imagehash')
datas += collect_data_files('utils')

hiddenimports = []
hiddenimports += collect_submodules('PySide6')
hiddenimports += collect_submodules('PySide6.QtWidgets')
hiddenimports += collect_submodules('PySide6.QtGui')
hiddenimports += collect_submodules('PySide6.QtCore')
hiddenimports += collect_submodules('PySide6.QtSvg')
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('imagehash')
hiddenimports += collect_submodules('utils')

a = Analysis(['main.py'],
             pathex=['.'],
             binaries=[],
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure,
          a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='郵票分類器IPAI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          icon = 'logo/favicon.ico',
          console=False)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='郵票分類器IPAI')
