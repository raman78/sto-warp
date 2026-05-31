; Inno Setup script for sto-warp Windows installer.
;
; Builds a per-user installer (no admin required) that lays down the
; PyInstaller onedir bundle from `dist\sto-warp\` under
; %LOCALAPPDATA%\Programs\sto-warp\, registers Start Menu + optional
; Desktop shortcuts, and adds an uninstaller. The version is injected
; from the CI workflow via /DSTOWarpVersion=...
;
; Invoke from the repo root:
;
;     iscc /DSTOWarpVersion=1.0.11 packaging\windows\sto-warp.iss

#ifndef STOWarpVersion
  #define STOWarpVersion "0.0.0"
#endif

[Setup]
; SourceDir anchors all relative paths below on the repo root rather
; than this .iss file's own directory (the Inno Setup default). The
; PyInstaller bundle lands in dist\sto-warp\ at the repo root and the
; workflow expects the installer at dist\installer\, so we resolve
; everything from there.
SourceDir=..\..
AppId={{A4D1F9B6-7E2C-4F18-9C2A-2F3D8D5E9A21}
AppName=sto-warp
AppVersion={#STOWarpVersion}
AppPublisher=raman78
AppPublisherURL=https://github.com/raman78/sto-warp
AppSupportURL=https://github.com/raman78/sto-warp/issues
DefaultDirName={localappdata}\Programs\sto-warp
DefaultGroupName=sto-warp
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=dist\installer
OutputBaseFilename=sto-warp-{#STOWarpVersion}-setup
SetupIconFile=packaging\windows\_build\sto-warp.ico
UninstallDisplayIcon={app}\sto-warp.exe
UninstallDisplayName=sto-warp {#STOWarpVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\sto-warp\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\sto-warp"; Filename: "{app}\sto-warp.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall sto-warp"; Filename: "{uninstallexe}"
Name: "{userdesktop}\sto-warp"; Filename: "{app}\sto-warp.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\sto-warp.exe"; Description: "Launch sto-warp"; Flags: nowait postinstall skipifsilent
