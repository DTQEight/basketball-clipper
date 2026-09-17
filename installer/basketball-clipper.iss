; ============================================================
;  篮球进球集锦助手 - Inno Setup 安装脚本
;
;  用法:
;    ISCC.exe installer\basketball-clipper.iss
;
;  产物:
;    dist\installer\BasketballClipper-Setup-1.0.0.exe
;
;  说明:
;    载荷使用 dist\basketball-clipper\ (PyInstaller onedir 目录版)
;    不使用单文件 exe，否则每次启动都要解压到临时目录
; ============================================================

#define MyAppName        "篮球进球集锦助手"
#define MyAppVersion     "1.0.0"
#define MyAppPublisher   "DTQEight"
#define MyAppURL         "https://github.com/DTQEight/basketball-clipper"
#define MyAppExeName     "basketball-clipper.exe"

[Setup]
; AppId 唯一标识本软件，升级安装时不会重复装一份，请勿随意修改
AppId={{B7C4E9A2-5D31-4F8E-9C6B-2A1D7E3F0B45}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} 安装程序
VersionInfoProductName={#MyAppName}

; 默认安装到 Program Files，该页允许用户自行修改
DefaultDirName={autopf}\Basketball Clipper
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
AllowNoIcons=yes

; 许可协议页
LicenseFile=LICENSE_CN.txt

; 仅支持 64 位 Windows
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0

; 写入 Program Files 需要管理员权限
PrivilegesRequired=admin

; 卸载信息
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}

; 输出
OutputDir=..\dist\installer
OutputBaseFilename=BasketballClipper-Setup-{#MyAppVersion}
WizardStyle=modern
SetupLogging=yes
CloseApplications=yes
RestartApplications=no

; 压缩（LZMA2 固实压缩，体积最小，编译耗时较长）
Compression=lzma2/max
SolidCompression=yes

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: checkedonce

[Files]
; 整个 onedir 目录（含 _internal）原样安装
; 排除 cache：里面是打包机上跑测试留下的日志，带过去会污染目标机的
; CUDA 预热状态判断（warmup_status.log）和历史记录
Source: "..\dist\basketball-clipper\*"; DestDir: "{app}"; \
    Excludes: "\cache\*,__pycache__\*"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即运行 {#MyAppName}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 清掉运行期产生的缓存目录
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\__pycache__"
