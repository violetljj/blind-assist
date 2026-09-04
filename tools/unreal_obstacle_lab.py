"""Create an editable UE lab from locally installed Epic template assets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--open', action='store_true', help='Open the finished lab in Unreal Editor')
    parser.add_argument('--verify', action='store_true', help='Render scene views and verify first-person PIE and motion')
    parser.add_argument('--scene', choices=('street', 'graybox'), default='street')
    parser.add_argument('--prepare-only', action='store_true', help='Copy template assets without starting the editor')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    name = 'BlindAssistStreetLab' if args.scene == 'street' else 'BlindAssistObstacleLab'
    root = repo / 'artifacts.local/unreal' / name
    engine = args.engine.resolve()
    editor = engine / 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe'
    if not editor.is_file():
        parser.error(f'Unreal Editor is missing: {editor}')
    root.mkdir(parents=True, exist_ok=True)
    project = root / f'{name}.uproject'
    if not project.exists():
        template = engine / 'Templates/TP_FirstPersonBP'
        shutil.copytree(template / 'Content/FirstPerson', root / 'Content/FirstPerson')
        for pack in ('Characters', 'Input', 'LevelPrototyping'):
            shutil.copytree(engine / f'Templates/TemplateResources/High/{pack}/Content',
                            root / f'Content/{pack}')
        if args.scene == 'street':
            for pack in ('ArchVis', 'Building', 'Vehicles'):
                shutil.copytree(engine / f'Templates/TemplateResources/Standard/{pack}/Content',
                                root / f'Content/{pack}')
        (root / 'Config').mkdir(exist_ok=True)
        shutil.copy2(template / 'Config/DefaultInput.ini', root / 'Config/DefaultInput.ini')
        project.write_text(json.dumps({
            'FileVersion': 3, 'EngineAssociation': '5.8',
            'Description': 'BlindAssist pedestrian obstacle laboratory; synthetic Development only',
            'Plugins': [{'Name': name, 'Enabled': True} for name in
                        ('PythonScriptPlugin', 'EditorScriptingUtilities', 'SequencerScripting',
                         'EnhancedInput', 'GameplayStateTree')]
        }, indent=2), encoding='utf-8')
        engine_config = '''[/Script/EngineSettings.GameMapsSettings]
EditorStartupMap=/Game/ObstacleLab/ObstacleLab
GameDefaultMap=/Game/ObstacleLab/ObstacleLab
GlobalDefaultGameMode=/Game/FirstPerson/Blueprints/BP_FirstPersonGameMode.BP_FirstPersonGameMode_C

[/Script/Engine.RendererSettings]
r.DynamicGlobalIlluminationMethod=0
r.ReflectionMethod=0
r.AllowStaticLighting=False
r.Shadow.Virtual.Enable=0
r.DefaultFeature.AutoExposure=False
r.DefaultFeature.MotionBlur=False

[/Script/Engine.Engine]
NearClipPlane=5.0

[/Script/UnrealEd.CookerSettings]
bEnableCookOnTheSide=False
'''
        if args.scene == 'street':
            engine_config = engine_config.replace('/Game/ObstacleLab/ObstacleLab', '/Game/StreetLab/StreetLab')
            engine_config = engine_config.replace('r.DynamicGlobalIlluminationMethod=0', 'r.DynamicGlobalIlluminationMethod=1\nr.GenerateMeshDistanceFields=True')
            engine_config = engine_config.replace('r.ReflectionMethod=0', 'r.ReflectionMethod=1')
            engine_config = engine_config.replace('r.Shadow.Virtual.Enable=0', 'r.Shadow.Virtual.Enable=1')
            engine_config = engine_config.replace('r.DefaultFeature.AutoExposure=False', 'r.DefaultFeature.AutoExposure=True\nr.DefaultFeature.AutoExposure.ExtendDefaultLuminanceRange=True\nr.DefaultFeature.AutoExposure.Bias=-1.0')
            engine_config += '\n[/Script/WindowsTargetPlatform.WindowsTargetSettings]\nDefaultGraphicsRHI=DefaultGraphicsRHI_DX12\n+D3D12TargetedShaderFormats=PCD3D_SM6\n'
        (root / 'Config/DefaultEngine.ini').write_text(engine_config, encoding='utf-8')
        (root / 'Config/DefaultGame.ini').write_text(
            '[/Script/EngineSettings.GeneralProjectSettings]\n'
            f'ProjectName={name}\n'
            'Description=Pedestrian obstacle scenarios. Synthetic Development playground.\n',encoding='utf-8')
    if args.prepare_only:
        print(f'PROJECT: {project}')
        return
    script = repo / 'research/active/dtr-r0/unreal' / ('build_street_lab.py' if args.scene == 'street' else 'build_obstacle_lab.py')
    log = root / 'Saved/Logs/build-lab.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cache = f'-LocalDataCachePath={root / "DerivedDataCache"}'
    map_file = 'Content/StreetLab/StreetLab.umap' if args.scene == 'street' else 'Content/ObstacleLab/ObstacleLab.umap'
    if not (root / map_file).exists():
        if args.scene == 'street':
            subprocess.run([sys.executable,str(script.with_name('download_street_materials.py'))],check=True)
        subprocess.run([str(editor), str(project), '-run=pythonscript', f'-script={script}',
                        '-unattended', '-nullrhi', '-nosound', '-nop4', '-NoSplash',
                        f'-abslog={log}', '-ddc=NoShared', cache], check=True)
    receipt = root / 'Saved/lab-build.json'
    if not receipt.is_file():
        raise RuntimeError(f'Build did not produce a receipt; inspect {log}')
    print(receipt.read_text(encoding='utf-8'))
    print(f'PROJECT: {project}')
    if args.verify:
        verify = script.with_name('verify_street_lab.py' if args.scene == 'street' else 'verify_obstacle_lab.py')
        subprocess.run([str(editor.with_name('UnrealEditor.exe')), str(project),
                        f'-ExecCmds=py {verify.as_posix()}', '-unattended', '-nosound', '-nop4',
                        '-NoSplash', '-RenderOffscreen', '-windowed', '-ResX=1600', '-ResY=1000',
                        '-ddc=NoShared', cache, f'-abslog={log.with_name("verify-lab.log")}'],
                       check=True)
        result = json.loads((root / 'Saved/lab-smoke.json').read_text(encoding='utf-8'))
        print(json.dumps(result, indent=2))
        if result['status'] != 'PASS':
            raise RuntimeError('Editor smoke check failed')
    if args.open:
        process = subprocess.Popen([str(editor.with_name('UnrealEditor.exe')), str(project),
                                   '-NoSplash', '-ddc=NoShared', cache],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        print(f'EDITOR_PID: {process.pid}')


if __name__ == '__main__':
    main()
