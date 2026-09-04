"""Create an editable UE lab from locally installed Epic template assets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--open', action='store_true', help='Open the finished lab in Unreal Editor')
    parser.add_argument('--verify', action='store_true', help='Render two views and verify first-person PIE and motion')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / 'artifacts.local/unreal/BlindAssistObstacleLab'
    engine = args.engine.resolve()
    editor = engine / 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe'
    if not editor.is_file():
        parser.error(f'Unreal Editor is missing: {editor}')
    root.mkdir(parents=True, exist_ok=True)
    project = root / 'BlindAssistObstacleLab.uproject'
    if not project.exists():
        template = engine / 'Templates/TP_FirstPersonBP'
        shutil.copytree(template / 'Content/FirstPerson', root / 'Content/FirstPerson')
        for pack in ('Characters', 'Input', 'LevelPrototyping'):
            shutil.copytree(engine / f'Templates/TemplateResources/High/{pack}/Content',
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
        (root / 'Config/DefaultEngine.ini').write_text('''[/Script/EngineSettings.GameMapsSettings]
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
''', encoding='utf-8')
        (root / 'Config/DefaultGame.ini').write_text('''[/Script/EngineSettings.GeneralProjectSettings]
ProjectName=BlindAssist Obstacle Lab
Description=Six pedestrian obstacle scenarios. Synthetic Development playground.
''', encoding='utf-8')
    script = repo / 'research/active/dtr-r0/unreal/build_obstacle_lab.py'
    log = root / 'Saved/Logs/build-lab.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cache = f'-LocalDataCachePath={root / "DerivedDataCache"}'
    if not (root / 'Content/ObstacleLab/ObstacleLab.umap').exists():
        subprocess.run([str(editor), str(project), '-run=pythonscript', f'-script={script}',
                        '-unattended', '-nullrhi', '-nosound', '-nop4', '-NoSplash',
                        f'-abslog={log}', '-ddc=NoShared', cache], check=True)
    receipt = root / 'Saved/lab-build.json'
    if not receipt.is_file():
        raise RuntimeError(f'Build did not produce a receipt; inspect {log}')
    print(receipt.read_text(encoding='utf-8'))
    print(f'PROJECT: {project}')
    if args.verify:
        verify = script.with_name('verify_obstacle_lab.py')
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
