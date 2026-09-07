"""Detached Windows worker with per-job logs and terminal receipts."""
import argparse
import base64
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(os.environ.get('BLINDASSIST_WORKER_ROOT', Path(__file__).resolve().parent.parent))

def ps(text):
    encoded = base64.b64encode(text.encode('utf-16le')).decode('ascii')
    return subprocess.run([str(ROOT.parent / 'tools/pwsh/pwsh.exe'), '-NoProfile', '-EncodedCommand', encoded], check=True, capture_output=True, text=True)

def quote(value):
    return "'" + str(value).replace("'", "''") + "'"

def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.job):
        parser.error('Use a unique alphanumeric job name')
    folder = ROOT / 'artifacts/evidence/jobs' / args.job
    if not args.worker:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            parser.error('A command is required')
        folder.mkdir(parents=True, exist_ok=False)
        write(folder / 'request.json', dict(command=command, cwd=os.getcwd(),
              source=os.environ.get('BLINDASSIST_SOURCE'), createdUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              resumable=False, scope='engineering or explicitly registered Development only',
              environment={key: value for key, value in os.environ.items() if key.upper() in ['PATH','TEMP','TMP','VIRTUAL_ENV','PYTHONUTF8','OMP_NUM_THREADS','MKL_NUM_THREADS','PIP_CACHE_DIR','HF_HOME','TORCH_HOME','GRADLE_USER_HOME','ANDROID_HOME','ANDROID_USER_HOME','ANDROID_SDK_ROOT','JAVA_HOME','CUDA_PATH','CUPY_CACHE_DIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME','MPLCONFIGDIR','YOLO_CONFIG_DIR','NPM_CONFIG_CACHE','UE_ENGINE_ROOT','UE-LOCALDATACACHEPATH'] or key.startswith('BLINDASSIST_')}))
        task_name = 'BlindAssist-' + args.job
        arguments = subprocess.list2cmdline([str(Path(__file__).resolve()), '--job', args.job, '--worker'])
        ps(f"$ErrorActionPreference='Stop'; $a=New-ScheduledTaskAction -Execute {quote(sys.executable)} -Argument {quote(arguments)} -WorkingDirectory {quote(os.getcwd())}; $p=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType S4U -RunLevel Highest; $s=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero); Register-ScheduledTask -TaskName {quote(task_name)} -Action $a -Principal $p -Settings $s | Out-Null; Start-ScheduledTask -TaskName {quote(task_name)}")
        write(folder / 'launcher.json', dict(taskName=task_name))
        print(folder)
        return
    request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
    os.environ.update(request['environment'])
    write(folder / 'progress.json', dict(stage='running', pid=os.getpid(), eta='unknown'))
    result = dict(startedUtc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        command = list(request['command'])
        executable = shutil.which(command[0])
        if executable is None:
            raise FileNotFoundError(f"Command is unavailable in the selected profile: {command[0]}")
        command[0] = executable
        result['executable'] = executable
        with (folder / 'stdout.log').open('ab') as out, (folder / 'stderr.log').open('ab') as err:
            child = subprocess.Popen(command, cwd=request['cwd'], stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            write(folder / 'progress.json', dict(stage='running', pid=os.getpid(), childPid=child.pid, eta='unknown'))
            result['exitCode'] = child.wait()
        result['status'] = 'success' if result['exitCode'] == 0 else 'failed'
    except Exception as exc:
        result.update(status='failed', error=str(exc))
    result['finishedUtc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(folder / 'result.json', result)
    write(folder / 'progress.json', dict(stage=result['status'], eta='complete'))
    try:
        ps(f"Unregister-ScheduledTask -TaskName {quote('BlindAssist-' + args.job)} -Confirm:$false -ErrorAction Stop")
    except subprocess.CalledProcessError as exc:
        write(folder / 'cleanup-error.json', dict(error=str(exc)))

if __name__ == '__main__':
    main()
