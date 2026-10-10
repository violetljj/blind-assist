"""USB Wi-Fi setup for the selected Atom; secrets never enter status or logs."""
from __future__ import annotations

import ipaddress
import json
import re
import subprocess
import threading
import time


class SetupError(ValueError):
    """Only fixed, safe setup messages may be returned to the viewer."""


def netsh(arguments):
    result = subprocess.run(['netsh', 'wlan', *arguments], capture_output=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=8)
    if result.returncode:
        raise ValueError('无法读取电脑 Wi-Fi 配置，请手动输入密码')
    # Hidden netsh can use a different language/code page from the parent shell.
    return [result.stdout.decode(codec, errors='replace') for codec in ('utf-8', 'gbk', 'utf-16le')]


def profiles():
    names = []
    current = None
    for output in netsh(['show', 'profiles']):
        values = re.findall(r'^\s*(?:All User Profile|所有用户配置文件)\s*:\s*(.+)$', output, re.M)
        if values:
            names = [value.rstrip('\r') for value in values]
            break
    for output in netsh(['show', 'interfaces']):
        match = re.search(r'^\s*(?:Profile|配置文件)\s*:\s*(.+)$', output, re.M)
        if match:
            current = match.group(1).rstrip()
            break
    return dict(profiles=names, current=current)


def saved_password(profile):
    if not isinstance(profile, str) or not profile or len(profile) > 128 or any(c in profile for c in '\r\n\x00'):
        raise ValueError('请选择电脑已保存的网络')
    for output in netsh(['show', 'profile', 'name='+profile, 'key=clear']):
        match = re.search(r'^\s*(?:Key Content|关键内容)\s*:\s*(.+)$', output, re.M)
        if match:
            return match.group(1).rstrip('\r')
    raise ValueError('该网络没有可复用的密码，请手动输入')


def validate_credentials(ssid, password):
    if (not isinstance(ssid, str) or not isinstance(password, str) or
            not 1 <= len(ssid.encode('utf-8')) <= 32 or not 8 <= len(password.encode('utf-8')) <= 63 or
            any(c in ssid+password for c in '\t\r\n\x00')):
        raise ValueError('网络名需为1–32字节，密码需为8–63字节，不能含换行或制表符')


class WifiSetup:
    def __init__(self, stop, camera, port, expected_serial):
        self.stop, self.camera, self.port = stop, camera, port
        self.expected_serial = expected_serial.replace(':', '').lower()
        self.lock = threading.Lock()
        self.worker = None
        self.state = dict(status='填写相机要连接的 2.4 GHz 网络', busy=False, saved=False,
                          connected=False, error=None, port_released=True)

    def snapshot(self):
        with self.lock:
            result = dict(self.state)
        if result['saved'] and not result['busy'] and self.camera and self.camera.snapshot()['available']:
            result.update(connected=True, status='相机画面已接通')
        return result

    def request(self, data):
        ssid, password = data.get('ssid'), data.get('password', '')
        profile = data.get('password_profile')
        validate_credentials(ssid, 'temporary' if profile else password)
        if profile and (not isinstance(profile, str) or len(profile) > 128 or any(c in profile for c in '\r\n\x00')):
            raise ValueError('请选择电脑已保存的网络')
        with self.lock:
            if self.state['busy']:
                raise ValueError('正在配置，请等待本次完成')
            self.state = dict(status='正在通过 USB 配置相机', busy=True, saved=False,
                              connected=False, error=None, port_released=True, ssid=ssid)
            self.worker = threading.Thread(target=self.configure, args=(ssid, password, profile), daemon=True)
            self.worker.start()

    def configure(self, ssid, password, profile=None):
        import serial
        from serial.tools import list_ports
        try:
            # Resolve only the explicitly selected profile in the worker, outside state locks.
            if profile:
                try:
                    password = saved_password(profile)
                    validate_credentials(ssid, password)
                except ValueError:
                    raise SetupError('无法复用该网络的密码，请手动输入') from None
            matches = [p for p in list_ports.comports() if p.device.lower() == self.port.lower()
                       and (p.serial_number or '').replace(':', '').lower() == self.expected_serial]
            if len(matches) != 1:
                raise SetupError('未找到指定 Atom USB 设备，请检查连接')
            if self.camera:
                self.camera.configuration_started()
            with serial.Serial(port=None, baudrate=115200, timeout=.2, write_timeout=3) as link:
                link.dtr = True
                link.rts = False
                link.port = self.port
                link.open()
                with self.lock:
                    self.state['port_released'] = False
                if self.stop.wait(.5):
                    return
                command = ('WIFI\t'+ssid+'\t'+password+'\n').encode('utf-8')
                if link.write(command) != len(command):
                    raise SetupError('USB 配置未完整发送，请重试')
                link.flush()
                del command, password
                deadline = time.monotonic()+35
                while not self.stop.is_set() and time.monotonic() < deadline:
                    try:
                        event = json.loads(link.readline())
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get('type') == 'wifi_config_saved':
                        with self.lock:
                            self.state.update(saved=True, status='配置已保存，正在等待联网')
                    elif event.get('type') == 'wifi_config_rejected':
                        raise SetupError('相机拒绝了网络配置，请检查输入')
                    elif event.get('type') == 'network_ready' and event.get('role') == 'camera':
                        try:
                            address = ipaddress.IPv4Address(event.get('ip', ''))
                            local = any(address in ipaddress.ip_network(n) for n in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))
                        except ValueError:
                            local = False
                        with self.lock:
                            if self.state['saved'] and local:
                                self.state.update(connected=True, status='相机已联网，正在等待画面')
                                if self.camera:
                                    self.camera.network_ready(str(address))
                                break
                with self.lock:
                    if self.stop.is_set():
                        self.state.update(status='配置已结束')
                    elif self.state['saved'] and not self.state['connected']:
                        self.state.update(status='配置已保存，连接待确认；请检查网络名、密码和 2.4 GHz 信号')
                    elif not self.state['saved']:
                        raise SetupError('未收到相机配置回执，请检查 USB 连接')
        except SetupError as exc:
            with self.lock:
                self.state.update(error=str(exc), status='配置未完成')
        except Exception:
            # Arbitrary transport exceptions can contain the command: never echo.
            with self.lock:
                self.state.update(error='USB 配置失败，请检查相机连接后重试', status='配置未完成')
        finally:
            if self.camera:
                self.camera.configuration_finished()
            with self.lock:
                self.state.update(busy=False, port_released=True)

    def close(self):
        if self.worker:
            self.worker.join(timeout=5)
