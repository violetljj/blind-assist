"""Provision the identity-selected XIAO over USB; never print or persist a password."""
from __future__ import annotations
import argparse
import getpass
import ipaddress
import json
import sys
import time
from m3_wifi_setup import saved_password, validate_credentials

EXPECTED_SERIAL = '98a316f7881c'

def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='通过USB为H3 Wi-Fi固件设置本地网络，不刷写')
    parser.add_argument('--port',default='COM5')
    parser.add_argument('--ssid',required=True)
    parser.add_argument('--password-profile',help='复用电脑已保存的指定Wi-Fi密码；不提供时安全提示输入')
    args=parser.parse_args(argv)
    import serial
    from serial.tools import list_ports
    matches=[p for p in list_ports.comports() if p.device.lower()==args.port.lower() and
             (p.serial_number or '').replace(':','').lower()==EXPECTED_SERIAL]
    if len(matches)!=1: parser.error('未找到指定XIAO身份，请检查USB连接，未发送网络密码')
    password=saved_password(args.password_profile) if args.password_profile else getpass.getpass('Wi-Fi密码：')
    validate_credentials(args.ssid,password)
    link=serial.Serial(port=None,baudrate=115200,timeout=.3,write_timeout=3)
    link.dtr=False;link.rts=False;link.port=args.port
    try:
        link.open()
        command=('WIFI\t'+args.ssid+'\t'+password+'\n').encode('utf-8')
        link.write(command);link.flush();del command,password
        deadline=time.monotonic()+40
        saved=False
        while time.monotonic()<deadline:
            try: event=json.loads(link.readline())
            except (ValueError,UnicodeError): continue
            if not isinstance(event,dict): continue
            if event.get('type')=='wifi_config_rejected': raise ValueError('固件拒绝网络配置')
            if event.get('type')=='wifi_config_saved': saved=True;print('网络配置已保存，等待联网…',flush=True)
            if saved and event.get('type')=='network_ready' and event.get('role')=='tof':
                address=ipaddress.IPv4Address(event['ip'])
                if not any(address in ipaddress.ip_network(n) for n in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16')):
                    raise ValueError('固件没有返回局域网地址')
                print(json.dumps({'connected':True,'tof_host':str(address),'tcp_port':3335},ensure_ascii=False))
                return 0
        raise ValueError('没有收到联网回执，请核对2.4GHz网络；配置及原固件数据保留')
    finally: link.close()

if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,OSError) as exc:
        print('配置停止：'+str(exc),file=sys.stderr);raise SystemExit(1)
