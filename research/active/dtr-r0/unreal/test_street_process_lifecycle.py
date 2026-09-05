"""Real child/socket cleanup and identity boundaries, with task-owned processes."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from street_process_lifecycle import TaskProcessTree,ProcessIdentity


class ProcessLifecycleTest(unittest.TestCase):
    def test_natural_exit_is_verified_without_a_signal(self):
        child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(.4)'])
        tree=TaskProcessTree(child,owner='natural-exit-unit')
        try:
            receipt=tree.cleanup(natural_timeout=3,poll_interval=.02)
            self.assertTrue(receipt['released'],receipt)
            self.assertEqual(receipt['actions'],[])
        finally:
            tree.cleanup(natural_timeout=0)

    def test_descendant_socket_is_released_and_sibling_survives(self):
        temporary=Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='street-process-test-',dir=temporary) as directory:
            root=Path(directory)
            server=root/'server.py'
            server.write_text("import json,os,socket,sys,time\nfrom pathlib import Path\ns=socket.socket();s.bind(('127.0.0.1',0));s.listen()\nPath(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'port':s.getsockname()[1]}))\ntime.sleep(60)\n")
            wrapper=root/'wrapper.py'
            wrapper.write_text("import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-B',sys.argv[1],sys.argv[2]])\ntime.sleep(60)\n")
            sibling=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)'])
            sibling_tree=TaskProcessTree(sibling,owner='unrelated-sibling-unit')
            child=subprocess.Popen([sys.executable,'-B',str(wrapper),str(server),str(root/'ready.json')])
            tree=TaskProcessTree(child,owner='wrapper-descendant-unit')
            try:
                deadline=time.monotonic()+10
                while not (root/'ready.json').exists() and time.monotonic()<deadline:
                    tree.capture()
                    time.sleep(.02)
                self.assertTrue((root/'ready.json').exists())
                ready=json.loads((root/'ready.json').read_text())
                tree.capture()
                self.assertIn(ready['pid'],[identity.pid for identity in tree.identities])
                self.assertNotIn(sibling.pid,[identity.pid for identity in tree.identities])
                with socket.socket() as probe:
                    self.assertEqual(0,probe.connect_ex(('127.0.0.1',ready['port'])))
                receipt=tree.cleanup(natural_timeout=.05,ports=[('127.0.0.1',ready['port'])],poll_interval=.02)
                self.assertTrue(receipt['released'],receipt)
                self.assertEqual(receipt['ports'][0]['state'],'CLOSED')
                self.assertIsNone(sibling.poll())
            finally:
                tree.cleanup(natural_timeout=0)
                sibling_tree.cleanup(natural_timeout=0)

    def test_reused_pid_identity_is_not_signalled(self):
        child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)'])
        tree=TaskProcessTree(child,owner='identity-unit')
        try:
            identity=tree.root_identity
            self.assertIsNotNone(identity)
            with patch.object(ProcessIdentity,'of',return_value=ProcessIdentity(identity.pid,identity.create_time+1)):
                self.assertEqual('PID_REUSED',tree._resolve(identity)[1])
                tree._signal_remaining('terminate')
                self.assertEqual(tree._actions,[])
            self.assertIsNone(child.poll())
        finally:
            self.assertTrue(tree.cleanup(natural_timeout=0)['released'])


if __name__=='__main__': unittest.main()
