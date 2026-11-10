import os
import subprocess
import time
from pathlib import Path

def load_env(path: Path):
    if not path.exists(): return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if '=' not in line: continue
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

# Load .env then .env.local (precedence)
load_env(Path('.env'))
load_env(Path('.env.local'))

# Kill existing on port 8000
try:
    subprocess.run(['lsof', '-ti:8000'], capture_output=True, check=True)
    subprocess.run('lsof -ti:8000 | xargs kill -9', shell=True)
    time.sleep(2)
except:
    pass

# Start uvicorn
log_file = open('/tmp/server_v10.log', 'w')
proc = subprocess.Popen(
    ['uv', 'run', 'uvicorn', 'shafi.api.app:create_app', '--factory', '--host', '0.0.0.0', '--port', '8000'],
    stdout=log_file,
    stderr=log_file,
    preexec_fn=os.setsid
)

with open('server_v10.pid', 'w') as f:
    f.write(str(proc.pid))

print(f"Server starting with PID={proc.pid}")
