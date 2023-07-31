import os

def CURRENT_DIR():
    cwd = os.getcwd()
    return os.chdir(cwd[:(cwd.index("Eliza")+5)])

CURRENT_DIR()

print(os.getcwd())

from src.config import Config

Config.run_uvicorn()