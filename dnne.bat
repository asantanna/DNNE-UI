@echo off
start "DNNE Server" /WAIT python main.py --front-end-root ../DNNE-UI-Frontend/dist --listen 0.0.0.0 --agent-server-terminal %*

