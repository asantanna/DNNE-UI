@echo off

rem start "DNNE Server" /WAIT --profile "dnne_server" python main.py --front-end-root ../DNNE-UI-Frontend/dist --listen 0.0.0.0 --agent-server-terminal %*

wt --profile "dnne_server" cmd /c python main.py --front-end-root ../DNNE-UI-Frontend/dist --listen 0.0.0.0 --agent-server-terminal %*
