@echo off

rem Check if --here flag is passed
if "%1"=="--here" (
    rem Run in current terminal, skip the --here argument
    python main.py --front-end-root ../DNNE-UI-Frontend/dist --listen 0.0.0.0 --agent-server-terminal %2 %3 %4 %5 %6 %7 %8 %9
) else (
    rem Launch in new Windows Terminal
    wt --profile "dnne_server" cmd /c python main.py --front-end-root ../DNNE-UI-Frontend/dist --listen 0.0.0.0 --agent-server-terminal %*
)
