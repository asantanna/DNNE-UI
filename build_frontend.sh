#!/bin/bash
# Build frontend and sync to Windows directory

echo "Building frontend with widget persistence fix..."
cd /home/asantanna/DNNE-UI-Frontend
npm run build

echo "Syncing to Windows directory..."
rsync -av /home/asantanna/DNNE-UI-Frontend/dist/ /mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/dist/

echo "Frontend build complete!"