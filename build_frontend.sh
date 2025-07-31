#!/bin/bash
# Build frontend and sync to Windows directory

echo "Cleaning dist directories to remove old build artifacts..."
rm -rf /home/asantanna/DNNE-UI-Frontend/dist/*
rm -rf /mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/dist/*
echo "✓ Dist directories cleaned"

echo "Building frontend..."
cd /home/asantanna/DNNE-UI-Frontend
npm run build

echo "Cleaning up unnecessary files from build..."
cd /home/asantanna/DNNE-UI-Frontend/dist

# Remove unused PrimeVue themes (only keep Aura which is used)
echo "  Removing unused themes..."
# rm -f assets/lib/@primevue/themes/umd/lara.min.js
# rm -f assets/lib/@primevue/themes/umd/material.min.js
# rm -f assets/lib/@primevue/themes/umd/nora.min.js

# Remove duplicate PrimeVue directory
echo "  Removing duplicate PrimeVue library..."
# rm -rf assets/lib/primevue

# Remove unused @primevue/icons directory
echo "  Removing unused PrimeVue icons..."
# rm -rf assets/lib/@primevue/icons

# Remove source map files (only needed for debugging production builds)
echo "  Removing source maps..."
# find . -name "*.map" -delete

echo "✓ Cleanup complete"

echo "Syncing to Windows directory..."
rsync -av /home/asantanna/DNNE-UI-Frontend/dist/ /mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/dist/

echo "Frontend build complete!"