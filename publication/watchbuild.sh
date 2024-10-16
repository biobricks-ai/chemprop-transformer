#!/bin/bash

# Function to run the build script
run_build() {
    echo "Changes detected. Running build script..."
    ./build.sh
}

# Initial build
run_build

# Watch for changes
while true; do
    inotifywait -r -e modify,create,delete .
    run_build
done
