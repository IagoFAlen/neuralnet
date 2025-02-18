rm -rf out/build/*
mkdir -p out/build
cmake -S . -B out/build
cmake --build out/build
cd out/build && make