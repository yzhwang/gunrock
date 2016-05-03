echo "Formatting code under $DIRECTORY/"
find -path ./externals -prune -name '*.h' -or -name '*.cpp' -or -name '*.cuh' -or -name '*.cu' | xargs clang-format -i
find -path ./externals -prune -name '*.h' -or -name '*.cpp' -or -name '*.cuh' -or -name '*.cu' | xargs sed -i -- 's/<< </<<</g' 
