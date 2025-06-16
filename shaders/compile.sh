glslc test.comp -o test.spv

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Shader compiled successfully."
else
    echo "Shader compilation failed."
    exit 1
fi