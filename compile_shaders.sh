glslc shaders/compute.comp -o shaders/compute.spv
glslc shaders/vertex.vert -o shaders/vertex.spv
glslc shaders/fragment.frag -o shaders/fragment.spv

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Shader compiled successfully."
else
    echo "Shader compilation failed."
    exit 1
fi
