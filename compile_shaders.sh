# glslc shaders/compute.comp -o shaders/compute.spv
# glslc shaders/vertex.vert -o shaders/vertex.spv
# glslc shaders/fragment.frag -o shaders/fragment.spv

slangc shaders/compute.slang -profile spirv_1_6 -target spirv-assembly -o shaders/disassembled_compute.spv -entry main
slangc shaders/vertex.slang -profile spirv_1_6 -target spirv-assembly -o shaders/disassembled_vertex.spv -entry main 
slangc shaders/fragment.slang -profile spirv_1_6 -target spirv-assembly -o shaders/disassembled_fragment.spv -entry main 

# Change the OpSource line in the disassembled shader
# This is necessary because the OpSource line is not compatible with the Vulkan SDK version
# sed -i '14s/.*/OpSource Unknown 0/' shaders/disassembled_compute.spv
# sed -i '11s/.*/OpSource Unknown 0/' shaders/disassembled_vertex.spv
# sed -i '12s/.*/OpSource Unknown 0/' shaders/disassembled_fragment.spv
sed -i 's/OpSource Slang 1/OpSource Unknown 0/' shaders/disassembled_compute.spv
sed -i 's/OpSource Slang 1/OpSource Unknown 0/' shaders/disassembled_vertex.spv
sed -i 's/OpSource Slang 1/OpSource Unknown 0/' shaders/disassembled_fragment.spv


# Compile the disassembled shader back to SPIR-V binary
spirv-as shaders/disassembled_compute.spv -o shaders/compute.spv
spirv-as shaders/disassembled_vertex.spv -o shaders/vertex.spv
spirv-as shaders/disassembled_fragment.spv -o shaders/fragment.spv

rm shaders/disassembled_compute.spv
rm shaders/disassembled_vertex.spv
rm shaders/disassembled_fragment.spv

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Shader compiled successfully."
else
    echo "Shader compilation failed."
    exit 1
fi
