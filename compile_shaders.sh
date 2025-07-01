# glslc shaders/compute.comp -o shaders/compute.spv
# glslc shaders/vertex.vert -o shaders/vertex.spv
# glslc shaders/fragment.frag -o shaders/fragment.spv

# slangc shaders/compute.slang -profile spirv_1_6 -target spirv -o shaders/compute.spv -entry computeMain -emit-spirv-directly -fvk-use-entrypoint-name

# Compile the shaders using Slang to SPIR-V assembly
slangc shaders/compute.slang -profile spirv_1_5 -target spirv-asm -o shaders/disassembled_compute.spvasm -entry computeMain -emit-spirv-directly -fvk-use-entrypoint-name
slangc shaders/vertex.slang -profile spirv_1_5 -target spirv-asm -o shaders/disassembled_vertex.spvasm -entry vertexMain -emit-spirv-directly -fvk-use-entrypoint-name
slangc shaders/fragment.slang -profile spirv_1_5 -target spirv-asm -o shaders/disassembled_fragment.spvasm -entry fragmentMain -emit-spirv-directly -fvk-use-entrypoint-name

# Change the OpSource line in the disassembled shader. This is necessary because the OpSource line is not compatible with the Vulkan SDK version
sed -i 's/OpSource Slang 1/OpSource Unknown 0/' shaders/disassembled_compute.spvasm
sed -i 's/OpSource Slang 1/OpSource Unknown 0/' shaders/disassembled_vertex.spvasm
sed -i 's/OpSource Slang 1/OpSource Unknown 0/' shaders/disassembled_fragment.spvasm

# Compile the disassembled shader back to SPIR-V binary
spirv-as shaders/disassembled_compute.spvasm -o shaders/compute.spv
spirv-as shaders/disassembled_vertex.spvasm -o shaders/vertex.spv
spirv-as shaders/disassembled_fragment.spvasm -o shaders/fragment.spv

# Clean up the disassembled files
rm shaders/disassembled_compute.spvasm
rm shaders/disassembled_vertex.spvasm
rm shaders/disassembled_fragment.spvasm

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Shader compiled successfully."
else
    echo "Shader compilation failed."
    exit 1
fi
