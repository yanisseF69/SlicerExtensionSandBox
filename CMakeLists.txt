cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerSlicerGPT)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SlicerSlicerGPT")
set(EXTENSION_CONTRIBUTORS "Yanisse Ferhaoui (Institut Pascal | UCA | UCBL)")
set(EXTENSION_DESCRIPTION "A 3D Slicer extension that integrates a local AI chatbot to provide context-aware help using your scene and official documentation.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SlicerSlicerGPT.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SlicerSlicerGPT/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerGPT)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
