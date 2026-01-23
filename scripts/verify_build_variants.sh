#!/bin/bash

# Script to verify and compile all build variants needed for testing

echo "=========================================="
echo "Build Variant Verification and Compilation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the project root
if [ ! -f "Makefile" ]; then
    echo -e "${RED}ERROR: Makefile not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Load modules if on Leonardo
if command -v module &> /dev/null; then
    echo "Loading modules..."
    module purge
    module load openmpi/4.1.6--gcc--12.2.0
    echo ""
fi

# Define all build variants
declare -A VARIANTS
VARIANTS[parallel_o0]="build/parallel_o0"
VARIANTS[parallel_o1]="build/parallel_o1"
VARIANTS[parallel_noarch]="build/parallel_noarch"
VARIANTS[parallel]="build/parallel"
VARIANTS[serial_omp_o0]="build/serial_omp_o0"
VARIANTS[serial_omp_o1]="build/serial_omp_o1"
VARIANTS[serial_omp_noarch]="build/serial_omp_noarch"
VARIANTS[serial_omp]="build/serial_omp"

# Check which variants exist
echo "Checking existing executables..."
echo "----------------------------------------"
MISSING=()
EXISTING=()

for variant in "${!VARIANTS[@]}"; do
    exe="${VARIANTS[$variant]}"
    if [ -f "$exe" ]; then
        echo -e "${GREEN}✓${NC} $exe exists"
        EXISTING+=("$variant")
    else
        echo -e "${RED}✗${NC} $exe missing"
        MISSING+=("$variant")
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "Existing: ${GREEN}${#EXISTING[@]}${NC} variants"
echo -e "Missing: ${RED}${#MISSING[@]}${NC} variants"

if [ ${#MISSING[@]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All build variants are present!${NC}"
    exit 0
fi

echo ""
echo "=========================================="
echo "Compiling Missing Variants"
echo "=========================================="

# Compile missing variants
for variant in "${MISSING[@]}"; do
    echo ""
    echo -e "${YELLOW}Compiling $variant...${NC}"
    
    case "$variant" in
        parallel_o0)
            make parallel_o0
            ;;
        parallel_o1)
            make parallel_o1
            ;;
        parallel_noarch)
            make parallel_noarch
            ;;
        parallel)
            make parallel
            ;;
        serial_omp_o0)
            make serial_omp_o0
            ;;
        serial_omp_o1)
            make serial_omp_o1
            ;;
        serial_omp_noarch)
            make serial_omp_noarch
            ;;
        serial_omp)
            make serial_omp
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully compiled $variant${NC}"
    else
        echo -e "${RED}✗ Failed to compile $variant${NC}"
    fi
done

echo ""
echo "=========================================="
echo "Final Verification"
echo "=========================================="

# Re-check all variants
ALL_PRESENT=true
for variant in "${!VARIANTS[@]}"; do
    exe="${VARIANTS[$variant]}"
    if [ ! -f "$exe" ]; then
        echo -e "${RED}✗ Still missing: $exe${NC}"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = true ]; then
    echo ""
    echo -e "${GREEN}All build variants are now present!${NC}"
    echo ""
    echo "You can now run tests with different BUILD_VARIANT values:"
    echo "  - o0 (no optimization)"
    echo "  - o1 (level 1 optimization)"
    echo "  - noarch (Ofast without -march=native)"
    echo "  - ofast (Ofast with -march=native)"
    echo "  - ofast_omp_improved (same as ofast, different code)"
else
    echo ""
    echo -e "${RED}Some variants are still missing. Please check compilation errors above.${NC}"
    exit 1
fi
