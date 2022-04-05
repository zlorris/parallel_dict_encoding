.SUFFIXES: .cu

main: *.cu
	nvcc *.cu -o $@
