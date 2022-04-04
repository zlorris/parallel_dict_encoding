.SUFFIXES: .cu

main: 
	nvcc *.cu -o $@
