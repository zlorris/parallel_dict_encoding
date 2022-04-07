.SUFFIXES: .cu

main: *.cu
	nvcc *.cu --relocatable-device-code=true -o $@
