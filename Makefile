.SUFFIXES: .cu

main: *.cu *.hu utility/error.hu
	nvcc *.cu --relocatable-device-code=true -g -Xcompiler -rdynamic -Xcompiler -fopenmp -lm -lineinfo -o $@
