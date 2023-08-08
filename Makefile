.PHONY: all install-deps install dev-install uninstall clean

all: install-deps install

install-deps:
	pip install -r requirements.txt

install: clean
	pip install .

dev-install: clean
	pip install -e .

uninstall: clean
	pip uninstall minitensor
	$(RM) minitensor_cuda.cpython-39-x86_64-linux-gnu.so
	$(RM) minitensor_cudnn.cpython-39-x86_64-linux-gnu.so

clean:
	$(RM) -rf build minitensor.egg-info
	$(RM) -rf minitensor/__pycache__
