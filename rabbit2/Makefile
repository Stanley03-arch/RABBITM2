CC     ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra

build/rabbit2: src/rabbit2.c
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< -lm

test: build/rabbit2
	python3 tests/test_forward.py
	python3 tests/test_convert.py
	python3 tests/test_layout_big.py

clean:
	rm -rf build

.PHONY: test clean
