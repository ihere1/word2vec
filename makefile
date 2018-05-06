CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: tree cal processtxt caltop

tree : tree.c
	$(CC) tree.c -o tree $(CFLAGS)
cal : cal.c
	$(CC) cal.c -o cal -mcmodel=large $(CFLAGS)
processtxt: processtxt.c
	$(CC) processtxt.c -o processtxt -mcmodel=large $(CFLAGS)
caltop: caltop.cpp
	g++ caltop.cpp -o caltop

	chmod +x *.sh

clean:
	rm -rf tree cal processtxt caltop
