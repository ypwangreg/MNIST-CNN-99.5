all: test train.csv test.csv

%.o: %.c
	g++ -g -c $<
%.o: %.cpp
	g++ -g -c $<
test: test.o
	g++ -o $@ $<  -lpthread
%.csv: %.csv.zip
	unzip $<
	touch $@

#train.csv: train.csv.zip
##	unzip $<
#	touch $@
#test.csv: test.csv.zip
#	unzip $<
#	touch $@
run: 
	./test

mcnn: webgui.c CNN.c
	gcc -o $@ $^ -lm -lpthread -O3

webgui.c: webgui.tar.gz
	tar xvf $<
	touch $@
vncsrv: vncsrv.cpp
	g++ -DTEST_VNCSRV -g -o $@ $< -lvncserver 
vncsrv1: vncsrv1.o
	g++ -o $@ $< -lvncserver 
tests1: tests1.o
	g++ -o $@ $<  -lpthread
#  sudo apt install libtbb-dev
tests2: tests2.cpp
	g++ -std=c++17 -o $@ $<  -lpthread -ltbb
tpnc: test_pthncore.o
	g++ -o $@ $<  -lpthread
testpp: testpp.o pthpool.o
	g++ -o $@ $^  -lpthread
