all: test train.csv test.csv

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
