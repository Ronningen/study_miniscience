all:
	for number in 1 2 3 4; do \
		clang++ -std=c++11 -L/usr/local/lib sample0$$number.cpp -lgmsh -o ../trash/sample0$$number; \
	done

clean:
	for number in 1 2 3 4; do \
        rm -f ../trash/sample0$$number; \
    done
	rm -f *msh