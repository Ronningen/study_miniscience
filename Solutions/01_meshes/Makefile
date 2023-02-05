all:
	for number in 0 1 2; do \
		clang++ -std=c++11 -L/usr/local/lib stage$$number.cpp -lgmsh -o ../trash/stage$$number; \
	done

clean:
	for number in 0 1 2; do \
        rm -f ../trash/stage$$number; \
    done
	# rm -f ../trash/*msh