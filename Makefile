
default:
	# No default target to prevent accidents
	
data: clean
	python3 generate.py 

clean:
	rm -rf output/*

