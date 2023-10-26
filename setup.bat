call git submodule update --init --recursive
call bash -c "./clean.sh"
call mkdir build
cd build
call cmake ..
