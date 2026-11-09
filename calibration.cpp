#include <array>
#include <pybind11/pybind11.h>

class Position{
    std::array<float, 2> left;
    std::array<float, 2> right;

    Position(std::array<float, 2> l, std::array<float, 2> r){
        left = l;
        right = r;
    }

    std::array<float, 2> mead_pos(){
        float resX = (left[0] - right[0]) / 2;
        float resY = (left[1] - right[1]) / 2;
        return {resX, resY};
    }
};


double data_acquisition(double left_x, double left_y, double right_x, double right_y){

}


int main(){

}