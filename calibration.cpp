#include <vector>
//#include <pybind11/pybind11.h>

class Position{
    double x_left = 0.5;
    double x_right = 0.5;
    double y_left = 0.5;
    double y_right = 0.5;

    Position(double Xleft, double Xright, double Yleft, double Yright){
        x_left = Xleft;
        x_right = Xright;
        y_left = Yleft;
        y_right = Yright;
    }

    std::vector<double> mead_pos(){
        double resX = (x_left - x_right) / 2;
        double resY = (y_left - x_right) / 2;
        std::vector<double> res = {resX, resY};
        return res;
    }
};


double data_acquisition(double left_x, double left_y, double right_x, double right_y){

}


int main(){

}