function [error,grad]=MIML_error(x,Weights,errors)
   
    error=0;
    grad=0;
    [tempvalue,tempsize]=size(Weights);
    for i=1:tempsize
        error=error+Weights(i)*exp((2*errors(i)-1)*x);
        grad=grad+Weights(i)*exp((2*errors(i)-1)*x)*(2*errors(i)-1);
    end