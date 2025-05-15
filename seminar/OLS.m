function [obj] = OLS(inc, edu, exper, exper2, beta)
    
    obj = 0;
    j = numel(inc);
    for i = 1:j
        obj = obj + (inc(i) - beta(1) -  beta(2)*edu(i) - beta(3)*exper(i) - beta(4)*exper2(i))^2;
    end
end
