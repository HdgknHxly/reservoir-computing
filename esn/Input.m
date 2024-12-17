classdef Input < handle
    properties
        Win  % Input-to-reservoir weight matrix: [Nx x Nu]
    end

    methods
        function obj = Input(Nu, Nx, inputScaling)
            % Uniform random numbers in the interval [-inputScaling, inputScaling]
            obj.Win = (2*rand(Nx, Nu) - 1) * inputScaling;
        end

        function Win_u = subsref(obj, u)
            Win_u = obj.Win * u(1).subs{1};
        end
    end
end
