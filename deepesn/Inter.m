classdef Inter < handle
    properties
        Wil  % Weight matrix for inter-layer connections: [Nx x Nx]
    end

    methods
        function obj = Inter(Nx, Nl, interScale)
            % Uniform random numbers in the interval [-interScale, interScale]
            obj.Wil = cell(Nl, 1);
            for iLayer = 1:Nl
                obj.Wil{iLayer} = (2*rand(Nx, Nx) - 1) * interScale;
            end
        end

        function Wil_x = subsref(obj, x0)
            x = x0(1).subs{1}(1:end-1);
            iLayer = x0(1).subs{1}(end);

            Wil_x = obj.Wil{iLayer} * x;
        end
    end
end
