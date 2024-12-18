classdef Output < handle
    properties
        Wout  % Reservoir-to-output weight matrix: [Ny x Nx*Nl]
    end

    methods
        function obj = Output(Nx, Ny, Nl)
            obj.Wout = normrnd(0, 1, Ny, Nx*Nl);
        end

        function Wout_x = subsref(obj, x)
            if strcmp(x(1).type, '()')
                Wout_x = obj.Wout * x(1).subs{1};
            elseif strcmp(x(1).subs, 'setWoutOpt')
                Wout_x = builtin('subsref', obj, x);
            end
        end

        % Set optimal Wout.
        function obj = setWoutOpt(obj, WoutOpt)
            obj.Wout = WoutOpt;
        end
    end
end
