classdef Reservoir < handle
    properties
        Wrec      % Reservoir-to-reservoir weight matrix [Nx x Nx]
        x         % Reservoir state vector: [Nx x 1]
        leakRate  % Leak rate of the leaky integrator model
    end

    methods
        function obj = Reservoir(Nx, networkDensity, spectralRadius, leakRate)
            obj.x = (2*rand(Nx, 1) - 1) * 0.1;
            obj.leakRate = leakRate;

            Wrec0 = zeros(Nx*Nx, 1);
            nNonzero = floor(Nx * Nx * networkDensity);
            Wrec0(1:nNonzero) = (2*rand(nNonzero, 1) - 1) * 0.1;
            Wrec0 = Wrec0(randperm(length(Wrec0)));
            Wrec = reshape(Wrec0, [Nx, Nx]);
            lambdaMax = abs(eigs(Wrec, 1, 'largestabs'));
            obj.Wrec = Wrec .* (spectralRadius / lambdaMax);
        end

        % Update reservoir state vector.
        function x = subsref(obj, x0)
            Win_u = x0(1).subs{1};
            obj.x = (1.0 - obj.leakRate) * obj.x ...
                + obj.leakRate * tanh(obj.Wrec*obj.x + Win_u);
            x = obj.x;
        end
    end
end
