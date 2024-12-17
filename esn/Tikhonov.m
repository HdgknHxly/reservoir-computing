classdef Tikhonov < handle
    properties
        Nx    % Number of nodes in the reservoir
        X_XT
        D_XT
        beta  % Regularization parameter
    end

    methods
        function obj = Tikhonov(Nx, Ny, beta)
            obj.Nx = Nx;
            obj.X_XT = zeros(Nx, Nx);
            obj.D_XT = zeros(Ny, Nx);
            obj.beta = beta;
        end

        % Update matrix for learning.
        function obj = update(obj, x, d)
            obj.X_XT = obj.X_XT + x * x';
            obj.D_XT = obj.D_XT + d * x';
        end

        % Calculate optimal Wout.
        function WoutOpt = getWoutOpt(obj)
           % XPseudoInv = inv(obj.X_XT + obj.beta*eye(obj.Nx));
           % WoutOpt = obj.D_XT * XPseudoInv;
           WoutOpt = obj.D_XT / (obj.X_XT + obj.beta*eye(obj.Nx));
        end
    end
end
