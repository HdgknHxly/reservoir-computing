classdef ESN < handle
    properties
        Nu         % Input dimension
        Nx         % Number of nodes in the reservoir
        Ny         % Output dimension
        Input
        Reservoir
        Output
    end
    
    methods
        function obj = ESN(Nu, Nx, Ny, inputScaling, networkDensity, spectralRadius, leakRate)
            obj.Nu = Nu;
            obj.Nx = Nx;
            obj.Ny = Ny;
            obj.Input = Input(Nu, Nx, inputScaling);
            obj.Reservoir = Reservoir(Nx, networkDensity, spectralRadius, leakRate);
            obj.Output = Output(Nx, Ny);
        end

        % Batch learning
        function train(obj, U, D, optimizer)
            for i = 1:length(U)
                u = U(i,:)';
                Win_u = obj.Input(u);
                
                % Reservoir state vector
                x = obj.Reservoir(Win_u);
                
                % Target data
                d = D(i,:)';

                % Optimizer
                optimizer.update(x, d);
            end
            % Set optimal Wout.
            WoutOpt = optimizer.getWoutOpt();
            obj.Output.setWoutOpt(WoutOpt);
        end

        % Prediction after batch learning
        function [XPred, YPred] = predict(obj, U)
            XPred = zeros(length(U), obj.Nx);
            YPred = zeros(length(U), obj.Ny);
            for i = 1:length(U)
                u = U(i,:)';
                Win_u = obj.Input(u);
                
                % Reservoir state vector
                x = obj.Reservoir(Win_u);
                XPred(i,:) = x';

                % Model output after training
                yPred = obj.Output(x);
                YPred(i,:) = yPred';
            end
        end

        % Prediction after batch learning (free run)
        function [XRun, YRun] = run(obj, U)
            XRun = zeros(length(U), obj.Nx);
            YRun = zeros(length(U), obj.Ny);
            u = U(1,:)';
            for i = 1:length(U)
                if (i > 1)
                    u(1:obj.Ny) = [];
                    u = [u; yRun'];
                end
                Win_u = obj.Input(u);

                % Reservoir state vector
                x = obj.Reservoir(Win_u);
                XRun(i,:) = x';

                % Model output after training
                yRun = obj.Output(x);
                YRun(i,:) = yRun';
            end
        end
    end
end
